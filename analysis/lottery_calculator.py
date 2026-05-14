"""
CFFEX Option Calculator (彩票计算器).
Calculates cost, payoff, and probability for CFFEX index options (IO/MO).

Supports:
- IO (沪深300股指期权, 乘数100)
- MO (中证1000股指期权, 乘数100)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Contract Specifications ────────────────────────────────────────────

@dataclass
class ContractSpec:
    name: str
    underlying: str
    multiplier: int  # 合约乘数 (元/点)
    min_tick: float   # 最小变动价位 (点)
    commission: float # 手续费 (元/张)
    trading_hours: tuple
    current_index: float = 0.0  # 当前指数点位


CONTRACTS = {
    "IO": ContractSpec(
        name="沪深300股指期权",
        underlying="沪深300",
        multiplier=100,
        min_tick=0.2,
        commission=15.0,
        trading_hours=(9.5, 15.0),
    ),
    "MO": ContractSpec(
        name="中证1000股指期权",
        underlying="中证1000",
        multiplier=100,
        min_tick=0.2,
        commission=15.0,
        trading_hours=(9.5, 15.0),
    ),
    "IH": ContractSpec(
        name="上证50股指期权",
        underlying="上证50",
        multiplier=100,
        min_tick=0.2,
        commission=15.0,
        trading_hours=(9.5, 15.0),
    ),
}


# ── Black-Scholes for Index Options ────────────────────────────────────

class IndexOptionPricer:
    """
    Black-Scholes pricer for European index options.
    Uses dividend yield to model index carry.
    """

    def __init__(
        self,
        S: float,          # Current index level
        r: float = 0.02,   # Risk-free rate (China ~2%)
        q: float = 0.02,   # Dividend yield (CSI ~2%)
        sigma: float = 0.20, # Implied volatility
        T: float = 7/365,  # Time to expiry (default: 7 days)
    ):
        self.S = S
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _d1(self, K: float) -> float:
        if self.sigma * math.sqrt(self.T) == 0:
            return 0
        return (math.log(self.S / K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * math.sqrt(self.T))

    def _d2(self, K: float) -> float:
        return self._d1(K) - self.sigma * math.sqrt(self.T)

    def put_price(self, K: float) -> float:
        """European put option price (in index points)."""
        if self.T <= 0:
            return max(K - self.S, 0)
        d1 = self._d1(K)
        d2 = self._d2(K)
        price = K * math.exp(-self.r * self.T) * self._norm_cdf(-d2) - \
                self.S * math.exp(-self.q * self.T) * self._norm_cdf(-d1)
        return max(price, 0.01)  # Minimum 0.01 points

    def call_price(self, K: float) -> float:
        """European call option price (in index points)."""
        if self.T <= 0:
            return max(self.S - K, 0)
        d1 = self._d1(K)
        d2 = self._d2(K)
        price = self.S * math.exp(-self.q * self.T) * self._norm_cdf(d1) - \
                K * math.exp(-self.r * self.T) * self._norm_cdf(d2)
        return max(price, 0.01)

    def put_delta(self, K: float) -> float:
        d1 = self._d1(K)
        return math.exp(-self.q * self.T) * (self._norm_cdf(d1) - 1)

    def prob_itm(self, K: float) -> float:
        """Probability of finishing ITM."""
        if self.T <= 0:
            return 1.0 if self.S < K else 0.0
        d2 = self._d2(K)
        return self._norm_cdf(-d2)


# ── Lottery Calculator ─────────────────────────────────────────────────

class LotteryCalculator:
    """
    Analyzes put option "lottery tickets" for CFFEX index options.
    """

    def __init__(
        self,
        contract: str = "MO",
        current_index: float = 0.0,
        iv: float = 0.20,
        days_to_expiry: int = 7,
    ):
        self.contract = CONTRACTS.get(contract)
        if not self.contract:
            raise ValueError(f"Unknown contract: {contract}")

        self.current_index = current_index
        self.iv = iv
        self.days_to_expiry = days_to_expiry

        self.pricer = IndexOptionPricer(
            S=current_index,
            r=0.02,
            q=0.02,
            sigma=iv,
            T=days_to_expiry / 365,
        )

    def _generate_strikes(self, otm_pct_range: tuple = (0.02, 0.15)) -> list[float]:
        """
        Generate put strikes from slightly OTM to deeply OTM.

        Args:
            otm_pct_range: (min_otm%, max_otm%) below current index
        """
        min_k = self.current_index * (1 - otm_pct_range[1])
        max_k = self.current_index * (1 - otm_pct_range[0])

        # Round to nearest 10 or 50 depending on index level
        if self.current_index > 5000:
            step = 50
        elif self.current_index > 3000:
            step = 20
        else:
            step = 10

        strikes = []
        k = round(min_k / step) * step
        max_k_rounded = round(max_k / step) * step
        while k <= max_k_rounded:
            strikes.append(k)
            k += step

        return strikes

    def analyze_strike(self, K: float) -> dict:
        """
        Analyze a single put strike.

        Returns dict with cost, payoff, probability, and risk metrics.
        """
        spec = self.contract
        m = spec.multiplier

        # Option price in points
        premium_points = self.pricer.put_price(K)
        premium_cost = premium_points * m  # 实际成本 (元)
        total_cost = premium_cost + spec.commission * 2  # 开+平仓手续费

        # Greeks
        delta = self.pricer.put_delta(K)
        prob_itm = self.pricer.prob_itm(K)

        # Breakeven: K - premium
        breakeven = K - premium_points

        # Payoff scenarios (if index drops)
        drops = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
        scenarios = []
        for pct_drop in drops:
            s_future = self.current_index * (1 - pct_drop)
            put_payoff = max(K - s_future, 0)
            option_value_future = put_payoff  # At expiry
            pnl = (option_value_future - premium_points) * m - spec.commission * 2
            pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
            scenarios.append({
                "drop_pct": pct_drop,
                "index_level": round(s_future, 0),
                "option_value": round(option_value_future * m, 0),
                "pnl": round(pnl, 0),
                "pnl_pct": round(pnl_pct, 0),
            })

        # Max profit (index goes to 0)
        max_profit = (K - premium_points) * m - spec.commission * 2
        max_profit_pct = max_profit / total_cost * 100 if total_cost > 0 else 0

        # Risk/reward
        risk_reward = max_profit / total_cost if total_cost > 0 else 0

        return {
            "strike": K,
            "otm_pct": round((self.current_index - K) / self.current_index * 100, 1),
            "premium_points": round(premium_points, 2),
            "premium_cost": round(premium_cost, 0),
            "total_cost": round(total_cost, 0),
            "delta": round(delta, 4),
            "prob_itm": round(prob_itm * 100, 1),
            "breakeven": round(breakeven, 1),
            "max_profit": round(max_profit, 0),
            "max_profit_pct": round(max_profit_pct, 0),
            "risk_reward": round(risk_reward, 1),
            "scenarios": scenarios,
        }

    def analyze_all(self) -> list[dict]:
        """Analyze all OTM put strikes."""
        strikes = self._generate_strikes()
        return [self.analyze_strike(K) for K in strikes]

    def format_report(self) -> str:
        """Generate a readable report of all put strikes."""
        results = self.analyze_all()
        spec = self.contract
        m = spec.multiplier

        lines = []
        lines.append("=" * 100)
        lines.append(f"🎰 {spec.name} 认沽期权 彩票计算器")
        lines.append("=" * 100)
        lines.append(f"当前指数: {self.current_index:.0f} | IV: {self.iv*100:.0f}% | 到期天数: {self.days_to_expiry}天")
        lines.append(f"合约乘数: {m}元/点 | 手续费: {spec.commission}元/张(单边)")
        lines.append("")
        lines.append(f"{'行权价':>8} {'虚值%':>6} {'权利金':>8} {'成本(元)':>9} {'胜率%':>6} {'Delta':>7} "
                     f"{'跌1%':>10} {'跌3%':>10} {'跌5%':>10} {'跌8%':>10} {'最大收益':>10}")
        lines.append("-" * 100)

        for r in results:
            drops = {s["drop_pct"]: s["pnl"] for s in r["scenarios"]}
            pnl_1 = drops.get(0.01, 0)
            pnl_3 = drops.get(0.03, 0)
            pnl_5 = drops.get(0.05, 0)
            pnl_8 = drops.get(0.08, 0)

            lines.append(
                f"{r['strike']:>8.0f} {r['otm_pct']:>5.1f}% "
                f"{r['premium_points']:>6.1f}点 {r['total_cost']:>8.0f}元 "
                f"{r['prob_itm']:>5.1f}% {r['delta']:>7.3f} "
                f"{pnl_1:>9.0f}元 {pnl_3:>9.0f}元 {pnl_5:>9.0f}元 {pnl_8:>9.0f}元 "
                f"{r['max_profit']:>9.0f}元"
            )

        lines.append("")
        lines.append("💡 说明: 成本 = 权利金 + 开平仓手续费 | 胜率 = 到期实值概率")
        lines.append("   跌X%列 = 如果指数当天跌X%，你这张期权的盈亏（元）")
        lines.append("   最大收益 = 假设指数跌到0（极端情况）")
        lines.append("=" * 100)

        return "\n".join(lines)

    def format_detailed(self, strike: float) -> str:
        """Detailed breakdown for a specific strike."""
        r = self.analyze_strike(strike)

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"📊 详细分析: {self.contract.name} Put, 行权价 {r['strike']:.0f}")
        lines.append(f"{'='*60}")
        lines.append(f"  虚值程度: {r['otm_pct']:.1f}% (指数需要跌{r['otm_pct']:.1f}%才到行权价)")
        lines.append(f"  权利金: {r['premium_points']:.1f}点 = {r['premium_cost']:.0f}元")
        lines.append(f"  总成本: {r['total_cost']:.0f}元 (含手续费)")
        lines.append(f"  Delta: {r['delta']:.4f} (指数跌100点, Put涨约{abs(r['delta']*100):.0f}点)")
        lines.append(f"  到期实值概率: {r['prob_itm']:.1f}%")
        lines.append(f"  盈亏平衡点: {r['breakeven']:.0f} (指数需要跌到{r['breakeven']:.0f}以下才盈利)")
        lines.append(f"  最大可能收益: {r['max_profit']:.0f}元 ({r['max_profit_pct']:.0f}%回报)")
        lines.append("")
        lines.append("  💰 暴跌情景模拟:")
        for s in r["scenarios"]:
            emoji = "🟢" if s["pnl"] > 0 else "🔴"
            lines.append(f"    {emoji} 跌{s['drop_pct']*100:.0f}% → 指数{s['index_level']:.0f} → "
                         f"期权价值{s['option_value']:.0f}元 → 盈亏{s['pnl']:.0f}元 ({s['pnl_pct']:+.0f}%)")
        lines.append(f"{'='*60}")

        return "\n".join(lines)
