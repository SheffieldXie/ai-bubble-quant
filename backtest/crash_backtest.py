"""
Historical Crash Backtest Module.
Simulates buying 0DTE put options before major market crashes.

Tests the hypothesis: "If we had bought puts before each crash, how much would we have made?"
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CrashEvent:
    """A historical market crash event."""
    name: str
    date: str
    index_before: float
    index_after: float
    days_to_bottom: int
    max_drop_pct: float
    description: str


# ── Major Historical Crashes ───────────────────────────────────────────

HISTORICAL_CRASHES = [
    CrashEvent(
        name="2000 Dotcom Burst",
        date="2000-03-10",
        index_before=5048,  # Nasdaq peak
        index_after=3227,   # ~18 months later
        days_to_bottom=548,
        max_drop_pct=-36.0,
        description="📌 互联网泡沫破裂\n\n背景：1995-2000年科技股狂热，.com公司无盈利也能上市，PE普遍过百。纳斯达克从1000飙至5048。\n\n触发：2000年3月美联储连续加息，微软反垄断案败诉，信心崩塌。\n\n暴跌：两年内纳指暴跌78%（5048→1114），Amazon跌超90%，思科跌86%，VIX从20飙至60，科技板块市值蒸发约5万亿美元。",
    ),
    CrashEvent(
        name="2008 Financial Crisis",
        date="2008-09-15",
        index_before=4100,  # S&P 500 peak
        index_after=2100,
        days_to_bottom=517,
        max_drop_pct=-49.0,
        description="📌 全球金融危机\n\n背景：次级房贷泡沫膨胀，MBS/CDO层层加杠杆，华尔街把风险藏在表外。雷曼9月15日申请破产（6390亿资产，史上最大），AIG濒临倒闭。\n\n暴跌：标普500从4100跌至666（-84%），VIX飙至89创历史极值，TED利差扩大至464bp，LIBOR市场冻结。\n\n后果：各国央行联合降息+QE1救市，耗时18个月触底。 Dodd-Frank法案出台。",
    ),
    CrashEvent(
        name="2015 China Crash",
        date="2015-06-12",
        index_before=5178,  # 上证指数
        index_after=2850,
        days_to_bottom=72,
        max_drop_pct=-45.0,
        description="📌 A股杠杆牛转疯熊\n\n背景：2014-2015年杠杆牛市，场外配资规模超2万亿，上证从2000飙到5178，创业板PE突破140倍。\n\n触发：6月监管严查场外配资，去杠杆引发踩踏，千股跌停成为日常。\n\n暴跌：72天上证暴跌45%（5178→2850），创业板跌65%，两市融资盘强平超万亿。国家队2万亿入市救市，IPO暂停，限制做空。",
    ),
    CrashEvent(
        name="2018 Q4 Flash Crash",
        date="2018-10-03",
        index_before=2930,  # S&P 500
        index_after=2346,
        days_to_bottom=81,
        max_drop_pct=-20.0,
        description="📌 美联储加息 + 贸易战\n\n背景：2018年美联储连续4次加息至2.5%，同时进行缩表。中美贸易战持续升级，苹果警告营收下滑。\n\n暴跌：标普500从2930跌至2346（-20%），进入技术性熊市。12月24日VIX飙至36，科技股领跌，FAANG全线暴跌。\n\n反转：鲍威尔12月26日鸽派转向（\"耐心\"措辞），市场V型反弹，标普Q1反弹13%。",
    ),
    CrashEvent(
        name="2020 COVID Crash",
        date="2020-02-19",
        index_before=3386,  # S&P 500
        index_after=2237,
        days_to_bottom=33,
        max_drop_pct=-34.0,
        description="📌 新冠疫情全球暴跌\n\n背景：2020年2月新冠疫情爆发，3月多国封城，经济活动骤停。\n\n暴跌：标普500从3386跌至2237（-34%），仅23个交易日触发5次熔断（史上首次），VIX飙至82.7超越2008年，原油暴跌甚至3月30日出现负油价。\n\n政策：美联储3天内降息至0+无限量QE，财政部2万亿刺激。3月23日触底后V型反弹，标普8月即创新高。",
    ),
    CrashEvent(
        name="2022 Tech Selloff",
        date="2022-01-03",
        index_before=16200,  # Nasdaq
        index_after=10088,
        days_to_bottom=349,
        max_drop_pct=-33.0,
        description="📌 激进加息 + 科技股杀估值\n\n背景：2022年美联储全年加息425bp（0→4.25%），叠加俄乌战争、高通胀。10年美债收益率从1.5%飙至4.3%。\n\n暴跌：纳斯达克从16200跌至10088（-38%），标普跌25%。成长股PE从30x压缩到20x，ARKK基金暴跌67%，Meta一天跌26%。\n\n影响：科技股估值逻辑重构，从\"增长优先\"转向\"盈利为王\"。",
    ),
    CrashEvent(
        name="2024 Aug Yen Carry Unwind",
        date="2024-08-05",
        index_before=4300,  # S&P 500
        index_after=3950,
        days_to_bottom=3,
        max_drop_pct=-8.0,
        description="📌 日元套息交易平仓闪崩\n\n背景：7月日本央行意外加息15bp至0.25%，日元从160急升至142，全球万亿级套息交易大规模平仓。\n\n暴跌：日经单日暴跌12.4%（史上第二大），标普从5700跌至4950（-13%），VIX飙至65，英伟达一天跌10%，苹果跌7%。\n\n反转：日本央行释放鸽派信号，全球央行紧急安抚，一周内快速反弹修复。",
    ),
]


@dataclass
class BacktestResult:
    """Result of one backtested crash scenario."""
    crash: CrashEvent
    put_strike: float
    days_before_entry: int  # How many days before crash we bought
    premium_paid: float  # In index points
    days_held: int
    index_at_entry: float
    index_at_exit: float
    exit_premium: float  # In index points
    pnl_points: float
    pnl_pct: float
    contract_multiplier: int = 100

    @property
    def pnl_money(self) -> float:
        return self.pnl_points * self.contract_multiplier - 30  # ~30 yuan round-trip fees


class CrashBacktester:
    """
    Backtests the strategy: "Buy OTM puts X days before each crash."

    This is a thought experiment — we're testing whether our signal system
    would have caught these crashes in time.
    """

    def __init__(
        self,
        iv: float = 0.20,
        r: float = 0.02,
        q: float = 0.02,
        contract_multiplier: int = 100,
    ):
        self.iv = iv
        self.r = r
        self.q = q
        self.multiplier = contract_multiplier

    def _bs_put_price(self, S: float, K: float, T: float) -> float:
        """Black-Scholes put price (in index points)."""
        if T <= 0:
            return max(K - S, 0)
        sigma = self.iv
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        nd1_neg = 0.5 * (1 + math.erf(-d1 / math.sqrt(2)))
        nd2_neg = 0.5 * (1 + math.erf(-d2 / math.sqrt(2)))
        price = K * np.exp(-self.r * T) * nd2_neg - S * np.exp(-self.q * T) * nd1_neg
        return max(price, 0.01)

    def backtest_crash(
        self,
        crash: CrashEvent,
        days_before: int = 5,
        otm_pct: float = 0.05,
    ) -> BacktestResult:
        """
        Simulate buying a put before a crash.

        Args:
            crash: The crash event to backtest
            days_before: How many days before the crash we buy the put
            otm_pct: How far OTM the put is (e.g., 0.05 = 5% below current)
        """
        S = crash.index_before  # Index at crash peak
        K = S * (1 - otm_pct)   # OTM put strike
        T_entry = days_before / 365  # Time to expiry when we buy

        # Premium paid at entry
        premium_entry = self._bs_put_price(S, K, T_entry)

        # Index after crash (worst case)
        S_after = crash.index_after
        drop_pct = (S_after - S) / S

        # Time remaining at exit (assume we hold to bottom or expiry)
        days_to_bottom = min(crash.days_to_bottom, 30)  # Cap at 30 days for 0DTE
        T_exit = max(0, T_entry - abs(drop_pct) * 365 * 0.1)  # Rough estimate

        # Premium at exit (if we hold to crash bottom)
        if T_exit > 0:
            premium_exit = self._bs_put_price(S_after, K, T_exit)
        else:
            premium_exit = max(K - S_after, 0)

        # P&L
        pnl_points = premium_exit - premium_entry
        pnl_pct = pnl_points / premium_entry * 100 if premium_entry > 0 else 0

        return BacktestResult(
            crash=crash,
            put_strike=K,
            days_before_entry=days_before,
            premium_paid=premium_entry,
            days_held=days_before + abs(int(drop_pct * 365)),
            index_at_entry=S,
            index_at_exit=S_after,
            exit_premium=premium_exit,
            pnl_points=pnl_points,
            pnl_pct=pnl_pct,
            contract_multiplier=self.multiplier,
        )

    def run_full_backtest(
        self,
        days_before: int = 5,
        otm_pct: float = 0.05,
    ) -> list[BacktestResult]:
        """Run backtest on all historical crashes."""
        results = []
        for crash in HISTORICAL_CRASHES:
            try:
                result = self.backtest_crash(crash, days_before, otm_pct)
                results.append(result)
            except Exception as e:
                logger.warning(f"Backtest failed for {crash.name}: {e}")
        return results

    def format_report(self, results: list[BacktestResult]) -> str:
        """Format backtest results as a report."""
        lines = []
        lines.append("=" * 100)
        lines.append("📉 历史暴跌回测 —— 如果每次暴跌前5天买5%虚值认沽期权")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'事件':<25} {'买入点位':>8} {'行权价':>8} {'成本(点)':>8} {'暴跌后':>8} "
                     f"{'期权价值':>8} {'盈亏(点)':>8} {'盈亏%':>8} {'盈亏(元)':>10}")
        lines.append("-" * 100)

        total_pnl = 0
        wins = 0
        losses = 0

        for r in results:
            pnl_money = r.pnl_money
            total_pnl += pnl_money
            if r.pnl_pct > 0:
                wins += 1
            else:
                losses += 1

            emoji = "🟢" if r.pnl_pct > 0 else "🔴"
            lines.append(
                f"{emoji} {r.crash.name:<23} {r.index_at_entry:>8.0f} {r.put_strike:>8.0f} "
                f"{r.premium_paid:>7.1f}点 {r.index_at_exit:>8.0f} {r.exit_premium:>7.1f}点 "
                f"{r.pnl_points:>7.1f}点 {r.pnl_pct:>+7.0f}% {pnl_money:>+9.0f}元"
            )

        lines.append("-" * 100)
        lines.append(f"总计: {len(results)}次交易 | 盈利{wins}次 | 亏损{losses}次 | 总盈亏: {total_pnl:+,.0f}元")
        lines.append(f"胜率: {wins/len(results)*100:.0f}% | 平均盈亏: {total_pnl/len(results):+,.0f}元/次")
        lines.append("")
        lines.append("⚠️  注意: 这是理想化回测。实际操作中:")
        lines.append("   1. 我们不可能精确在暴跌前5天入场")
        lines.append("   2. IV在暴跌前可能很低（期权便宜），也可能已经升高")
        lines.append("   3. 流动性问题可能导致实际成交价偏离理论价")
        lines.append("   4. 这是用美股指数模拟，A股实际走势会有差异")
        lines.append("=" * 100)

        return "\n".join(lines)
