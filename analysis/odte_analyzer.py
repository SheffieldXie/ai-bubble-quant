"""
0DTE (Zero Days to Expiration) Options Analysis Module.
Analyzes SPY/QQQ options chains to find optimal short put trades.

CRITICAL: This module does NOT guarantee profits. 0DTE options are
extremely high-risk instruments. Past performance does not predict future results.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Black-Scholes Model ───────────────────────────────────────────────

class BlackScholes:
    """Standard Black-Scholes option pricing and Greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if sigma * math.sqrt(T) == 0:
            return 0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative normal distribution (approximation)."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        price = K * math.exp(-r * T) * BlackScholes.norm_cdf(-d2) - \
                S * BlackScholes.norm_cdf(-d1)
        return max(price, 0)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        price = S * BlackScholes.norm_cdf(d1) - \
                K * math.exp(-r * T) * BlackScholes.norm_cdf(d2)
        return max(price, 0)

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Put option delta (negative for long puts)."""
        if T <= 0:
            return -1 if S < K else 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return BlackScholes.norm_cdf(d1) - 1

    @staticmethod
    def implied_volatility(
        market_price: float, S: float, K: float, T: float, r: float,
        option_type: str = "put", max_iter: int = 100, tol: float = 1e-6
    ) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson."""
        sigma = 0.3  # Initial guess
        for _ in range(max_iter):
            if option_type == "put":
                price = BlackScholes.put_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.call_price(S, K, T, r, sigma)

            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            # Vega (sensitivity to vol)
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            vega = S * math.sqrt(T) * BlackScholes._norm_pdf(d1) * 0.01

            if vega == 0:
                break

            sigma = sigma - diff / vega
            sigma = max(0.01, min(sigma, 5.0))

        return sigma

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal probability density."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ── 0DTE Options Strategy Analyzer ──────────────────────────────────────

class ZeroDTEAnalyzer:
    """
    Analyzes 0DTE options strategies for SPY/QQQ.

    The fundamental trade-off:
    - Deep ITM puts: High win rate, low payout (10-30% ROI)
    - ATM puts: Moderate win rate, moderate payout (50-100% ROI)
    - OTM puts: Low win rate, massive payout (200-1000%+ ROI)

    There is no free lunch. Every point of win rate costs points of payout.
    """

    # SPY/QQQ typical characteristics
    DEFAULT_RISK_FREE_RATE = 0.053  # Current Fed funds rate
    TRADING_HOURS_PER_DAY = 6.5  # 9:30 AM - 4:00 PM ET

    def __init__(
        self,
        current_price: float,
        implied_volatility: float,
        risk_free_rate: float = None,
    ):
        """
        Args:
            current_price: Current SPY/QQQ price
            implied_volatility: Current IV as decimal (e.g. 0.15 for 15%)
            risk_free_rate: Annual risk-free rate (default: 5.3%)
        """
        self.S = current_price
        self.IV = implied_volatility
        self.r = risk_free_rate or self.DEFAULT_RISK_FREE_RATE
        logger.info(f"ZeroDTEAnalyzer: S={self.S}, IV={self.IV*100:.1f}%, r={self.r*100:.1f}%")

    def _time_to_expiry(self, hours_remaining: float = 6.5) -> float:
        """Convert hours remaining to years for BS model.
        252 trading days * 6.5 hours = 1638 trading hours per year.
        """
        trading_hours_per_year = 252 * self.TRADING_HOURS_PER_DAY
        return hours_remaining / trading_hours_per_year

    def analyze_put_strikes(
        self,
        current_price: float = None,
        iv: float = None,
        hours_remaining: float = 6.5,
        strike_range_pct: float = 0.05,  # +/- 5% from current
    ) -> pd.DataFrame:
        """
        Analyze all put strikes within range.

        Returns DataFrame with columns:
        - strike, price, delta, prob_itm, prob_profit,
          max_profit, max_loss, risk_reward, breakeven
        """
        S = current_price or self.S
        sigma = iv or self.IV
        T = self._time_to_expiry(hours_remaining)

        if T <= 0:
            logger.warning("Options already expired")
            return pd.DataFrame()

        # Generate strikes
        min_strike = S * (1 - strike_range_pct)
        max_strike = S * (1 + strike_range_pct)
        # Round to nearest 0.5 (SPY strike increment)
        min_strike = round(min_strike * 2) / 2
        max_strike = round(max_strike * 2) / 2

        results = []
        strike = min_strike
        while strike <= max_strike:
            price = BlackScholes.put_price(S, strike, T, self.r, sigma)
            delta = BlackScholes.put_delta(S, strike, T, self.r, sigma)

            # Probability ITM (approximation via delta)
            prob_itm = abs(delta)  # |delta| ≈ probability of finishing ITM

            # More precise probability using d2
            d2 = BlackScholes.d2(S, strike, T, self.r, sigma)
            prob_itm_precise = BlackScholes.norm_cdf(-d2)

            # For a LONG put:
            # Max profit = strike - premium paid (if stock goes to 0)
            # Max loss = premium paid (100% loss if OTM at expiry)
            # Breakeven = strike - premium
            intrinsic = max(strike - S, 0)
            time_value = price - intrinsic

            max_profit = strike - price  # If stock goes to $0
            max_loss = price  # 100% of premium paid
            risk_reward = (max_profit / max_loss) if max_loss > 0 else 0
            breakeven = strike - price

            # Probability of profit (stock must drop below breakeven)
            d2_be = BlackScholes.d2(S, breakeven, T, self.r, sigma)
            prob_profit = BlackScholes.norm_cdf(-d2_be)

            # Expected value
            expected_value = prob_profit * max_profit - (1 - prob_profit) * max_loss

            results.append({
                "strike": round(strike, 2),
                "option_price": round(price, 4),
                "delta": round(delta, 4),
                "prob_itm": round(prob_itm_precise * 100, 2),
                "prob_profit": round(prob_profit * 100, 2),
                "max_profit": round(max_profit, 4),
                "max_loss": round(max_loss, 4),
                "risk_reward_ratio": round(risk_reward, 2),
                "breakeven": round(breakeven, 2),
                "expected_value": round(expected_value, 4),
                "moneyness": self._classify_moneyness(S, strike),
            })

            strike = round(strike + 0.5, 2)

        df = pd.DataFrame(results)
        return df

    def _classify_moneyness(self, S: float, K: float) -> str:
        """Classify strike as ITM/ATM/OTM."""
        pct = (K - S) / S
        if pct > 0.005:
            return "ITM"
        elif pct > -0.005:
            return "ATM"
        else:
            return "OTM"

    def find_optimal_strategy(
        self,
        criterion: str = "sharpe",
        min_prob_profit: float = 0.0,
        max_prob_profit: float = 1.0,
    ) -> dict:
        """
        Find the optimal put strike based on criterion.

        Criteria:
        - 'sharpe': Best risk-adjusted return (expected value / risk)
        - 'win_rate': Highest probability of profit
        - 'payout': Highest risk/reward ratio (max payout)
        - 'balanced': Optimal balance between win rate and payout
        """
        df = self.analyze_put_strikes()
        if df.empty:
            return {"error": "No data to analyze"}

        # Filter by probability constraints
        df_filtered = df[
            (df["prob_profit"] >= min_prob_profit * 100) &
            (df["prob_profit"] <= max_prob_profit * 100)
        ]
        if df_filtered.empty:
            return {"error": f"No strikes in probability range {min_prob_profit}-{max_prob_profit}"}

        best = None

        if criterion == "sharpe":
            # Best risk-adjusted: highest expected value relative to max loss
            df_filtered = df_filtered[df_filtered["expected_value"] > 0]
            if df_filtered.empty:
                return {"error": "No positive EV strikes available"}
            df_filtered["sharpe"] = df_filtered["expected_value"] / df_filtered["max_loss"]
            idx = df_filtered["sharpe"].idxmax()
            best = df_filtered.loc[idx]

        elif criterion == "win_rate":
            # Highest probability of profit
            idx = df_filtered["prob_profit"].idxmax()
            best = df_filtered.loc[idx]

        elif criterion == "payout":
            # Highest risk/reward
            df_filtered = df_filtered[df_filtered["prob_profit"] >= 10]  # At least 10% win rate
            if df_filtered.empty:
                idx = df["risk_reward_ratio"].idxmax()
                best = df.loc[idx]
            else:
                idx = df_filtered["risk_reward_ratio"].idxmax()
                best = df_filtered.loc[idx]

        elif criterion == "balanced":
            # Optimize: maximize (win_rate * log(payout))
            df_filtered["score"] = (
                df_filtered["prob_profit"] *
                np.log1p(df_filtered["risk_reward_ratio"])
            )
            idx = df_filtered["score"].idxmax()
            best = df_filtered.loc[idx]

        return {
            "criterion": criterion,
            "strike": best["strike"],
            "moneyness": best["moneyness"],
            "option_price": best["option_price"],
            "delta": best["delta"],
            "prob_itm": best["prob_itm"],
            "prob_profit": best["prob_profit"],
            "max_profit": best["max_profit"],
            "max_loss": best["max_loss"],
            "risk_reward_ratio": best["risk_reward_ratio"],
            "breakeven": best["breakeven"],
            "expected_value": best["expected_value"],
        }

    def compare_all_strategies(self) -> str:
        """Generate a comparison report of all strategy criteria."""
        criteria = ["sharpe", "win_rate", "payout", "balanced"]
        results = {}

        for c in criteria:
            try:
                results[c] = self.find_optimal_strategy(criterion=c)
            except Exception as e:
                results[c] = {"error": str(e)}

        lines = []
        lines.append("=" * 80)
        lines.append("0DTE OPTIONS STRATEGY COMPARISON")
        lines.append("=" * 80)
        lines.append(f"Underlying: ${self.S:.2f} | IV: {self.IV*100:.1f}%")
        lines.append("")

        for c in criteria:
            r = results[c]
            if "error" in r:
                lines.append(f"\n{c.upper()}: {r['error']}")
                continue

            lines.append(f"\n{'─' * 40}")
            lines.append(f"Strategy: {c.upper()}")
            lines.append(f"{'─' * 40}")
            lines.append(f"  Strike: ${r['strike']:.2f} ({r['moneyness']})")
            lines.append(f"  Option Price: ${r['option_price']:.4f}")
            lines.append(f"  Delta: {r['delta']:.4f}")
            lines.append(f"  Win Rate (Prob Profit): {r['prob_profit']:.1f}%")
            lines.append(f"  Prob ITM: {r['prob_itm']:.1f}%")
            lines.append(f"  Max Profit: ${r['max_profit']:.4f}")
            lines.append(f"  Max Loss: ${r['max_loss']:.4f}")
            lines.append(f"  Risk/Reward: {r['risk_reward_ratio']:.2f}x")
            lines.append(f"  Breakeven: ${r['breakeven']:.2f}")
            lines.append(f"  Expected Value: ${r['expected_value']:.4f}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("WARNING: These are theoretical probabilities based on Black-Scholes.")
        lines.append("Actual results depend on realized volatility, market conditions,")
        lines.append("and timing. 0DTE options can lose 100% of premium rapidly.")
        lines.append("=" * 80)

        return "\n".join(lines)

    def monte_carlo_simulation(
        self,
        strike: float,
        hours_remaining: float = 6.5,
        n_sims: int = 10000,
    ) -> dict:
        """
        Monte Carlo simulation of put option outcome.

        Simulates price paths using geometric Brownian motion
        and calculates probability of profit.
        """
        S = self.S
        T = self._time_to_expiry(hours_remaining)
        sigma = self.IV

        # Option cost
        option_cost = BlackScholes.put_price(S, strike, T, self.r, sigma)

        # Simulate terminal stock prices
        np.random.seed(42)
        z = np.random.standard_normal(n_sims)
        S_T = S * np.exp((self.r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)

        # Put payoff at expiration
        put_payoff = np.maximum(strike - S_T, 0)
        pnl = put_payoff - option_cost

        # Statistics
        prob_profit = np.mean(pnl > 0) * 100
        avg_profit = np.mean(pnl[pnl > 0]) if np.any(pnl > 0) else 0
        avg_loss = np.mean(pnl[pnl < 0]) if np.any(pnl < 0) else 0
        win_rate = prob_profit
        avg_win = avg_profit
        avg_lose = avg_loss
        max_win = np.max(pnl)
        max_lose = np.min(pnl)
        expected_pnl = np.mean(pnl)

        # Percentiles
        p5 = np.percentile(pnl, 5)
        p25 = np.percentile(pnl, 25)
        p50 = np.percentile(pnl, 50)
        p75 = np.percentile(pnl, 75)
        p95 = np.percentile(pnl, 95)

        return {
            "strike": strike,
            "option_cost": round(option_cost, 4),
            "n_simulations": n_sims,
            "win_rate": round(win_rate, 2),
            "expected_pnl": round(expected_pnl, 4),
            "avg_win": round(avg_win, 4),
            "avg_lose": round(avg_lose, 4),
            "max_win": round(max_win, 4),
            "max_lose": round(max_lose, 4),
            "p5": round(p5, 4),
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p75": round(p75, 4),
            "p95": round(p95, 4),
        }

    def simulate_all_strikes(
        self,
        hours_remaining: float = 6.5,
        n_sims: int = 10000,
    ) -> pd.DataFrame:
        """Run Monte Carlo on all strikes and compare."""
        df = self.analyze_put_strikes(hours_remaining=hours_remaining)
        if df.empty:
            return pd.DataFrame()

        mc_results = []
        for _, row in df.iterrows():
            strike = row["strike"]
            mc = self.monte_carlo_simulation(strike, hours_remaining, n_sims)
            mc_results.append(mc)

        mc_df = pd.DataFrame(mc_results)
        return mc_df
