"""
Fundamental analysis module.
Evaluates fundamental factors that indicate bubble conditions.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """Fundamental analysis for bubble detection."""

    @staticmethod
    def analyze_pe_anomaly(trailing_pe: float, forward_pe: float) -> dict:
        """
        Compare trailing vs forward PE.
        Large gap (trailing >> forward) suggests market is pricing in massive future growth.

        Returns:
            dict with pe_gap assessment
        """
        if trailing_pe is None or forward_pe is None or forward_pe <= 0:
            return {"trailing_pe": trailing_pe, "forward_pe": forward_pe, "gap_pct": None, "assessment": "insufficient data"}

        gap_pct = ((trailing_pe - forward_pe) / forward_pe) * 100

        if gap_pct > 50:
            assessment = "VERY HIGH expectations — bubble risk"
        elif gap_pct > 25:
            assessment = "HIGH expectations — watch closely"
        elif gap_pct > 0:
            assessment = "MODERATE expectations"
        else:
            assessment = "Forward PE >= Trailing (earnings expected to decline)"

        return {
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "gap_pct": round(gap_pct, 1),
            "assessment": assessment,
        }

    @staticmethod
    def analyze_revenue_vs_price(
        price_history: pd.DataFrame,
        revenue_growth_rate: Optional[float] = None,
    ) -> dict:
        """
        Compare price appreciation vs revenue growth.
        If price is running far ahead of revenue, it's a warning sign.
        """
        if price_history.empty or "adj_close" not in price_history.columns:
            return {"price_return_1y": None, "price_return_3y": None, "revenue_growth": revenue_growth, "assessment": "insufficient data"}

        prices = price_history["adj_close"]
        current = prices.iloc[-1]

        # 1-year return
        if len(prices) >= 252:
            price_1y_ago = prices.iloc[-252]
            return_1y = ((current - price_1y_ago) / price_1y_ago) * 100
        else:
            return_1y = None

        # 3-year return (roughly)
        if len(prices) >= 756:
            price_3y_ago = prices.iloc[-756]
            return_3y = ((current - price_3y_ago) / price_3y_ago) * 100
        else:
            return_3y = None

        # Compare price growth to revenue growth
        if return_1y is not None and revenue_growth_rate is not None:
            if return_1y > revenue_growth_rate * 2:
                assessment = "Price far outpacing revenue — RED FLAG"
            elif return_1y > revenue_growth_rate * 1.5:
                assessment = "Price outpacing revenue — CAUTION"
            else:
                assessment = "Price and revenue growth aligned"
        else:
            assessment = "Insufficient data for comparison"

        return {
            "price_return_1y": round(return_1y, 1) if return_1y is not None else None,
            "price_return_3y": round(return_3y, 1) if return_3y is not None else None,
            "revenue_growth_rate": revenue_growth_rate,
            "assessment": assessment,
        }

    @staticmethod
    def analyze_insider_activity(insider_txns: pd.DataFrame) -> dict:
        """
        Analyze insider trading patterns.
        Heavy insider selling = bearish signal.
        """
        if insider_txns is None or insider_txns.empty:
            return {"net_insider_activity": None, "assessment": "no data"}

        # yfinance provides: Shares, Value, Transaction (Buy/Sale)
        if "Transaction" not in insider_txns.columns:
            return {"net_insider_activity": None, "assessment": "insufficient data"}

        # Separate buys and sells
        buys = insider_txns[insider_txns["Transaction"].str.contains("Buy", case=False, na=False)]
        sells = insider_txns[insider_txns["Transaction"].str.contains("Sale|Stock", case=False, na=False)]

        buy_value = buys["Value"].sum() if "Value" in buys.columns else 0
        sell_value = sells["Value"].sum() if "Value" in sells.columns else 0

        net = buy_value - sell_value

        if net < -1_000_000_000:
            assessment = "MASSIVE insider selling — BEARISH"
        elif net < -100_000_000:
            assessment = "Significant insider selling — BEARISH"
        elif net < -10_000_000:
            assessment = "Moderate insider selling — CAUTION"
        elif net > 10_000_000:
            assessment = "Net insider buying — BULLISH"
        else:
            assessment = "Neutral insider activity"

        return {
            "insider_buys": round(buy_value, 0) if buy_value else 0,
            "insider_sells": round(sell_value, 0) if sell_value else 0,
            "net_activity": round(net, 0),
            "assessment": assessment,
        }

    @staticmethod
    def valuation_red_flags(metrics: dict) -> list[str]:
        """
        Check for classic valuation red flags.

        Args:
            metrics: dict from fetcher.get_key_metrics()

        Returns:
            List of warning strings
        """
        flags = []

        pe = metrics.get("trailing_pe")
        if pe and pe > 50:
            flags.append(f"Very high trailing PE: {pe}")

        f_pe = metrics.get("forward_pe")
        if f_pe and f_pe > 40:
            flags.append(f"High forward PE: {f_pe}")

        ps = metrics.get("price_to_sales")
        if ps and ps > 15:
            flags.append(f"Extremely high P/S ratio: {ps}")

        pb = metrics.get("price_to_book")
        if pb and pb > 15:
            flags.append(f"High P/B ratio: {pb}")

        rev_growth = metrics.get("revenue_growth")
        if pe and rev_growth and pe > rev_growth * 100 * 2:
            flags.append(f"PE ({pe}) far exceeds revenue growth ({rev_growth*100:.1f}%) — PEG > 2")

        return flags
