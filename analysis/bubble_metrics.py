"""
Bubble metrics calculation.

Implements established bubble detection indicators:
1. Shiller PE (CAPE) for tech / AI sector
2. Market Cap to GDP (Buffett Indicator)
3. Sector concentration risk (Nvidia dominance)
4. CAPEX intensity (AI infrastructure spending vs revenue)
5. Insider selling patterns
6. Retail sentiment indicators
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BubbleMetrics:
    """Calculates and tracks bubble-related metrics."""

    def __init__(self):
        logger.info("BubbleMetrics initialized")

    # ── 1. Shiller PE (CAPE) Approximation ─────────────────────────────

    @staticmethod
    def shiller_pe(price_history: pd.DataFrame, earnings_history: pd.DataFrame) -> Optional[float]:
        """
        Cyclically Adjusted P/E Ratio.
        Uses 10-year average of real earnings.

        For individual stocks we use a simplified version:
        trailing PE adjusted by earnings growth stability.

        Args:
            price_history: DataFrame with 'adj_close'
            earnings_history: DataFrame with EPS data (columns: date, eps)

        Returns:
            Shiller PE value, or None if insufficient data
        """
        if price_history.empty or earnings_history.empty:
            return None

        current_price = price_history["adj_close"].iloc[-1]

        # Use 10-year average earnings if available, else 3-year
        recent_earnings = earnings_history.tail(40)  # 10 quarters
        if len(recent_earnings) < 8:
            recent_earnings = earnings_history.tail(12)  # fallback to 3 years

        avg_earnings = recent_earnings.mean()
        if avg_earnings is None or avg_earnings <= 0:
            return None

        # Annualize quarterly EPS
        annual_eps = avg_earnings * 4
        cape = current_price / annual_eps

        return round(cape, 2)

    # ── 2. Market Cap to GDP (Buffett Indicator) ────────────────────────

    @staticmethod
    def market_cap_to_gdp(
        total_market_cap: float,
        gdp: float,
    ) -> Optional[float]:
        """
        Total US stock market cap / US GDP.
        Historical context:
          - < 50%: significantly undervalued
          - 75-90%: fair value
          - 100-120%: overvalued
          - > 150%: bubble territory

        Args:
            total_market_cap: Total US equity market cap in USD
            gdp: US GDP in USD (same units)

        Returns:
            Ratio as a percentage
        """
        if gdp <= 0:
            return None
        ratio = (total_market_cap / gdp) * 100
        return round(ratio, 2)

    @staticmethod
    def interpret_buffett_indicator(ratio: float) -> str:
        """Human-readable interpretation of the Buffett Indicator."""
        if ratio < 50:
            return "significantly undervalued"
        elif ratio < 75:
            return "undervalued"
        elif ratio < 100:
            return "fair value"
        elif ratio < 120:
            return "overvalued"
        elif ratio < 150:
            return "significantly overvalued"
        else:
            return "EXTREME BUBBLE TERRITORY"

    # ── 3. Sector Concentration Risk ────────────────────────────────────

    @staticmethod
    def concentration_risk(
        top_stock_weight_in_index: float,
        top_5_weight_in_index: float,
    ) -> dict:
        """
        Measures how concentrated the market is in AI-related names.

        Args:
            top_stock_weight_in_index: e.g. NVDA's weight in SPY
            top_5_weight_in_index: Top 5 AI stocks' combined weight

        Returns:
            dict with risk metrics and assessment
        """
        result = {
            "top_stock_weight": top_stock_weight_in_index,
            "top_5_weight": top_5_weight_in_index,
            "concentration_score": 0.0,
            "assessment": "normal",
        }

        # Scoring logic
        score = 0.0
        if top_stock_weight_in_index > 0.07:
            score += 0.3
        elif top_stock_weight_in_index > 0.05:
            score += 0.15

        if top_5_weight_in_index > 0.25:
            score += 0.4
        elif top_5_weight_in_index > 0.20:
            score += 0.2

        result["concentration_score"] = min(score, 1.0)

        if score > 0.6:
            result["assessment"] = "EXTREME concentration"
        elif score > 0.4:
            result["assessment"] = "high concentration"
        elif score > 0.2:
            result["assessment"] = "moderate concentration"
        else:
            result["assessment"] = "normal"

        return result

    # ── 4. CAPEX Intensity (AI Infrastructure Spending) ─────────────────

    @staticmethod
    def capex_to_revenue_ratio(capex: float, revenue: float) -> Optional[float]:
        """
        AI capex as a fraction of revenue for hyperscalers.
        When capex >> revenue growth, it signals potential over-investment.

        Args:
            capex: Capital expenditures (USD)
            revenue: Total revenue (USD)

        Returns:
            Ratio, or None
        """
        if revenue <= 0:
            return None
        return round(capex / revenue, 4)

    # ── 5. Price Momentum / Deviation from Moving Averages ──────────────

    @staticmethod
    def price_deviation_from_ma(
        price_history: pd.DataFrame,
        ma_periods: list[int] = [50, 100, 200],
    ) -> dict:
        """
        How far current price is from key moving averages.
        Large deviations = potential mean reversion candidates.

        Returns:
            dict of {period: deviation_pct}
        """
        if price_history.empty or "adj_close" not in price_history.columns:
            return {}

        prices = price_history["adj_close"]
        current = prices.iloc[-1]
        result = {}

        for period in ma_periods:
            if len(prices) < period:
                continue
            ma = prices.rolling(window=period).mean().iloc[-1]
            if ma > 0:
                deviation = ((current - ma) / ma) * 100
                result[f"ma_{period}"] = round(deviation, 2)

        return result

    # ── 6. RSI Extremes ────────────────────────────────────────────────

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return None

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi_value = 100 - (100 / (1 + rs.iloc[-1]))

        return round(rsi_value, 2)

    # ── Composite Bubble Score ──────────────────────────────────────────

    @staticmethod
    def compute_bubble_score(
        metrics: dict,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Compute an overall bubble score from 0 to 1.

        Args:
            metrics: dict with individual metric values
            weights: optional weighting dict (defaults to equal weight)

        Returns:
            Bubble score [0, 1] where 1 = maximum bubble condition
        """
        default_weights = {
            "shiller_pe_normalized": 0.25,
            "buffett_indicator_normalized": 0.20,
            "concentration_score": 0.15,
            "capex_intensity_normalized": 0.15,
            "insider_selling_normalized": 0.10,
            "retail_sentiment_normalized": 0.15,
        }

        w = weights or default_weights

        score = 0.0
        total_weight = 0.0

        for key, weight in w.items():
            value = metrics.get(key)
            if value is not None:
                # Ensure value is in [0, 1]
                value = max(0.0, min(1.0, float(value)))
                score += value * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(score / total_weight, 4)

    @staticmethod
    def interpret_bubble_score(score: float) -> str:
        """Translate numeric score to actionable label."""
        if score >= 0.85:
            return "EXTREME — Strong short candidate"
        elif score >= 0.70:
            return "HIGH — Consider building short positions"
        elif score >= 0.50:
            return "MODERATE — Monitor closely"
        elif score >= 0.30:
            return "LOW — Caution warranted"
        else:
            return "MINIMAL — No bubble signals detected"
