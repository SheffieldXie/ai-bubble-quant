"""
Market sentiment analysis module.
Tracks sentiment indicators that correlate with bubble conditions.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes market sentiment data for bubble signals."""

    @staticmethod
    def vix_analysis(vix_history: pd.DataFrame) -> dict:
        """
        Analyze VIX (fear gauge) levels.
        Low VIX during high prices = complacency (bullish sentiment, bearish signal).
        """
        if vix_history.empty:
            return {"current_vix": None, "vix_percentile": None, "assessment": "insufficient data"}

        # Get column name (yfinance uses 'adj_close' or 'close')
        col = "adj_close" if "adj_close" in vix_history.columns else "close"
        current = vix_history[col].iloc[-1]
        historical = vix_history[col]

        percentile = (historical < current).sum() / len(historical) * 100

        if current < 12:
            assessment = "EXTREME COMPLACENCY — high bubble risk"
        elif current < 15:
            assessment = "LOW FEAR — elevated bubble risk"
        elif current < 20:
            assessment = "NORMAL"
        elif current < 30:
            assessment = "ELEVATED FEAR — lower bubble risk"
        else:
            assessment = "HIGH FEAR — potential buying opportunity"

        return {
            "current_vix": round(current, 2),
            "vix_percentile": round(percentile, 1),
            "assessment": assessment,
        }

    @staticmethod
    def fear_greed_score(
        vix_level: Optional[float] = None,
        rsi_market: Optional[float] = None,
        put_call_ratio: Optional[float] = None,
    ) -> Optional[float]:
        """
        Composite fear & greed score (0 = extreme fear, 100 = extreme greed).
        Extreme greed often coincides with bubble tops.
        """
        score = 50.0  # neutral baseline
        components = 0

        if vix_level is not None:
            # Low VIX = greed, high VIX = fear
            if vix_level < 12:
                score += 20
            elif vix_level < 15:
                score += 10
            elif vix_level > 30:
                score -= 15
            elif vix_level > 20:
                score -= 5
            components += 1

        if rsi_market is not None:
            if rsi_market > 70:
                score += 15
            elif rsi_market > 60:
                score += 5
            elif rsi_market < 30:
                score -= 15
            elif rsi_market < 40:
                score -= 5
            components += 1

        if put_call_ratio is not None:
            # High put/call = fear, low = greed
            if put_call_ratio < 0.6:
                score += 15
            elif put_call_ratio < 0.8:
                score += 5
            elif put_call_ratio > 1.2:
                score -= 10
            components += 1

        if components == 0:
            return None

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def interpret_fear_greed(score: float) -> str:
        """Translate fear/greed score to label."""
        if score >= 80:
            return "EXTREME GREED — high bubble risk"
        elif score >= 65:
            return "GREED — elevated bubble risk"
        elif score >= 45:
            return "NEUTRAL"
        elif score >= 30:
            return "FEAR"
        else:
            return "EXTREME FEAR"
