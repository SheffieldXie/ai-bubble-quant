"""
Technical analysis module.
Indicators useful for timing short entries.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical indicators for timing and confirmation."""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return None
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
        return round(rsi_val, 2)

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """
        MACD indicator.

        Returns:
            dict with 'macd', 'signal', 'histogram'
        """
        if len(prices) < slow + signal:
            return {"macd": None, "signal": None, "histogram": None}

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": round(macd_line.iloc[-1], 4),
            "signal": round(signal_line.iloc[-1], 4),
            "histogram": round(histogram.iloc[-1], 4),
        }

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
        """
        Bollinger Bands.

        Returns:
            dict with 'upper', 'middle', 'lower', '%b' (position within bands)
        """
        if len(prices) < period:
            return {"upper": None, "middle": None, "lower": None, "percent_b": None}

        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        current = prices.iloc[-1]
        band_width = upper.iloc[-1] - lower.iloc[-1]
        if band_width > 0:
            percent_b = (current - lower.iloc[-1]) / band_width
        else:
            percent_b = 0.5

        return {
            "upper": round(upper.iloc[-1], 2),
            "middle": round(middle.iloc[-1], 2),
            "lower": round(lower.iloc[-1], 2),
            "percent_b": round(percent_b, 4),
        }

    @staticmethod
    def moving_averages(prices: pd.Series, periods: list[int] = [20, 50, 200]) -> dict:
        """Calculate key moving averages."""
        result = {}
        for p in periods:
            if len(prices) >= p:
                result[f"ma_{p}"] = round(prices.rolling(window=p).mean().iloc[-1], 2)
        return result

    @staticmethod
    def volume_analysis(df: pd.DataFrame, period: int = 20) -> dict:
        """
        Analyze volume patterns.
        Rising volume on down days = distribution (bearish).
        """
        if "volume" not in df.columns or len(df) < period:
            return {"avg_volume": None, "recent_volume_ratio": None, "distribution_days": None}

        avg_vol = df["volume"].tail(period).mean()
        recent_vol = df["volume"].tail(5).mean()
        ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # Count distribution days (down days with above-average volume)
        if "adj_close" in df.columns:
            returns = df["adj_close"].pct_change()
            distribution = ((returns < 0) & (df["volume"] > avg_vol)).sum()
        else:
            distribution = 0

        return {
            "avg_volume": round(avg_vol, 0),
            "recent_volume_ratio": round(ratio, 2),
            "distribution_days": int(distribution),
        }

    @staticmethod
    def get_short_timing_signals(prices: pd.Series, volume_df: pd.DataFrame = None) -> dict:
        """
        Combine multiple technical indicators into short timing signals.

        Returns:
            dict with various signals and an overall timing assessment
        """
        rsi_val = TechnicalAnalyzer.rsi(prices)
        macd_val = TechnicalAnalyzer.macd(prices)
        bb_val = TechnicalAnalyzer.bollinger_bands(prices)
        ma_val = TechnicalAnalyzer.moving_averages(prices)

        signals = {
            "rsi": rsi_val,
            "rsi_overbought": rsi_val is not None and rsi_val > 70,
            "macd": macd_val,
            "macd_bearish_cross": macd_val["macd"] is not None and macd_val["macd"] < macd_val["signal"],
            "bollinger_bands": bb_val,
            "price_above_upper_band": bb_val["percent_b"] is not None and bb_val["percent_b"] > 1.0,
            "moving_averages": ma_val,
        }

        # Overall timing assessment
        bearish_signals = sum([
            signals["rsi_overbought"],
            signals["macd_bearish_cross"],
            signals["price_above_upper_band"],
        ])

        if bearish_signals >= 2:
            signals["timing"] = "FAVORABLE for short entry"
        elif bearish_signals >= 1:
            signals["timing"] = "NEUTRAL — watch for confirmation"
        else:
            signals["timing"] = "UNFAVORABLE — wait for better entry"

        return signals
