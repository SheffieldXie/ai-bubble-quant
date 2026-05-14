"""
Composite bubble scoring module.
Combines all individual metrics into a unified bubble assessment.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CompositeScorer:
    """Combines multiple analysis results into a single bubble score."""

    # Default weights for each metric category
    DEFAULT_WEIGHTS = {
        "valuation": 0.30,        # PE, P/S, P/B vs historical norms
        "momentum": 0.20,         # Price vs MAs, RSI, Bollinger Bands
        "sentiment": 0.15,        # VIX, fear/greed, retail enthusiasm
        "fundamentals": 0.20,     # Revenue growth vs price growth, CAPEX
        "insider": 0.15,          # Insider selling patterns
    }

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            for key in self.weights:
                self.weights[key] /= total

    def normalize_pe(self, pe: float) -> float:
        """Normalize PE to 0-1 scale. Historical avg ~15-16, bubble > 40."""
        if pe is None or pe <= 0:
            return 0.5  # neutral when unknown
        # Sigmoid-like scaling
        normalized = min(1.0, max(0.0, (pe - 10) / 50))
        return round(normalized, 4)

    def normalize_revenue_price_gap(self, price_return: float, revenue_growth: float) -> float:
        """
        Normalize the gap between price appreciation and revenue growth.
        Higher gap = higher bubble risk.
        """
        if price_return is None or revenue_growth is None:
            return 0.5
        # revenue_growth is typically a decimal (0.15 = 15%)
        revenue_pct = revenue_growth * 100
        gap = price_return - revenue_pct
        # Gap of 100%+ = extreme, 50% = high, 0% = normal
        normalized = min(1.0, max(0.0, gap / 100))
        return round(normalized, 4)

    def normalize_rsi(self, rsi: float) -> float:
        """Normalize RSI to bubble risk scale. RSI > 70 = overbought."""
        if rsi is None:
            return 0.5
        if rsi > 70:
            return min(1.0, (rsi - 70) / 30)
        elif rsi < 30:
            return 0.0
        else:
            return (rsi - 30) / 80 * 0.3  # 0-0.3 range for neutral RSI

    def normalize_percent_b(self, percent_b: float) -> float:
        """Normalize Bollinger %b to bubble risk."""
        if percent_b is None:
            return 0.5
        if percent_b > 1.0:
            return min(1.0, percent_b - 1.0)
        elif percent_b > 0.8:
            return (percent_b - 0.8) * 2  # 0-0.4 range
        else:
            return 0.0

    def normalize_insider_selling(self, net_activity: float) -> float:
        """Normalize insider selling to bubble risk."""
        if net_activity is None:
            return 0.5
        # Negative = selling (bearish), positive = buying
        if net_activity < -1_000_000_000:
            return 1.0
        elif net_activity < -100_000_000:
            return 0.8
        elif net_activity < -10_000_000:
            return 0.5
        elif net_activity > 10_000_000:
            return 0.0
        else:
            return 0.3

    def compute(
        self,
        pe_normalized: Optional[float] = None,
        rsi_normalized: Optional[float] = None,
        bb_normalized: Optional[float] = None,
        revenue_price_gap_normalized: Optional[float] = None,
        insider_normalized: Optional[float] = None,
        sentiment_normalized: Optional[float] = None,
    ) -> dict:
        """
        Compute composite bubble score.

        Returns:
            dict with score, breakdown, and interpretation
        """
        component_scores = {
            "valuation": pe_normalized,
            "momentum": None,
            "sentiment": sentiment_normalized,
            "fundamentals": revenue_price_gap_normalized,
            "insider": insider_normalized,
        }

        # Combine momentum from RSI and Bollinger Bands
        momentum_components = [v for v in [rsi_normalized, bb_normalized] if v is not None]
        if momentum_components:
            component_scores["momentum"] = sum(momentum_components) / len(momentum_components)

        # Calculate weighted score
        score = 0.0
        total_weight = 0.0
        available_components = {}

        for component, value in component_scores.items():
            weight = self.weights.get(component, 0)
            if value is not None and weight > 0:
                score += value * weight
                total_weight += weight
                available_components[component] = {
                    "value": round(value, 4),
                    "weight": weight,
                    "contribution": round(value * weight, 4),
                }

        if total_weight == 0:
            return {
                "score": 0.5,  # neutral default when no data
                "interpretation": "Insufficient data — using neutral baseline",
                "components": available_components,
                "data_coverage": f"0/{len(self.weights)} components available",
            }

        final_score = round(score / total_weight, 4)

        # Interpretation
        if final_score >= 0.75:
            interpretation = "EXTREME BUBBLE — Strong short candidate"
        elif final_score >= 0.60:
            interpretation = "HIGH BUBBLE RISK — Consider building shorts"
        elif final_score >= 0.45:
            interpretation = "MODERATE RISK — Monitor and wait"
        elif final_score >= 0.30:
            interpretation = "LOW RISK — Some warning signs"
        else:
            interpretation = "MINIMAL RISK — No significant bubble signals"

        return {
            "score": final_score,
            "interpretation": interpretation,
            "components": available_components,
            "data_coverage": f"{len(available_components)}/{len(self.weights)} components available",
        }
