"""
Signal generation module.
Combines bubble scores, technical signals, and fundamental data
to generate actionable short/long signals.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals based on analysis results."""

    def __init__(
        self,
        min_bubble_score: float = 0.65,
        rsi_overbought_threshold: float = 70,
        confirm_technical: bool = True,
    ):
        self.min_bubble_score = min_bubble_score
        self.rsi_overbought = rsi_overbought_threshold
        self.confirm_technical = confirm_technical

    def generate_signal(
        self,
        ticker: str,
        bubble_score: float,
        technical_signals: Optional[dict] = None,
        fundamental_flags: Optional[list] = None,
        current_price: Optional[float] = None,
    ) -> dict:
        """
        Generate a trading signal for a specific ticker.
        """
        # Handle None score
        if bubble_score is None:
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "bubble_score": 0.0,
                "action": "HOLD",
                "strength": 0.0,
                "reasoning": ["Insufficient data to compute bubble score"],
                "current_price": current_price,
            }
        signal = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "bubble_score": bubble_score,
            "action": "HOLD",
            "strength": 0.0,
            "reasoning": [],
            "current_price": current_price,
        }

        # ── Step 1: Evaluate bubble score ───────────────────────────────
        if bubble_score < self.min_bubble_score:
            signal["action"] = "HOLD"
            signal["reasoning"].append(f"Bubble score {bubble_score:.2f} below threshold {self.min_bubble_score}")
            return signal

        signal["reasoning"].append(f"High bubble score: {bubble_score:.2f}")

        # ── Step 2: Check technical confirmation ────────────────────────
        technical_score = 0.0
        if technical_signals:
            if technical_signals.get("rsi_overbought"):
                technical_score += 0.3
                signal["reasoning"].append(f"RSI overbought at {technical_signals.get('rsi')}")

            if technical_signals.get("macd_bearish_cross"):
                technical_score += 0.3
                signal["reasoning"].append("MACD bearish crossover")

            if technical_signals.get("price_above_upper_band"):
                technical_score += 0.2
                signal["reasoning"].append("Price above Bollinger upper band")

            if technical_signals.get("timing"):
                signal["reasoning"].append(f"Technical timing: {technical_signals['timing']}")

            if technical_signals.get("volume_analysis"):
                dist_days = technical_signals["volume_analysis"].get("distribution_days", 0)
                if dist_days >= 3:
                    technical_score += 0.2
                    signal["reasoning"].append(f"{dist_days} distribution days (institutional selling)")
        else:
            technical_score = 0.5  # neutral if no data

        # ── Step 3: Check fundamental red flags ─────────────────────────
        fundamental_score = 0.0
        if fundamental_flags:
            flag_count = len(fundamental_flags)
            fundamental_score = min(0.3, flag_count * 0.1)
            if flag_count > 0:
                signal["reasoning"].append(f"{flag_count} fundamental red flags: {', '.join(fundamental_flags)}")

        # ── Step 4: Combine into final signal ───────────────────────────
        overall_strength = bubble_score * 0.5 + technical_score * 0.35 + fundamental_score * 0.15

        if self.confirm_technical and technical_score < 0.3 and bubble_score < 0.8:
            signal["action"] = "WATCH"
            signal["reasoning"].append("Waiting for technical confirmation before shorting")
            signal["strength"] = round(overall_strength * 0.5, 4)
        elif overall_strength > 0.7:
            signal["action"] = "STRONG SHORT"
            signal["strength"] = round(min(1.0, overall_strength), 4)
        elif overall_strength > 0.5:
            signal["action"] = "SHORT"
            signal["strength"] = round(overall_strength, 4)
        elif overall_strength > 0.35:
            signal["action"] = "WEAK SHORT"
            signal["strength"] = round(overall_strength, 4)
        else:
            signal["action"] = "WATCH"
            signal["strength"] = round(overall_strength, 4)

        return signal

    @staticmethod
    def format_signals(signals: list[dict]) -> str:
        """Format a list of signals for display."""
        if not signals:
            return "No signals generated."

        output = []
        output.append("=" * 70)
        output.append("AI BUBBLE QUANT — TRADING SIGNALS")
        output.append("=" * 70)

        # Sort by strength (strongest first)
        signals = sorted(signals, key=lambda s: s["strength"], reverse=True)

        for sig in signals:
            action_icon = {
                "STRONG SHORT": "🔴",
                "SHORT": "🔴",
                "WEAK SHORT": "🟡",
                "WATCH": "⚪",
                "HOLD": "⚪",
            }.get(sig["action"], "⚪")

            output.append(f"\n{action_icon} {sig['ticker']} — {sig['action']}")
            output.append(f"   Strength: {sig['strength']:.2f}")
            if sig.get("current_price"):
                output.append(f"   Price: ${sig['current_price']:.2f}")
            for reason in sig["reasoning"]:
                output.append(f"   • {reason}")

        output.append("\n" + "=" * 70)
        return "\n".join(output)
