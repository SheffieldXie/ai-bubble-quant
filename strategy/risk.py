"""
Risk management module.
Monitors and enforces risk limits for short positions.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk limits and monitors portfolio risk."""

    def __init__(
        self,
        stop_loss_pct: float = 0.10,
        take_profit_pct: float = 0.20,
        max_drawdown_pct: float = 0.15,
        max_single_position_pct: float = 0.15,
        max_total_exposure_pct: float = 0.80,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_single_pct = max_single_position_pct
        self.max_exposure_pct = max_total_exposure_pct

    def check_stop_losses(
        self,
        positions: dict,
        current_prices: dict[str, float],
    ) -> list[dict]:
        """
        Check which positions have hit stop-loss or take-profit.

        Returns:
            List of positions that need action
        """
        actions = []

        for ticker, pos in positions.items():
            if ticker not in current_prices:
                continue

            current_price = current_prices[ticker]
            entry_price = pos["entry_price"]

            # For shorts: loss when price rises above entry
            loss_pct = (current_price - entry_price) / entry_price
            profit_pct = (entry_price - current_price) / entry_price

            if loss_pct >= self.stop_loss_pct:
                actions.append({
                    "ticker": ticker,
                    "action": "STOP_LOSS",
                    "reason": f"Stop loss hit: price up {loss_pct*100:.1f}%",
                    "exit_price": current_price,
                })
            elif profit_pct >= self.take_profit_pct:
                actions.append({
                    "ticker": ticker,
                    "action": "TAKE_PROFIT",
                    "reason": f"Take profit hit: price down {profit_pct*100:.1f}%",
                    "exit_price": current_price,
                })

        return actions

    def check_portfolio_risk(
        self,
        portfolio_value: float,
        total_unrealized_loss: float,
        exposure_pct: float,
        high_watermark: float,
    ) -> dict:
        """
        Check portfolio-level risk limits.

        Returns:
            dict with risk assessment and any violations
        """
        violations = []

        # Drawdown check
        current_value = portfolio_value + total_unrealized_loss
        drawdown = (high_watermark - current_value) / high_watermark if high_watermark > 0 else 0

        if drawdown > self.max_drawdown_pct:
            violations.append({
                "type": "MAX_DRAWDOWN",
                "severity": "CRITICAL",
                "message": f"Portfolio drawdown {drawdown*100:.1f}% exceeds limit {self.max_drawdown_pct*100:.1f}%",
            })

        # Exposure check
        if exposure_pct > self.max_exposure_pct:
            violations.append({
                "type": "MAX_EXPOSURE",
                "severity": "HIGH",
                "message": f"Exposure {exposure_pct*100:.1f}% exceeds limit {self.max_exposure_pct*100:.1f}%",
            })

        risk_level = "CRITICAL" if any(v["severity"] == "CRITICAL" for v in violations) else \
                     "HIGH" if violations else "OK"

        return {
            "risk_level": risk_level,
            "drawdown": round(drawdown * 100, 2),
            "exposure_pct": round(exposure_pct * 100, 2),
            "violations": violations,
        }

    def get_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop-loss price for a short position."""
        return round(entry_price * (1 + self.stop_loss_pct), 2)

    def get_take_profit_price(self, entry_price: float) -> float:
        """Calculate take-profit price for a short position."""
        return round(entry_price * (1 - self.take_profit_pct), 2)
