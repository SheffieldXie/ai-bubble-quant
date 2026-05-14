"""
Position sizing module.
Determines how much to short based on conviction and risk parameters.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates optimal position sizes for short trades."""

    def __init__(
        self,
        max_portfolio_pct: float = 0.8,
        max_single_position_pct: float = 0.15,
        stop_loss_pct: float = 0.10,
        max_risk_per_trade: float = 0.02,
    ):
        self.max_portfolio_pct = max_portfolio_pct
        self.max_single_pct = max_single_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_size(
        self,
        portfolio_value: float,
        signal_strength: float,
        current_price: float,
        existing_exposure_pct: float = 0.0,
    ) -> dict:
        """
        Calculate position size using Kelly-inspired approach with constraints.

        Args:
            portfolio_value: Total portfolio value
            signal_strength: Signal strength (0-1)
            current_price: Current stock price
            existing_exposure_pct: Current short exposure as % of portfolio

        Returns:
            dict with position details
        """
        available_pct = self.max_portfolio_pct - existing_exposure_pct
        if available_pct <= 0:
            return {
                "shares": 0,
                "dollar_amount": 0,
                "pct_of_portfolio": 0,
                "reason": "Max portfolio exposure reached",
            }

        # Kelly-inspired sizing: higher conviction = larger position
        kelly_fraction = signal_strength * self.max_single_pct
        # Cap at single position limit
        position_pct = min(kelly_fraction, self.max_single_pct, available_pct)

        dollar_amount = portfolio_value * position_pct
        shares = int(dollar_amount / current_price)

        # Risk check: how much would we lose if stop hits?
        risk_amount = shares * current_price * self.stop_loss_pct
        risk_pct = risk_amount / portfolio_value if portfolio_value > 0 else 0

        if risk_pct > self.max_risk_per_trade:
            # Reduce size to meet risk constraint
            max_shares = int((portfolio_value * self.max_risk_per_trade) / (current_price * self.stop_loss_pct))
            shares = min(shares, max_shares)
            dollar_amount = shares * current_price
            position_pct = dollar_amount / portfolio_value

        return {
            "shares": shares,
            "dollar_amount": round(dollar_amount, 2),
            "pct_of_portfolio": round(position_pct * 100, 2),
            "stop_loss_price": round(current_price * (1 + self.stop_loss_pct), 2),
            "max_loss": round(shares * current_price * self.stop_loss_pct, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "reason": "OK",
        }
