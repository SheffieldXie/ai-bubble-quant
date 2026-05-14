"""
Portfolio management module.
Handles portfolio state, position tracking, and rebalancing.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages short portfolio positions and allocation."""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> {shares, entry_price, entry_date}
        self.history = []

    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + margin from shorts)."""
        return self.cash + sum(
            pos["entry_price"] * pos["shares"] for pos in self.positions.values()
        )

    @property
    def short_exposure(self) -> float:
        """Total dollar value of short positions."""
        return sum(
            pos["entry_price"] * pos["shares"] for pos in self.positions.values()
        )

    @property
    def exposure_pct(self) -> float:
        """Short exposure as percentage of portfolio."""
        pv = self.portfolio_value
        if pv <= 0:
            return 0
        return self.short_exposure / pv

    def open_short(
        self,
        ticker: str,
        shares: int,
        price: float,
        date: Optional[str] = None,
    ) -> bool:
        """Open a new short position."""
        if shares <= 0:
            return False

        margin_required = shares * price * 0.5  # 50% margin requirement

        if margin_required > self.cash:
            logger.warning(f"Insufficient margin to short {ticker}: need {margin_required:.0f}, have {self.cash:.0f}")
            return False

        self.positions[ticker] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date or datetime.now().isoformat(),
            "ticker": ticker,
        }
        self.cash -= margin_required
        self.history.append({
            "action": "SHORT_OPEN",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "date": date or datetime.now().isoformat(),
        })
        logger.info(f"Shorted {shares}x {ticker} @ ${price:.2f}")
        return True

    def close_short(
        self,
        ticker: str,
        price: float,
        date: Optional[str] = None,
    ) -> Optional[float]:
        """Close a short position. Returns P&L."""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        # P&L for short: (entry_price - exit_price) * shares
        pnl = (pos["entry_price"] - price) * pos["shares"]

        # Return margin + P&L to cash
        self.cash += pos["shares"] * pos["entry_price"] * 0.5 + pnl

        self.history.append({
            "action": "SHORT_CLOSE",
            "ticker": ticker,
            "shares": pos["shares"],
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "pnl": pnl,
            "date": date or datetime.now().isoformat(),
        })

        del self.positions[ticker]
        logger.info(f"Closed short {ticker} @ ${price:.2f}, P&L: ${pnl:.2f}")
        return pnl

    def mark_to_market(self, current_prices: dict[str, float]) -> dict:
        """
        Calculate current portfolio metrics given current prices.

        Returns:
            dict with unrealized P&L, total value, etc.
        """
        unrealized_pnl = {}
        total_pnl = 0.0

        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                pnl = (pos["entry_price"] - current_price) * pos["shares"]
                unrealized_pnl[ticker] = {
                    "shares": pos["shares"],
                    "entry_price": pos["entry_price"],
                    "current_price": current_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pos["entry_price"] / current_price - 1) * 100, 2),
                }
                total_pnl += pnl

        return {
            "cash": round(self.cash, 2),
            "positions": unrealized_pnl,
            "total_unrealized_pnl": round(total_pnl, 2),
            "portfolio_value": round(self.cash + sum(p["entry_price"] * p["shares"] for p in self.positions.values()), 2),
        }

    def get_summary(self) -> str:
        """Generate a text summary of portfolio state."""
        lines = []
        lines.append(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        lines.append(f"Cash: ${self.cash:,.2f}")
        lines.append(f"Short Exposure: ${self.short_exposure:,.2f} ({self.exposure_pct*100:.1f}%)")
        lines.append(f"Open Positions: {len(self.positions)}")

        if self.positions:
            lines.append("\nPositions:")
            for ticker, pos in self.positions.items():
                lines.append(f"  {ticker}: {pos['shares']} shares @ ${pos['entry_price']:.2f} ({pos['entry_date']})")

        return "\n".join(lines)
