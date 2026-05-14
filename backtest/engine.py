"""
Backtest engine.
Simulates trading strategy on historical data.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from strategy.portfolio import PortfolioManager
from strategy.risk import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs historical simulation of the short strategy."""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.portfolio = PortfolioManager(initial_capital)
        self.risk_manager = RiskManager()

        # Trade log
        self.trade_log = []
        self.daily_values = []

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: list[dict],
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        Run backtest simulation.

        Args:
            price_data: dict of ticker -> DataFrame with OHLCV data
            signals: list of signal dicts with timestamp, ticker, action, strength
            start_date: Start date for backtest (ISO format)
            end_date: End date for backtest

        Returns:
            dict with backtest results
        """
        logger.info(f"Starting backtest with {len(price_data)} tickers, {len(signals)} signals")

        # Collect all dates
        all_dates = set()
        for df in price_data.values():
            if not df.empty:
                all_dates.update(df.index)

        dates = sorted(all_dates)
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        # Create signal lookup
        signal_lookup = {}
        for sig in signals:
            date = sig["timestamp"][:10]  # YYYY-MM-DD
            key = (date, sig["ticker"])
            signal_lookup[key] = sig

        high_watermark = self.initial_capital

        for date in dates:
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]

            # Update portfolio with current prices
            current_prices = {}
            for ticker, df in price_data.items():
                if date in df.index:
                    row = df.loc[date]
                    col = "adj_close" if "adj_close" in df.columns else "close"
                    current_prices[ticker] = row[col]

            # Check risk limits
            mtm = self.portfolio.mark_to_market(current_prices)
            unrealized_pnl = mtm["total_unrealized_pnl"]
            current_value = mtm["portfolio_value"]

            if current_value > high_watermark:
                high_watermark = current_value

            risk_check = self.risk_manager.check_portfolio_risk(
                portfolio_value=current_value,
                total_unrealized_loss=unrealized_pnl,
                exposure_pct=self.portfolio.exposure_pct,
                high_watermark=high_watermark,
            )

            # Execute risk actions
            if risk_check["risk_level"] == "CRITICAL":
                # Close all positions
                for ticker in list(self.portfolio.positions.keys()):
                    if ticker in current_prices:
                        price = current_prices[ticker] * (1 + self.slippage_pct)
                        pnl = self.portfolio.close_short(ticker, price, date_str)
                        self.trade_log.append({
                            "date": date_str,
                            "action": "FORCE_CLOSE",
                            "ticker": ticker,
                            "price": price,
                            "pnl": pnl,
                            "reason": "Risk limit breach",
                        })

            # Check stop losses and take profits
            risk_actions = self.risk_manager.check_stop_losses(
                self.portfolio.positions, current_prices
            )
            for action in risk_actions:
                ticker = action["ticker"]
                price = action["exit_price"] * (1 + self.slippage_pct)
                pnl = self.portfolio.close_short(ticker, price, date_str)
                pnl -= price * self.portfolio.positions.get(ticker, {}).get("shares", 0) * self.commission
                self.trade_log.append({
                    "date": date_str,
                    "action": action["action"],
                    "ticker": ticker,
                    "price": price,
                    "pnl": pnl,
                    "reason": action["reason"],
                })

            # Check for new signals
            for ticker in price_data.keys():
                key = (date_str, ticker)
                if key in signal_lookup:
                    sig = signal_lookup[key]
                    if sig["action"] in ["SHORT", "STRONG SHORT"] and ticker in current_prices:
                        if ticker not in self.portfolio.positions:
                            entry_price = current_prices[ticker] * (1 - self.slippage_pct)
                            # Simple sizing: signal strength * fixed amount
                            size = int(sig["strength"] * 100)  # simplified
                            cost = size * entry_price * 0.5  # margin
                            if cost < self.portfolio.cash * 0.1:  # max 10% of cash
                                self.portfolio.open_short(ticker, size, entry_price, date_str)
                                commission_cost = size * entry_price * self.commission
                                self.portfolio.cash -= commission_cost
                                self.trade_log.append({
                                    "date": date_str,
                                    "action": "SHORT_OPEN",
                                    "ticker": ticker,
                                    "price": entry_price,
                                    "shares": size,
                                    "signal_strength": sig["strength"],
                                    "commission": commission_cost,
                                })

            # Record daily value
            self.daily_values.append({
                "date": date_str,
                "portfolio_value": mtm["portfolio_value"],
                "cash": mtm["cash"],
                "unrealized_pnl": unrealized_pnl,
                "exposure_pct": self.portfolio.exposure_pct,
            })

        logger.info(f"Backtest complete: {len(self.trade_log)} trades, {len(self.daily_values)} days")

        return {
            "trade_log": self.trade_log,
            "daily_values": self.daily_values,
            "final_portfolio_value": self.portfolio.portfolio_value,
            "total_return": (self.portfolio.portfolio_value - self.initial_capital) / self.initial_capital,
            "total_trades": len(self.trade_log),
        }
