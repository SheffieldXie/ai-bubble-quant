"""
Visualization module for backtest results.
Generates charts using matplotlib.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Generates visualization charts for backtest results."""

    def __init__(self, output_dir: str = "./backtest/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_portfolio_value(self, daily_values: list[dict], save: bool = True) -> str:
        """Plot portfolio value over time."""
        df = pd.DataFrame(daily_values)
        if df.empty:
            return ""

        df["date"] = pd.to_datetime(df["date"])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Portfolio value
        axes[0].plot(df["date"], df["portfolio_value"], color="#2196F3", linewidth=1.5)
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].set_title("AI Bubble Quant — Backtest Results")
        axes[0].grid(True, alpha=0.3)

        # Daily returns
        df["daily_return"] = df["portfolio_value"].pct_change() * 100
        colors = ["#4CAF50" if x >= 0 else "#F44336" for x in df["daily_return"].dropna()]
        axes[1].bar(df["date"].iloc[1:], df["daily_return"].dropna(), color=colors, alpha=0.7, width=1)
        axes[1].set_ylabel("Daily Return (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color="black", linewidth=0.5)

        # Exposure
        axes[2].plot(df["date"], df["exposure_pct"] * 100, color="#FF9800", linewidth=1.5)
        axes[2].set_ylabel("Short Exposure (%)")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / "backtest_portfolio.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved chart: {path}")
            return str(path)

        plt.close()
        return ""

    def plot_drawdown(self, daily_values: list[dict], save: bool = True) -> str:
        """Plot drawdown over time."""
        df = pd.DataFrame(daily_values)
        if df.empty:
            return ""

        df["date"] = pd.to_datetime(df["date"])
        cumulative = (1 + df["portfolio_value"].pct_change().fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = ((cumulative - running_max) / running_max) * 100

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(df["date"], drawdown, 0, alpha=0.5, color="#F44336")
        ax.plot(df["date"], drawdown, color="#D32F2F", linewidth=1)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Portfolio Drawdown")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()

        if save:
            path = self.output_dir / "backtest_drawdown.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved chart: {path}")
            return str(path)

        plt.close()
        return ""

    def generate_all_charts(self, daily_values: list[dict]) -> list[str]:
        """Generate all charts and return file paths."""
        paths = []
        paths.append(self.plot_portfolio_value(daily_values))
        paths.append(self.plot_drawdown(daily_values))
        return [p for p in paths if p]
