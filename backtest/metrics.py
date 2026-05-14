"""
Backtest performance metrics calculation.
"""

import numpy as np
import pandas as pd


class BacktestMetrics:
    """Calculates performance metrics for backtest results."""

    def __init__(self, daily_values: list[dict], risk_free_rate: float = 0.05):
        self.df = pd.DataFrame(daily_values)
        self.risk_free_rate = risk_free_rate

        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.sort_values("date")
            self.df["daily_return"] = self.df["portfolio_value"].pct_change()
            self.df["cumulative_return"] = (self.df["portfolio_value"] / self.df["portfolio_value"].iloc[0]) - 1

    def calculate_all(self) -> dict:
        """Calculate all performance metrics."""
        if self.df.empty:
            return {"error": "No data to analyze"}

        returns = self.df["daily_return"].dropna()
        total_days = len(self.df)

        metrics = {
            # Basic
            "total_return_pct": round(self.df["cumulative_return"].iloc[-1] * 100, 2),
            "trading_days": total_days,
            "annualized_return_pct": round(self._annualized_return(returns, total_days), 2),

            # Risk
            "volatility_annual_pct": round(returns.std() * np.sqrt(252) * 100, 2),
            "max_drawdown_pct": round(self._max_drawdown() * 100, 2),
            "var_95_pct": round(returns.quantile(0.05) * 100, 2),

            # Risk-adjusted
            "sharpe_ratio": round(self._sharpe_ratio(returns), 2),
            "sortino_ratio": round(self._sortino_ratio(returns), 2),
            "calmar_ratio": round(self._calmar_ratio(), 2),

            # Trade stats
            "avg_daily_return_pct": round(returns.mean() * 100, 4),
            "best_day_pct": round(returns.max() * 100, 2),
            "worst_day_pct": round(returns.min() * 100, 2),
            "positive_days_pct": round((returns > 0).sum() / len(returns) * 100, 1),

            # Final state
            "final_portfolio_value": round(self.df["portfolio_value"].iloc[-1], 2),
            "initial_portfolio_value": round(self.df["portfolio_value"].iloc[0], 2),
        }

        return metrics

    def _annualized_return(self, returns: pd.Series, total_days: int) -> float:
        total_return = (1 + returns).prod() ** (252 / total_days) - 1
        return total_return * 100

    def _max_drawdown(self) -> float:
        cumulative = (1 + self.df["daily_return"].fillna(0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        excess = returns.mean() - self.risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return excess / returns.std() * np.sqrt(252)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        excess = returns.mean() - self.risk_free_rate / 252
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0
        return excess / downside * np.sqrt(252)

    def _calmar_ratio(self) -> float:
        ann_return = self.df["cumulative_return"].iloc[-1] * 252 / len(self.df)
        max_dd = abs(self._max_drawdown())
        if max_dd == 0:
            return 0
        return ann_return / max_dd

    def format_report(self, metrics: dict) -> str:
        """Format metrics as a readable report."""
        lines = []
        lines.append("=" * 50)
        lines.append("BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Return:          {metrics.get('total_return_pct', 'N/A')}%")
        lines.append(f"Annualized Return:     {metrics.get('annualized_return_pct', 'N/A')}%")
        lines.append(f"Volatility (Ann.):     {metrics.get('volatility_annual_pct', 'N/A')}%")
        lines.append(f"Max Drawdown:          {metrics.get('max_drawdown_pct', 'N/A')}%")
        lines.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 'N/A')}")
        lines.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 'N/A')}")
        lines.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 'N/A')}")
        lines.append(f"VaR (95%):             {metrics.get('var_95_pct', 'N/A')}%")
        lines.append(f"Best Day:              {metrics.get('best_day_pct', 'N/A')}%")
        lines.append(f"Worst Day:             {metrics.get('worst_day_pct', 'N/A')}%")
        lines.append(f"Positive Days:         {metrics.get('positive_days_pct', 'N/A')}%")
        lines.append(f"Trading Days:          {metrics.get('trading_days', 'N/A')}")
        lines.append(f"Final Value:           ${metrics.get('final_portfolio_value', 0):,.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)
