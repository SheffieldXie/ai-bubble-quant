"""
0DTE Options Scanner — Main Entry Point
Scrapes real-time SPY/QQQ data and analyzes 0DTE put opportunities.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.scraper import MarketDataScraper
from analysis.odte_analyzer import ZeroDTEAnalyzer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_spodte_scan(ticker: str, criterion: str, hours: float, n_sims: int):
    """Run full 0DTE scan for SPY or QQQ."""
    logger = logging.getLogger("scanner")

    # Step 1: Scrape real-time data
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching real-time data from Eastmoney/Sina")
    logger.info("=" * 60)

    scraper = MarketDataScraper()

    quote = scraper.get_quote(ticker)
    if not quote:
        logger.error(f"Failed to get quote for {ticker}. Check network.")
        logger.info("Falling back to synthetic data for demonstration...")
        # Fallback prices (approximate as of May 2026)
        fallback_prices = {"SPY": 540.0, "QQQ": 490.0, "^GSPC": 5400.0, "^IXIC": 19000.0}
        current_price = fallback_prices.get(ticker, 500.0)
        logger.info(f"Using fallback price: ${current_price:.2f}")
    else:
        current_price = quote.get("price") or quote.get("current_price")
        logger.info(f"Live price for {ticker}: ${current_price:.2f}")
        logger.info(f"Source: {quote.get('source', 'unknown')}")

    # Estimate IV from recent price movements
    # In production, this would come from options data
    # For now, use reasonable estimates
    iv_estimates = {
        "SPY": 0.13,   # ~13% IV (low vol environment)
        "QQQ": 0.18,   # ~18% IV (higher vol)
        "^GSPC": 0.13,
        "^IXIC": 0.18,
    }
    iv = iv_estimates.get(ticker, 0.15)

    # If we have history, calculate realized vol
    history = scraper.get_history(ticker, period="1y")
    if history is not None and not history.empty:
        col = "adj_close" if "adj_close" in history.columns else "close"
        returns = history[col].pct_change().dropna()
        realized_vol = returns.std() * (252 ** 0.5)
        logger.info(f"Realized vol (1Y): {realized_vol*100:.1f}%")
        # Use max of IV and realized vol for conservative estimate
        iv = max(iv, realized_vol)
        logger.info(f"Using volatility: {iv*100:.1f}%")
    else:
        logger.info(f"Using estimated IV: {iv*100:.1f}% (no history available)")

    # Step 2: Analyze 0DTE options
    logger.info("=" * 60)
    logger.info("STEP 2: Analyzing 0DTE put options")
    logger.info("=" * 60)

    analyzer = ZeroDTEAnalyzer(
        current_price=current_price,
        implied_volatility=iv,
    )

    # Show full comparison
    print(analyzer.compare_all_strategies())

    # Step 3: Monte Carlo simulation for recommended strike
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Monte Carlo simulation for optimal strike")
    logger.info("=" * 60)

    optimal = analyzer.find_optimal_strategy(criterion=criterion)
    if "error" not in optimal:
        strike = optimal["strike"]
        mc = analyzer.monte_carlo_simulation(strike, hours_remaining=hours, n_sims=n_sims)

        print("")
        print("=" * 60)
        print(f"MONTE CARLO SIMULATION ({n_sims:,} simulations)")
        print(f"Strike: ${strike:.2f} | Option Cost: ${mc['option_cost']:.4f}")
        print("=" * 60)
        print(f"  Win Rate:         {mc['win_rate']:.1f}%")
        print(f"  Expected P&L:     ${mc['expected_pnl']:.4f}")
        print(f"  Avg Win:          ${mc['avg_win']:.4f}")
        print(f"  Avg Loss:         ${mc['avg_lose']:.4f}")
        print(f"  Max Win:          ${mc['max_win']:.4f}")
        print(f"  Max Loss:         ${mc['max_lose']:.4f}")
        print(f"  5th Percentile:   ${mc['p5']:.4f}")
        print(f"  25th Percentile:  ${mc['p25']:.4f}")
        print(f"  Median:           ${mc['p50']:.4f}")
        print(f"  75th Percentile:  ${mc['p75']:.4f}")
        print(f"  95th Percentile:  ${mc['p95']:.4f}")

    # Step 4: Full strike-by-strike comparison
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: All strikes comparison")
    logger.info("=" * 60)

    mc_df = analyzer.simulate_all_strikes(hours_remaining=hours, n_sims=min(n_sims, 5000))
    if not mc_df.empty:
        print("")
        print("STRIKE COMPARISON (sorted by win rate):")
        print("-" * 90)
        print(f"{'Strike':>8} {'Type':>5} {'Cost':>8} {'Win%':>7} {'EV':>8} {'AvgWin':>8} {'AvgLoss':>8} {'MaxWin':>8} {'MaxLoss':>8}")
        print("-" * 90)
        for _, row in mc_df.sort_values("win_rate", ascending=False).iterrows():
            # Get moneyness from analytical
            df_analytical = analyzer.analyze_put_strikes()
            moneyness = "OTM"
            if not df_analytical.empty:
                match = df_analytical[df_analytical["strike"] == row["strike"]]
                if not match.empty:
                    moneyness = match.iloc[0]["moneyness"]

            print(f"${row['strike']:>6.2f} {moneyness:>5} ${row['option_cost']:>6.4f} {row['win_rate']:>6.1f}% ${row['expected_pnl']:>6.4f} ${row['avg_win']:>6.4f} ${row['avg_lose']:>6.4f} ${row['max_win']:>6.4f} ${row['max_lose']:>6.4f}")

    print("")
    print("=" * 60)
    print("DISCLAIMER: This is educational analysis, not financial advice.")
    print("0DTE options are extremely high-risk. You can lose 100% of premium.")
    print("Past performance does not predict future results.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="0DTE Options Scanner")
    parser.add_argument(
        "--ticker",
        choices=["SPY", "QQQ"],
        default="SPY",
        help="Underlying to analyze (default: SPY)",
    )
    parser.add_argument(
        "--criterion",
        choices=["sharpe", "win_rate", "payout", "balanced"],
        default="balanced",
        help="Optimization criterion (default: balanced)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=6.5,
        help="Hours remaining until expiry (default: 6.5 = full day)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    run_spodte_scan(args.ticker, args.criterion, args.hours, args.sims)


if __name__ == "__main__":
    main()
