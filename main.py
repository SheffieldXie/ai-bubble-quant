"""
AI Bubble Quantitative Assessment Framework
Main entry point — orchestrates data fetch, analysis, signals, and backtest.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import MarketDataFetcher
from data.storage import DataStorage
from analysis.bubble_metrics import BubbleMetrics
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.fundamental import FundamentalAnalyzer
from analysis.composite import CompositeScorer
from strategy.signals import SignalGenerator
from strategy.sizing import PositionSizer
from backtest.engine import BacktestEngine
from backtest.metrics import BacktestMetrics
from backtest.visualization import BacktestVisualizer


# ── Configuration ──────────────────────────────────────────────────────

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tickers(tickers_path: str = "config/tickers.yaml") -> dict:
    with open(tickers_path) as f:
        return yaml.safe_load(f)


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Pipeline Steps ─────────────────────────────────────────────────────

def step_fetch(config: dict, tickers_cfg: dict, args) -> dict:
    """Fetch all required data."""
    logger = logging.getLogger("fetch")
    logger.info("=" * 50)
    logger.info("STEP 1: Fetching data")
    logger.info("=" * 50)

    storage = DataStorage(config["data"]["cache_dir"])
    fetcher = MarketDataFetcher(cache=storage)

    # Collect all tickers
    all_tickers = []
    for group in tickers_cfg.values():
        all_tickers.extend(group.keys())
    all_tickers = list(set(all_tickers))

    logger.info(f"Fetching data for {len(all_tickers)} tickers...")
    if getattr(args, "synthetic", False):
        logger.info("SYNTHETIC MODE: Generating test data (no API calls)")

    # Fetch price data
    price_data = fetcher.get_multiple_tickers(
        all_tickers,
        period=f"{config['data']['lookback_years']}y",
        skip_api=getattr(args, "synthetic", False),
    )

    # Fetch fundamentals for AI tickers
    fundamentals = {}
    ai_tickers = []
    for group in ["ai_chips", "ai_hyperscalers", "ai_infrastructure", "ai_picks_shovels"]:
        if group in tickers_cfg:
            ai_tickers.extend(tickers_cfg[group].keys())

    if getattr(args, "synthetic", False):
        # Generate synthetic fundamentals directly
        for ticker in ai_tickers:
            fundamentals[ticker] = fetcher._synthetic_fundamentals(ticker)
    else:
        try:
            fundamentals[ticker] = fetcher.get_key_metrics(ticker)
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {ticker}: {e}")

    # Fetch VIX
    if getattr(args, "synthetic", False):
        vix_data = fetcher._generate_synthetic_data("^VIX")
    else:
        vix_data = fetcher.get_vix_history()

    logger.info(f"Data fetch complete: {len(price_data)} tickers, {len(fundamentals)} fundamentals")

    return {
        "price_data": price_data,
        "fundamentals": fundamentals,
        "vix_data": vix_data,
    }


def step_analyze(data: dict, config: dict, tickers_cfg: dict) -> dict:
    """Run all analysis modules."""
    logger = logging.getLogger("analyze")
    logger.info("=" * 50)
    logger.info("STEP 2: Running analysis")
    logger.info("=" * 50)

    bubble_metrics = BubbleMetrics()
    technical = TechnicalAnalyzer()
    sentiment = SentimentAnalyzer()
    fundamental = FundamentalAnalyzer()
    scorer = CompositeScorer(weights=config["analysis"].get("bubble_weights"))

    ai_tickers = []
    for group in ["ai_chips", "ai_hyperscalers", "ai_infrastructure", "ai_picks_shovels"]:
        if group in tickers_cfg:
            ai_tickers.extend(tickers_cfg[group].keys())

    results = {}

    for ticker in ai_tickers:
        price_df = data["price_data"].get(ticker)
        if price_df is None or price_df.empty:
            continue

        col = "adj_close" if "adj_close" in price_df.columns else "close"
        prices = price_df[col]

        # Technical analysis
        tech_signals = technical.get_short_timing_signals(prices, price_df)

        # Fundamental analysis
        fund = data["fundamentals"].get(ticker, {})
        pe_analysis = fundamental.analyze_pe_anomaly(
            fund.get("trailing_pe"), fund.get("forward_pe")
        )
        red_flags = fundamental.valuation_red_flags(fund)

        # Bubble metrics (simplified for now — will expand)
        rsi_val = bubble_metrics.rsi(prices)
        deviation = bubble_metrics.price_deviation_from_ma(price_df)

        # Composite score components
        pe_norm = scorer.normalize_pe(fund.get("trailing_pe"))
        rsi_norm = scorer.normalize_rsi(rsi_val)
        bb_norm = scorer.normalize_percent_b(tech_signals.get("bollinger_bands", {}).get("percent_b"))

        composite = scorer.compute(
            pe_normalized=pe_norm,
            rsi_normalized=rsi_norm,
            bb_normalized=bb_norm,
        )

        results[ticker] = {
            "price": prices.iloc[-1] if not prices.empty else None,
            "composite_score": composite,
            "technical_signals": tech_signals,
            "pe_analysis": pe_analysis,
            "red_flags": red_flags,
            "deviation_from_ma": deviation,
            "rsi": rsi_val,
        }

        logger.info(f"  {ticker}: Score={composite['score']}, {composite['interpretation']}")

    return results


def step_signals(analysis_results: dict, config: dict) -> list[dict]:
    """Generate trading signals."""
    logger = logging.getLogger("signals")
    logger.info("=" * 50)
    logger.info("STEP 3: Generating signals")
    logger.info("=" * 50)

    signal_gen = SignalGenerator(
        min_bubble_score=config["strategy"]["min_score_to_short"],
    )

    signals = []
    for ticker, result in analysis_results.items():
        composite = result["composite_score"]
        signal = signal_gen.generate_signal(
            ticker=ticker,
            bubble_score=composite["score"],
            technical_signals=result["technical_signals"],
            fundamental_flags=result["red_flags"],
            current_price=result["price"],
        )
        signals.append(signal)

    print(SignalGenerator.format_signals(signals))
    return signals


def step_backtest(data: dict, signals: list[dict], config: dict):
    """Run backtest simulation."""
    logger = logging.getLogger("backtest")
    logger.info("=" * 50)
    logger.info("STEP 4: Running backtest")
    logger.info("=" * 50)

    bt_config = config["backtest"]
    engine = BacktestEngine(
        initial_capital=bt_config["initial_capital"],
        commission=bt_config["commission"],
        slippage_pct=bt_config["slippage_pct"],
    )

    result = engine.run(
        price_data=data["price_data"],
        signals=signals,
    )

    # Calculate metrics
    metrics_calc = BacktestMetrics(result["daily_values"])
    metrics = metrics_calc.calculate_all()
    print(metrics_calc.format_report(metrics))

    # Generate charts
    viz = BacktestVisualizer()
    chart_paths = viz.generate_all_charts(result["daily_values"])
    logger.info(f"Charts saved: {chart_paths}")

    return result, metrics


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Bubble Quantitative Assessment")
    parser.add_argument("--step", choices=["fetch", "analyze", "signals", "backtest", "all"], default="all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (skip API calls)")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--tickers", default="config/tickers.yaml", help="Tickers config file path")
    args = parser.parse_args()

    setup_logging(args.verbose)

    config = load_config(args.config)
    tickers_cfg = load_tickers(args.tickers)

    if args.step in ["fetch", "all"]:
        data = step_fetch(config, tickers_cfg, args)

    if args.step == "fetch":
        return

    if args.step in ["analyze", "all"]:
        analysis_results = step_analyze(data, config, tickers_cfg)

    if args.step == "analyze":
        return

    if args.step in ["signals", "all"]:
        signals = step_signals(analysis_results, config)

    if args.step == "signals":
        return

    if args.step in ["backtest", "all"]:
        bt_result, bt_metrics = step_backtest(data, signals, config)

    logger = logging.getLogger("main")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
