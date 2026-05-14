"""
CFFEX Option Scanner — All-in-One Command Center.
Runs all analysis modules: data fetch, bubble assessment, signal monitoring,
lottery calculator, and crash backtest.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.scraper import MarketDataScraper
from analysis.bubble_comparison import build_default_assessment
from analysis.signal_monitor import build_default_dashboard
from analysis.lottery_calculator import LotteryCalculator
from backtest.crash_backtest import CrashBacktester, HISTORICAL_CRASHES


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def fetch_live_data(scraper: MarketDataScraper) -> dict:
    """Fetch all live data needed for analysis."""
    logger = logging.getLogger("data")
    logger.info("=" * 60)
    logger.info("STEP 1: 获取实时数据（腾讯行情 + 东方财富）")
    logger.info("=" * 60)

    data = {}

    # US indices (for bubble assessment)
    for ticker in ["SPY", "QQQ"]:
        q = scraper.get_quote(ticker)
        if q:
            data[ticker] = q
            logger.info(f"  {ticker}: ${q['price']:.2f} ({q['source']})")
        else:
            logger.warning(f"  {ticker}: 获取失败")

    # A-share indices (via Eastmoney proxy)
    # 沪深300 and 中证1000 via Eastmoney
    em_tickers = {
        "CSI300": "1.000300",  # 东方财富编码
        "CSI1000": "0.000905",
    }

    for name, code in em_tickers.items():
        try:
            # Use Eastmoney push API for A-share indices
            import requests
            url = "https://push2.eastmoney.com/api/qt/stock/get"
            resp = requests.get(url, params={
                "secid": code,
                "fields": "f43,f44,f45,f46,f47,f57,f58",
                "ut": "fa5fd1943c7b386f172d6893dbbd1",
            }, timeout=10)
            d = resp.json().get("data", {})
            if d and d.get("f43"):
                # f43 is price in cents for A-shares
                price = d["f43"] / 100 if d["f43"] > 100 else d["f43"]
                data[name] = {"price": price, "source": "eastmoney"}
                logger.info(f"  {name}: {price:.2f} (eastmoney)")
            else:
                logger.warning(f"  {name}: 无数据")
        except Exception as e:
            logger.warning(f"  {name}: 获取失败 - {e}")

    return data


def run_bubble_comparison(data: dict):
    """Run bubble assessment."""
    logger = logging.getLogger("bubble")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: AI泡沫 vs 2000互联网泡沫 对比分析")
    logger.info("=" * 60)

    assessment = build_default_assessment()

    # Populate with available data
    # Update Shiller CAPE based on SPY level
    spy_price = data.get("SPY", {}).get("price", 540)
    # Rough CAPE estimate from SPY level (CAPE ~ SPY/14.5)
    estimated_cape = spy_price / 14.5
    assessment.indicators[6].current_value = estimated_cape

    # Buffett indicator (rough estimate)
    assessment.indicators[2].current_value = 195

    # Concentration
    assessment.indicators[0].current_value = 23

    # Nasdaq PE
    assessment.indicators[1].current_value = 38

    # AI Capex
    assessment.indicators[3].current_value = 13

    # Retail sentiment
    assessment.indicators[4].current_value = 48

    # IPO returns
    assessment.indicators[5].current_value = 18

    # Margin debt
    assessment.indicators[7].current_value = 3.5

    print(assessment.format_report())
    return assessment


def run_signal_monitor(data: dict):
    """Run signal monitoring dashboard."""
    logger = logging.getLogger("signals")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: 交易信号监控面板")
    logger.info("=" * 60)

    dashboard = build_default_dashboard()

    # Update signals with live data where possible
    qqq = data.get("QQQ", {})
    if qqq.get("price"):
        # Estimate VIX from QQQ price movement (rough proxy)
        # In production, we'd get actual VIX data
        pass

    print(dashboard.format_report())
    return dashboard


def run_lottery_calculator(data: dict, contract: str, days: int):
    """Run the lottery option calculator."""
    logger = logging.getLogger("calculator")
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 4: 🎰 {contract} 认沽期权 彩票计算器")
    logger.info("=" * 60)

    # Determine current index level
    if contract == "MO":
        # 中证1000 — use estimate if not available
        csi1000 = data.get("CSI1000", {}).get("price", 6500)
        current_index = csi1000
    else:
        # IO — 沪深300
        csi300 = data.get("CSI300", {}).get("price", 3900)
        current_index = csi300

    logger.info(f"  {contract} 当前指数: {current_index:.0f}")
    logger.info(f"  到期天数: {days}天")

    calc = LotteryCalculator(
        contract=contract,
        current_index=current_index,
        iv=0.20,
        days_to_expiry=days,
    )

    print(calc.format_report())

    # Show detail for 2-3 interesting strikes
    if contract == "MO":
        # Show a few key strikes for MO
        interesting_strikes = [
            current_index * 0.95,  # 5% OTM
            current_index * 0.90,  # 10% OTM
            current_index * 0.85,  # 15% OTM
        ]
        for k in interesting_strikes:
            k = round(k / 10) * 10  # Round to nearest 10
            print(calc.format_detailed(k))

    return calc


def run_crash_backtest():
    """Run historical crash backtest."""
    logger = logging.getLogger("backtest")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: 历史暴跌回测")
    logger.info("=" * 60)

    backtester = CrashBacktester(iv=0.20, r=0.02, q=0.02)

    # Test buying 5 days before crash, 5% OTM
    results = backtester.run_full_backtest(days_before=5, otm_pct=0.05)
    print(backtester.format_report(results))

    # Also test buying 1 day before (if we had perfect timing)
    logger.info("")
    logger.info("--- 理想情况：暴跌前1天入场 ---")
    results_1d = backtester.run_full_backtest(days_before=1, otm_pct=0.03)
    print(backtester.format_report(results_1d))

    return results


def main():
    parser = argparse.ArgumentParser(description="CFFEX Option Scanner — 指挥中心")
    parser.add_argument("--contract", choices=["MO", "IO"], default="MO")
    parser.add_argument("--days", type=int, default=7, help="Days to expiry")
    parser.add_argument("--step", choices=["all", "data", "bubble", "signals", "calculator", "backtest"], default="all")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    scraper = MarketDataScraper()

    if args.step in ["data", "all"]:
        data = fetch_live_data(scraper)
    else:
        data = {}

    if args.step in ["bubble", "all"]:
        run_bubble_comparison(data)

    if args.step in ["signals", "all"]:
        run_signal_monitor(data)

    if args.step in ["calculator", "all"]:
        run_lottery_calculator(data, args.contract, args.days)

    if args.step in ["backtest", "all"]:
        run_crash_backtest()

    logger = logging.getLogger("main")
    logger.info("")
    logger.info("✅ 分析完成。等待下一个信号触发...")


if __name__ == "__main__":
    main()
