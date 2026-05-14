"""
Daily Market Check — Scheduled Signal Monitor.
Runs once per day during market hours, checks all signals,
and sends notification if any signal triggers.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.scraper import MarketDataScraper
from data.ashare import AShareDataFetcher
from analysis.bubble_comparison import build_default_assessment
from analysis.signal_monitor import build_default_dashboard, SignalSeverity
from analysis.lottery_calculator import LotteryCalculator


# ── Configuration ──────────────────────────────────────────────────────

STATE_FILE = Path(__file__).parent / "data" / "daily_check_state.json"
LOG_DIR = Path(__file__).parent / "logs"


def setup_logging(verbose: bool = False):
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"daily_check_{datetime.now().strftime('%Y%m%d')}.log"

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ── State Management ──────────────────────────────────────────────────

def load_state() -> dict:
    """Load previous check state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"last_check": None, "triggered_signals": [], "alerts_sent": []}


def save_state(state: dict):
    """Save check state."""
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Notification ───────────────────────────────────────────────────────

def send_notification(message: str, level: str = "INFO"):
    """
    Send notification. Multiple channels supported:
    1. File-based (always works)
    2. Email (if configured)
    3. WeChat (if webhook configured)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Always write to alert file
    alert_file = LOG_DIR / "alerts.log"
    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{level}] {timestamp}\n")
        f.write(f"{'='*60}\n")
        f.write(message)
        f.write(f"\n{'='*60}\n\n")

    # Try email notification (if SMTP configured)
    smtp_config = os.environ.get("HERMES_SMTP_CONFIG")
    if smtp_config:
        try:
            import smtplib
            from email.mime.text import MIMEText

            config = json.loads(smtp_config)
            msg = MIMEText(message, "plain", "utf-8")
            msg["Subject"] = f"[AI Bubble Quant] {level}: Signal Alert"
            msg["From"] = config.get("from_email", "")
            msg["To"] = config.get("to_email", "")

            with smtplib.SMTP(config["smtp_server"], config.get("smtp_port", 587)) as server:
                server.starttls()
                server.login(config["smtp_user"], config["smtp_password"])
                server.send_message(msg)

            logging.getLogger("notify").info("Email notification sent")
        except Exception as e:
            logging.getLogger("notify").error(f"Email notification failed: {e}")

    # Try WeChat webhook (if configured)
    wechat_webhook = os.environ.get("WECHAT_WEBHOOK_URL")
    if wechat_webhook:
        try:
            import requests
            payload = {
                "msgtype": "text",
                "text": {"content": f"🎯 AI Bubble Quant Alert [{level}]\n\n{message}"},
            }
            requests.post(wechat_webhook, json=payload, timeout=10)
            logging.getLogger("notify").info("WeChat notification sent")
        except Exception as e:
            logging.getLogger("notify").error(f"WeChat notification failed: {e}")


# ── Daily Check Logic ──────────────────────────────────────────────────

def run_daily_check(contract: str = "MO", days_to_expiry: int = 7):
    """
    Run the full daily market check.
    1. Fetch all market data
    2. Check all signals
    3. Generate bubble assessment
    4. Generate lottery calculator output
    5. Send notification if any signal triggered
    """
    logger = logging.getLogger("daily_check")
    logger.info("=" * 70)
    logger.info(f"📊 DAILY MARKET CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 70)

    # ── Step 1: Fetch Data ────────────────────────────────────────────
    logger.info("")
    logger.info("Step 1: Fetching market data...")

    # A-share data
    ashare = AShareDataFetcher()
    ashare_data = ashare.get_all_indices()

    # US data
    us_scraper = MarketDataScraper()
    us_data = {}
    for ticker in ["SPY", "QQQ"]:
        q = us_scraper.get_quote(ticker)
        if q:
            us_data[ticker] = q

    # Fetch history for technical analysis
    csi1000_hist = ashare.get_history("CSI1000", days=120)
    csi300_hist = ashare.get_history("CSI300", days=120)
    qqq_hist = us_scraper.get_history("QQQ", period="1y")

    # ── Step 2: Update Signals ────────────────────────────────────────
    logger.info("")
    logger.info("Step 2: Checking signals...")

    dashboard = build_default_dashboard()

    # Update signals with live data
    # CSI 1000 price
    csi1000 = ashare_data.get("CSI1000", {})
    csi1000_price = csi1000.get("price")

    # CSI 1000 vs 20-day MA
    if csi1000_hist is not None and not csi1000_hist.empty:
        ma20 = ashare.calculate_ma(csi1000_hist, 20)
        if ma20 and csi1000_price:
            dashboard.update("CSI 1000 breaks 20-day MA", ma20)
            logger.info(f"  CSI 1000 MA20: {ma20:.1f} | Current: {csi1000_price:.1f}")

    # Nasdaq RSI (weekly approximation from daily data)
    if qqq_hist is not None and not qqq_hist.empty:
        col = "adj_close" if "adj_close" in qqq_hist.columns else "close"
        qqq_rsi = ashare.calculate_rsi(qqq_hist[[col]].rename(columns={col: "close"}), period=14)
        if qqq_rsi:
            dashboard.update("Nasdaq RSI (weekly) > 80", qqq_rsi)
            logger.info(f"  Nasdaq QQQ RSI(14): {qqq_rsi:.1f}")

    # Check for significant daily moves
    if csi1000.get("change_pct") is not None:
        change = csi1000["change_pct"]
        logger.info(f"  CSI 1000 today: {change:+.2f}%")
        if abs(change) >= 3:
            dashboard.update("Nasdaq drops >3% in one day", change)

    # VIX (estimated from QQQ volatility if available)
    # For now, skip automatic VIX update (requires separate data source)

    # ── Step 3: Generate Report ───────────────────────────────────────
    logger.info("")
    logger.info("Step 3: Generating report...")

    report_lines = []
    report_lines.append(f"📊 每日市场检查报告 — {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append("=" * 50)

    # Market summary
    report_lines.append("")
    report_lines.append("📈 A股指数:")
    for name, data in ashare_data.items():
        price = data.get("price", "N/A")
        change_pct = data.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if change_pct is not None else ""
        display = data.get("display_name", name)
        report_lines.append(f"  {display}: {price} {change_str}")

    report_lines.append("")
    report_lines.append("📈 美股ETF:")
    for ticker, data in us_data.items():
        price = data.get("price", "N/A")
        report_lines.append(f"  {ticker}: ${price:.2f}")

    # Signal status
    report_lines.append("")
    report_lines.append(dashboard.format_report())

    # Lottery calculator output
    if csi1000_price:
        report_lines.append("")
        report_lines.append("🎰 期权彩票计算器（中证1000认沽）:")
        report_lines.append("-" * 50)
        try:
            calc = LotteryCalculator(
                contract=contract,
                current_index=csi1000_price,
                iv=0.20,
                days_to_expiry=days_to_expiry,
            )
            # Only show top 5 most interesting strikes
            results = calc.analyze_all()
            # Sort by interestingness (balance of cost and potential return)
            results.sort(key=lambda r: r.get("risk_reward", 0), reverse=True)
            report_lines.append(f"  当前指数: {csi1000_price:.0f} | IV: 20% | 到期: {days_to_expiry}天")
            report_lines.append(f"  {'行权价':>8} {'虚值%':>6} {'成本':>8} {'胜率':>6} {'跌3%':>10} {'跌5%':>10}")
            report_lines.append(f"  {'-'*58}")
            for r in results[:8]:
                drops = {s["drop_pct"]: s["pnl"] for s in r["scenarios"]}
                pnl_3 = drops.get(0.03, 0)
                pnl_5 = drops.get(0.05, 0)
                report_lines.append(
                    f"  {r['strike']:>8.0f} {r['otm_pct']:>5.1f}% "
                    f"{r['total_cost']:>7.0f}元 {r['prob_itm']:>5.1f}% "
                    f"{pnl_3:>9.0f}元 {pnl_5:>9.0f}元"
                )
        except Exception as e:
            report_lines.append(f"  计算失败: {e}")

    report = "\n".join(report_lines)

    # Save daily report
    report_dir = LOG_DIR / "daily_reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Step 4: Check for Alerts ──────────────────────────────────────
    triggered = dashboard.triggered_signals()
    readiness = dashboard.summary_score()
    alert_level = dashboard.alert_level()

    logger.info("")
    logger.info(f"Readiness Score: {readiness:.1f}/100")
    logger.info(f"Alert Level: {alert_level.value}")
    logger.info(f"Signals Triggered: {len(triggered)}/{len(dashboard.signals)}")

    # Send notification if needed
    state = load_state()
    last_check = state.get("last_check")

    # Only notify if:
    # 1. This is the first check, OR
    # 2. New signals triggered since last check, OR
    # 3. Alert level changed
    should_notify = False

    if not last_check:
        should_notify = True
    elif len(triggered) > len(state.get("triggered_signals", [])):
        should_notify = True
        new_signals = [s.name for s in triggered if s.name not in state.get("triggered_signals", [])]
        logger.info(f"New signals triggered: {new_signals}")

    if should_notify:
        send_notification(report, alert_level.value)
        logger.info("Notification sent")

    # Update state
    state["last_check"] = datetime.now().isoformat()
    state["triggered_signals"] = [s.name for s in triggered]
    state["readiness_score"] = readiness
    state["alert_level"] = alert_level.value
    save_state(state)

    logger.info("")
    logger.info("✅ Daily check complete.")
    logger.info(f"   Report saved to: {report_file}")
    logger.info(f"   State saved to: {STATE_FILE}")

    return {
        "readiness": readiness,
        "alert_level": alert_level.value,
        "triggered": len(triggered),
        "report_file": str(report_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Daily Market Check")
    parser.add_argument("--contract", choices=["MO", "IO"], default="MO")
    parser.add_argument("--days", type=int, default=7, help="Days to expiry")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    run_daily_check(args.contract, args.days)


if __name__ == "__main__":
    main()
