"""
AI Bubble Quant — Professional Trading Terminal (Web UI)
Flask-based dashboard with real-time market data, signal monitoring,
bubble assessment, option calculator, and crash backtest.

Usage: python3 web_terminal.py [--port 5000]
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
import numpy as np


# Recursive converter: numpy types → native Python types
def _to_native(obj):
    """Recursively convert numpy/other non-serializable types to native Python."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.scraper import MarketDataScraper, EastmoneyScraper
from data.ashare import AShareDataFetcher
from analysis.bubble_comparison import build_default_assessment
from analysis.signal_monitor import build_default_dashboard
from analysis.lottery_calculator import LotteryCalculator
from analysis.investment_masters import InvestmentMastersEngine, build_masters_evaluation, build_master_profiles
from backtest.crash_backtest import CrashBacktester, HISTORICAL_CRASHES

# ── Signal-Crash Correlation Data ──────────────────────────────────────

# Historical correlation: which crashes each signal warned about.
# Each entry: {crash_name: "原因说明"} — explains the bottom-layer mechanism
# connecting the indicator to that crash event.
SIGNAL_CRASH_CORR = {
    "VIX 恐慌指数突破30": {
        "2000 Dotcom Burst": "纳指见顶后VIX飙升，恐慌指数从12飙到50+，是泡沫破裂最直接的确认信号。底层原因：机构在估值极端分化后集体止损退出。",
        "2008 Financial Crisis": "雷曼倒闭当天VIX从30跳到45，金融系统性风险瞬间暴露。底层原因：杠杆资金全线爆仓，流动性瞬间蒸发。",
        "2020 COVID Crash": "VIX在33天内从15飙到82，历史最快。底层原因：全球市场同时恐慌，跨资产无差别抛售。",
        "2022 Tech Selloff": "VIX多次触及30+，反映连续加息预期下的持续恐慌。底层原因：高估值成长股对利率敏感度极高。",
        "2024 Aug Yen Carry Unwind": "VIX单日暴涨150%至20+。底层原因：日元套利交易大规模平仓引发全球流动性冲击。",
    },
    "美债10年期收益率突破4.8%": {
        "2000 Dotcom Burst": "1999-2000年美联储加息，长端利率上行挤压科技股估值。底层原因：利率上行直接降低DCF模型远期现金流的现值。",
        "2022 Tech Selloff": "10年期国债从1.5%飙到4.2%，科技股杀估值的主因。底层原因：无风险利率上升→折现率上升→高成长股估值压缩。",
    },
    "美元兑人民币突破7.35": {
        "2015 China Crash": "人民币汇率波动叠加资本外流，加速A股下跌。底层原因：外汇市场压力→外资撤离→流动性双重打击。",
    },
    "纳斯达克RSI周线超买>80": {
        "2000 Dotcom Burst": "纳指周线RSI在2000年1月触及89，历史极值。底层原因：全市场一致做多，再无新增资金推动上涨。",
        "2022 Tech Selloff": "2021年底纳指RSI再度超买后开始杀估值。底层原因：流动性拐点到来，超买状态不可维持。",
    },
    "中证1000跌破20日均线": {
        "2015 China Crash": "中证1000前身指数跌破20日均线后连续暴跌。底层原因：杠杆资金平仓引发的连锁下跌。",
        "2015 China Crash": "千股跌停时中小盘流动性最先枯竭。底层原因：配资清理→强制平仓→踩踏式下跌。",
    },
    "纳指单日暴跌>3%": {
        "2000 Dotcom Burst": "2000年4月4日纳指单日跌7.9%，是后续漫长熊市的开端。底层原因：趋势一旦逆转，止损盘连环触发。",
        "2018 Q4 Flash Crash": "2018年12月多次单日暴跌3%+，VIX飙升至36。底层原因：ETF自动止损+程序化交易加速下跌。",
        "2020 COVID Crash": "3月多次单日跌5-9%，熔断机制触发。底层原因：恐慌性抛售+无差别去杠杆。",
    },
    "A股AI概念涨停家数>20": {
        "2015 China Crash": "2015年互联网+概念连续涨停是泡沫见顶的明显信号。底层原因：散户一致看多→买盘耗尽→反转开始。",
    },
    "期权Put/Call比率<0.6": {
        "2000 Dotcom Burst": "2000年初Put/Call比率低于0.5，市场几乎无人看空。底层原因：过度乐观=反向指标，流动性拐点出现时无人对冲。",
        "2008 Financial Crisis": "危机前Put/Call比率持续低迷，机构普遍不加保护。底层原因：系统性风险被完全定价忽略。",
    },
    "英伟达营收增速降至<40%": {
        "2022 Tech Selloff": "2022年Q2 NVDA数据中心营收增速低于预期，引发AI芯片链集体下跌。底层原因：AI基建周期见顶→估值锚崩塌→板块重定价。",
    },
    "科技巨头削减AI资本开支": {
        "2022 Tech Selloff": "Meta/Google等2022年Q4开始下调资本开支指引。底层原因：AI投资回报率不及预期→叙事瓦解→杀估值。",
    },
    "美联储释放鹰派信号": {
        "2000 Dotcom Burst": "1999-2000年连续加息6次。底层原因：流动性收紧→估值模型分母端恶化→泡沫破裂。",
        "2018 Q4 Flash Crash": "鲍威尔'仍远离正常水平'讲话引发市场暴跌。底层原因：加息节奏超预期+QT加速。",
        "2022 Tech Selloff": "美联储一年内加息425bp+缩表。底层原因：零利率时代终结，高估值成长股首当其冲。",
    },
    "中国科技监管政策收紧": {
        "2015 China Crash": "2015年场外配资清理政策直接触发踩踏。底层原因：政策去杠杆→杠杆资金被迫平仓→连锁反应。",
    },
}


def get_signal_crash_correlation(signal_name):
    """Get the historical correlation for a signal with crash events."""
    return SIGNAL_CRASH_CORR.get(signal_name, {})


# ── Flask App ──────────────────────────────────────────────────────────

app = Flask(__name__)

# Cache for data — initialized with empty shell so frontend can render immediately
_EMPTY_DATA = {
    "timestamp": None,
    "ashare": [],
    "us": [],
    "bubble": {"overall": 0, "verdict": "等待数据...", "indicators": []},
    "signals": {"score": 0, "verdict": "", "alert_level": "INFO", "triggered_count": 0, "total_count": 12, "triggered_list": [], "list": [], "matrix": []},
    "calculator": {"1000": {"index": 0, "list": []}, "300": {"index": 0, "list": []}, "50": {"index": 0, "list": []}},
    "backtest": {"results": [], "total_pnl": 0, "wins": 0, "total": 0},
    "masters": {"masters": [], "portfolio_decision": {}},
}

# Fast quotes cache — 30s TTL for market prices + bubble + calculator + masters
_QUOTES_CACHE_EMPTY = {"timestamp": None, "ashare": [], "us": [], "bubble": {"overall": 0, "verdict": "等待数据...", "indicators": []}}
_quotes_cache = dict(_QUOTES_CACHE_EMPTY)
_quotes_ttl = 30

# Full data cache — 1h TTL (includes signals + backtest + history)
_data_cache = dict(_EMPTY_DATA)
_data_ttl = 3600  # 1 hour

# Master profiles cache — long TTL (static track records + market-data-driven analysis)
_profiles_cache = {}
_profiles_ttl = 7200  # 2 hours

_cache_ttl = 60  # seconds (legacy, kept for compatibility)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("web_terminal")


# ── Investment Masters Helper ─────────────────────────────────────────

def _fetch_masters(bubble_result, dashboard, us_list):
    """Fetch investment masters evaluation from current market data."""
    try:
        bubble_score = bubble_result.get("overall", 50)
        # Find CAPE from bubble indicators
        cape = 30
        for ind in bubble_result.get("indicators", []):
            if "CAPE" in str(ind.get("name", "")) or "席勒" in str(ind.get("name", "")):
                cape = ind.get("current") or 30
                break

        # VIX
        vix_val = 20
        for s in dashboard.signals:
            if "VIX" in s.name:
                vix_val = s.current_value or 20
                break

        # SPY PE (approximate from price / 22)
        spy_price = next((u.get("price") for u in us_list if u["name"] == "SPY"), None)
        spy_pe = (spy_price / 22) if spy_price else 25
        spy_change = next((u.get("change_pct", 0) for u in us_list if u["name"] == "SPY"), 0)
        qqq_change = next((u.get("change_pct", 0) for u in us_list if u["name"] == "QQQ"), 0)
        nvda_change = next((u.get("change_pct", 0) for u in us_list if u["name"] == "NVDA"), 0)

        # QQQ RSI
        qqq_rsi = 50
        for s in dashboard.signals:
            if "RSI" in s.name:
                qqq_rsi = s.current_value or 50
                break

        # Concentration (approximate)
        concentration = 18  # default assumption

        # Buffett indicator (market cap / GDP)
        buffett_indicator = 180  # default approximation

        # LLM enabled via env var (disabled by default to avoid slow responses)
        use_llm = os.environ.get("USE_LLM_FOR_MASTERS", "").lower() == "true"

        return build_masters_evaluation(
            bubble_score=bubble_score,
            cape=cape,
            vix=vix_val,
            spy_pe=round(spy_pe, 1),
            spy_change=round(spy_change, 1),
            qqq_change=round(qqq_change, 1),
            qqq_rsi=round(qqq_rsi, 1),
            nvda_change=round(nvda_change, 1),
            concentration=concentration,
            buffett_indicator=buffett_indicator,
            use_llm=use_llm,
        )
    except Exception as e:
        logger.error(f"Error in masters evaluation: {e}")
        return {"error": str(e), "masters": [], "portfolio_decision": {}}


# ── Data Fetching Functions ───────────────────────────────────────────

def get_quotes_data():
    """Fast quotes-only fetch: A-share + US prices + VIX + bubble + calculator + masters.

    No historical data requests (no QQQ RSI, no CSI1000 MA).
    30s TTL cache.
    """
    now = time.time()
    if _quotes_cache["timestamp"] and (now - _quotes_cache["timestamp"]) < _quotes_ttl:
        return _quotes_cache

    # A-share data
    ashare_fetcher = AShareDataFetcher()
    ashare_data = ashare_fetcher.get_all_indices()
    ashare_list = []
    for name, d in ashare_data.items():
        ashare_list.append({
            "name": d.get("display_name", name),
            "code": name,
            "price": d.get("price"),
            "change_pct": d.get("change_pct"),
            "high": d.get("high"),
            "low": d.get("low"),
            "volume": d.get("volume"),
        })

    # US data — split into indices (first row) and stocks (second row)
    us_scraper = MarketDataScraper()

    # Row 1: Major indices via EastMoney (m:100)
    us_indices = []
    _us_index_map = [
        ("100.SPX", "SPX"),
        ("100.DJIA", "DJIA"),
        ("100.NDX", "NDX"),
    ]
    try:
        _em = EastmoneyScraper()
        for _secid, _name in _us_index_map:
            try:
                _resp = _em.session.get(
                    "https://push2.eastmoney.com/api/qt/stock/get",
                    params={
                        "secid": _secid,
                        "fields": "f43,f44,f45,f170",
                        "ut": "fa5fd1943c7b386f172d6893dbbd1",
                    },
                    timeout=10,
                )
                _jd = _resp.json().get("data")
                if _jd and _jd.get("f43"):
                    us_indices.append({
                        "name": _name,
                        "price": round(_jd["f43"] / 100.0, 2),
                        "change_pct": round(_jd.get("f170", 0) / 100.0, 2),
                        "high": _jd.get("f44", 0) / 100.0,
                        "low": _jd.get("f45", 0) / 100.0,
                    })
                time.sleep(0.15)
            except Exception as _ie:
                logger.warning(f"Failed to fetch {_name}: {_ie}")
    except Exception as e:
        logger.warning(f"Failed to fetch US indices: {e}")

    # Row 2: ETFs + VIX via Tencent
    us_etfs = []
    for ticker in ["SPY", "QQQ", "VIX"]:
        q = us_scraper.get_quote(ticker)
        if q:
            us_etfs.append({
                "name": ticker,
                "price": q.get("price"),
                "change_pct": q.get("change_pct"),
                "high": q.get("high"),
                "low": q.get("low"),
            })

    # Row 3: Magnificent 7 via Tencent
    us_mag7 = []
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]:
        q = us_scraper.get_quote(ticker)
        if q:
            us_mag7.append({
                "name": ticker,
                "price": q.get("price"),
                "change_pct": q.get("change_pct"),
                "high": q.get("high"),
                "low": q.get("low"),
            })

    # Bubble assessment
    assessment = build_default_assessment()
    spy_price = next((u["price"] for u in us_etfs if u["name"] == "SPY"), 540)
    if spy_price:
        assessment.indicators[6].current_value = spy_price / 14.5
    bubble_result = assessment.score()

    _quotes_cache.update({
        "timestamp": now,
        "ashare": ashare_list,
        "us_indices": us_indices,
        "us_etfs": us_etfs,
        "us_mag7": us_mag7,
        "us_stocks": us_etfs + us_mag7,
        # Backward compat: combined list
        "us": us_indices + us_etfs + us_mag7,
        "bubble": bubble_result,
    })

    return _quotes_cache


def get_all_data(force_fresh=False):
    """Fetch all data for the dashboard.

    If force_fresh=False and cached data exists, return it immediately (even if stale).
    If force_fresh=True, execute full data fetch.
    TTL: 3600s (1 hour). Quotes layer refreshes every 30s via /api/quotes.
    """
    now = time.time()
    if _data_cache["timestamp"] and (now - _data_cache["timestamp"]) < _data_ttl and not force_fresh:
        return _data_cache

    logger.info("Fetching fresh data for dashboard...")

    # Get fast quotes data (A-share + US + bubble) — reused from quotes cache
    quotes = get_quotes_data()
    ashare_fetcher = AShareDataFetcher()
    ashare_list = quotes["ashare"]
    us_list = quotes["us"]
    bubble_result = quotes["bubble"]

    # ── Signal dashboard — needs historical data (slow part) ──
    us_scraper = MarketDataScraper()
    dashboard = build_default_dashboard()

    # VIX
    vix_data = us_scraper.get_quote("VIX")
    if vix_data and vix_data.get("price"):
        vix_val = vix_data["price"]
        dashboard.update("VIX 恐慌指数突破30", vix_val)

    # QQQ RSI (daily proxy)
    try:
        qqq_hist = us_scraper.get_history("QQQ", period="1y")
        if qqq_hist is not None and not qqq_hist.empty:
            col = "adj_close" if "adj_close" in qqq_hist.columns else "close"
            qqq_rsi = ashare_fetcher.calculate_rsi(qqq_hist[[col]].rename(columns={col: "close"}), period=14)
            if qqq_rsi:
                dashboard.update("纳斯达克RSI周线超买>80", qqq_rsi)
    except Exception:
        pass

    # CSI 1000 vs 20-day MA
    csi1000 = next((a for a in ashare_list if a["code"] == "CSI1000"), None)
    if csi1000 and csi1000.get("price"):
        try:
            csi1000_hist = ashare_fetcher.get_history("CSI1000", days=120)
            if csi1000_hist is not None and not csi1000_hist.empty:
                ma20 = ashare_fetcher.calculate_ma(csi1000_hist, 20)
                if ma20 and csi1000["price"] < ma20:
                    gap_pct = (csi1000["price"] - ma20) / ma20 * 100
                    dashboard.update("中证1000跌破20日均线", gap_pct)
        except Exception:
            pass

    # QQQ daily change
    try:
        qqq_quote = us_scraper.get_quote("QQQ")
        if qqq_quote and qqq_quote.get("change_pct") is not None:
            dashboard.update("纳指单日暴跌>3%", qqq_quote["change_pct"])
    except Exception:
        pass

    signals_result = {"overall": dashboard.summary_score()}
    signals_result["verdict"] = dashboard.alert_level().value
    signals_result["alert_level"] = dashboard.alert_level().value
    signals_result["triggered_count"] = len(dashboard.triggered_signals())
    signals_result["total_count"] = len(dashboard.signals)

    # Triggered signals detail
    triggered_detail = []
    for s in dashboard.triggered_signals():
        triggered_detail.append({
            "name": s.name,
            "current_value": s.current_value,
            "threshold": s.threshold,
            "direction": s.direction,
            "weight": s.weight,
        })
    signals_result["triggered_list"] = triggered_detail

    # Build signal list
    signal_list = []
    for s in dashboard.signals:
        signal_list.append({
            "name": s.name,
            "category": s.category.value,
            "description": s.description,
            "current_value": s.current_value,
            "threshold": s.threshold,
            "triggered": s.triggered,
            "severity": s.severity.value,
            "weight": s.weight,
            "direction": s.direction,
        })

    # Lottery calculators for all 3 contracts (uses live index prices)
    def calc_options(contract, index_price):
        if not index_price:
            return []
        try:
            c = LotteryCalculator(contract=contract, current_index=index_price, iv=0.20, days_to_expiry=7)
            results = c.analyze_all()
            results.sort(key=lambda r: r.get("strike", 0), reverse=True)
            return results[:6]
        except Exception:
            return []

    csi1000_price = next((a["price"] for a in ashare_list if a["code"] == "CSI1000"), 6500)
    csi300_price = next((a["price"] for a in ashare_list if a["code"] == "CSI300"), 4000)
    sse50_price = next((a["price"] for a in ashare_list if a["code"] == "SSE50"), 2700)

    calc_1000 = calc_options("MO", csi1000_price)
    calc_300 = calc_options("IO", csi300_price)
    calc_50 = calc_options("IH", sse50_price)

    # Backtest
    backtester = CrashBacktester(iv=0.20, r=0.02, q=0.02)
    bt_results = backtester.run_full_backtest(days_before=5, otm_pct=0.05)
    bt_list = []
    for r in bt_results:
        bt_list.append({
            "name": r.crash.name,
            "description": r.crash.description,
            "index_before": r.index_at_entry,
            "strike": r.put_strike,
            "premium": r.premium_paid,
            "index_after": r.index_at_exit,
            "exit_premium": r.exit_premium,
            "pnl_pct": r.pnl_pct,
            "pnl_money": r.pnl_money,
        })
    total_pnl = sum(r["pnl_money"] for r in bt_list)
    wins = sum(1 for r in bt_list if r["pnl_pct"] > 0)

    # Build real signal-event correlation matrix
    signal_event_matrix = []
    for s in dashboard.signals:
        corr = get_signal_crash_correlation(s.name)
        event_values = {}
        event_notes = {}
        for bt in bt_list:
            if bt["name"] in corr:
                event_values[bt["name"]] = int(round(bt["pnl_pct"], 0))
                event_notes[bt["name"]] = corr[bt["name"]]
            else:
                event_values[bt["name"]] = None
                event_notes[bt["name"]] = ""
        signal_event_matrix.append({
            "name": s.name,
            "description": s.description,
            "current_value": s.current_value,
            "threshold": s.threshold,
            "triggered": s.triggered,
            "direction": s.direction,
            "events": event_values,
            "notes": event_notes,
        })

    # Investment masters evaluation
    _t0 = time.time()
    masters_data = _fetch_masters(bubble_result, dashboard, us_list)
    logger.info(f" Masters evaluation: {time.time() - _t0:.2f}s")

    _data_cache.update({
        "timestamp": now,
        "ashare": ashare_list,
        "us_indices": quotes.get("us_indices", []),
        "us_etfs": quotes.get("us_etfs", []),
        "us_mag7": quotes.get("us_mag7", []),
        "us_stocks": quotes.get("us_stocks", []),
        "us": us_list,
        "bubble": bubble_result,
        "signals": {
            "score": signals_result["overall"],
            "verdict": signals_result.get("verdict", ""),
            "alert_level": signals_result["alert_level"],
            "triggered_count": signals_result["triggered_count"],
            "total_count": signals_result["total_count"],
            "triggered_list": signals_result["triggered_list"],
            "list": signal_list,
            "matrix": signal_event_matrix,
        },
        "calculator": {
            "1000": {"index": csi1000_price, "list": calc_1000},
            "300": {"index": csi300_price, "list": calc_300},
            "50": {"index": sse50_price, "list": calc_50},
        },
        "backtest": {
            "results": bt_list,
            "total_pnl": total_pnl,
            "wins": wins,
            "total": len(bt_list),
        },
        "masters": masters_data,
    })

    return _data_cache


# ── HTML Template ─────────────────────────────────────────────────────

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI BUBBLE QUANT — 专业交易终端</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1a2332;
            --bg-hover: #243447;
            --border: #2d3748;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --red: #ef4444;
            --red-bg: rgba(239, 68, 68, 0.1);
            --green: #10b981;
            --green-bg: rgba(16, 185, 129, 0.1);
            --yellow: #f59e0b;
            --yellow-bg: rgba(245, 158, 11, 0.1);
            --purple: #8b5cf6;
            --cyan: #06b6d4;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 13px;
            line-height: 1.5;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .mono { font-family: 'SF Mono', 'SF Mono-Regular', 'Fira Code', 'Consolas', monospace; font-variant-numeric: tabular-nums; }

        /* ── Header ─────────────────────────────── */
        .header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-left { display: flex; align-items: center; gap: 16px; }

        .logo {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 2px;
            background: linear-gradient(135deg, var(--accent), var(--cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-bar {
            display: flex;
            gap: 20px;
            font-size: 11px;
            color: var(--text-muted);
        }

        .status-item { display: flex; align-items: center; gap: 6px; }
        .status-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--green);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .clock {
            font-size: 14px;
            font-weight: 600;
            color: var(--cyan);
        }

        .refresh-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 6px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 12px;
            transition: all 0.2s;
        }
        .refresh-btn:hover {
            background: var(--accent);
            border-color: var(--accent);
        }

        /* ── Layout ─────────────────────────────── */
        .main-container {
            display: grid;
            grid-template-columns: 1fr;
            height: calc(100vh - 96px);
        }

        /* ── Ticker Bar ────────────────────────── */
        .ticker-bar {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            align-items: center;
            gap: 16px;
            overflow: hidden;
        }

        .ticker-section {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: nowrap;
            overflow-x: auto;
            flex: 1;
            min-width: 0;
        }

        .ticker-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            white-space: nowrap;
            flex-shrink: 0;
        }

        .ticker-divider {
            width: 1px;
            height: 24px;
            background: var(--border);
            flex-shrink: 0;
        }

        .ticker-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 4px;
            white-space: nowrap;
            flex-shrink: 0;
            font-size: 12px;
        }

        .ticker-item-name {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .ticker-item-price {
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            font-size: 12px;
        }

        .ticker-item-change {
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 11px;
            padding: 1px 5px;
            border-radius: 3px;
            font-weight: 600;
        }

        /* Stock rows — slightly smaller to de-emphasize */
        #us-etf-ticker .ticker-item-name,
        #us-mag7-ticker .ticker-item-name {
            font-size: 10px;
            color: var(--text-muted);
        }
        #us-etf-ticker .ticker-item-price,
        #us-mag7-ticker .ticker-item-price {
            font-size: 11px;
        }
        #us-etf-ticker .ticker-item-change,
        #us-mag7-ticker .ticker-item-change {
            font-size: 10px;
        }

        /* ── Main Content ───────────────────────── */
        .main-content {
            padding: 12px 16px;
            overflow-y: auto;
            overflow-x: visible;
            min-width: 0;
        }

        /* Row 1: Bubble (50%) | Right: Signal (25%) + Readiness (25%) */
        .row-1 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
            align-items: stretch;
        }
        .row-1 > * { min-width: 0; }

        .right-half {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            align-items: stretch;
        }
        .right-half > * { min-width: 0; }

        /* Row 2: Calculator 3 equal columns */
        .row-2 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
            align-items: stretch;
        }
        .row-2 > * { min-width: 0; }

        /* Row 3: Backtest table (50%) | Chart (50%) */
        .row-3 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            align-items: stretch;
        }
        .row-3 > * { min-width: 0; }

        /* Row 2.5: Investment Masters (full width) */
        .row-masters {
            margin-bottom: 16px;
        }

        .masters-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px 10px 0 0;
            flex-wrap: wrap;
            gap: 8px;
        }

        .masters-header-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .masters-header-signal {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .masters-consensus-badge {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .masters-consensus-badge.bullish { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
        .masters-consensus-badge.bearish { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
        .masters-consensus-badge.neutral { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }

        .masters-consensus-score {
            font-size: 11px;
            color: var(--text-muted);
        }

        .masters-consensus-bar {
            width: 100px;
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }
        .masters-consensus-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }

        .masters-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            padding: 8px 0;
        }

        .master-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            min-height: 110px;
            position: relative;
        }
        .master-card:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .master-card.bullish { border-left: 3px solid var(--green); }
        .master-card.bearish { border-left: 3px solid var(--red); }
        .master-card.neutral { border-left: 3px solid var(--yellow); }

        .master-card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .master-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 700;
            flex-shrink: 0;
        }

        .master-name-group {
            flex: 1;
            min-width: 0;
        }

        .master-name-en {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .master-name-zh {
            font-size: 10px;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .master-live-badge {
            font-size: 8px;
            padding: 1px 4px;
            border-radius: 3px;
            background: #22c55e20;
            color: #22c55e;
            border: 1px solid #22c55e40;
            font-weight: 600;
            letter-spacing: 0.5px;
            flex-shrink: 0;
        }

        .master-signal-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }

        .master-signal {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .master-signal.bullish { color: var(--green); }
        .master-signal.bearish { color: var(--red); }
        .master-signal.neutral { color: var(--yellow); }

        .master-confidence {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .master-confidence-bar {
            flex: 1;
            height: 4px;
            background: var(--bg-primary);
            border-radius: 2px;
            overflow: hidden;
        }
        .master-confidence-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        .master-confidence-fill.bullish { background: var(--green); }
        .master-confidence-fill.bearish { background: var(--red); }
        .master-confidence-fill.neutral { background: var(--yellow); }

        .master-confidence-value {
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            flex-shrink: 0;
        }

        .master-reasoning {
            font-size: 10px;
            color: var(--text-secondary);
            line-height: 1.4;
            margin-top: auto;
            padding-top: 4px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* Portfolio Manager Summary */
        .pm-summary {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0 0 10px 10px;
            padding: 12px 16px;
        }

        .pm-top {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .pm-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .pm-signal-badge {
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }
        .pm-signal-badge.bullish { background: var(--green-bg); color: var(--green); }
        .pm-signal-badge.bearish { background: var(--red-bg); color: var(--red); }
        .pm-signal-badge.neutral { background: var(--yellow-bg); color: var(--yellow); }

        .pm-action {
            font-size: 11px;
            color: var(--text-secondary);
        }
        .pm-action strong {
            color: var(--text-primary);
        }

        .pm-vote-bar {
            display: flex;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 8px;
            background: var(--bg-primary);
        }
        .pm-vote-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: 600;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            min-width: 20px;
        }
        .pm-vote-segment.bullish { background: var(--green); }
        .pm-vote-segment.bearish { background: var(--red); }
        .pm-vote-segment.neutral { background: var(--yellow); }

        .pm-reasons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .pm-reasons-bull h4, .pm-reasons-bear h4 {
            font-size: 11px;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .pm-reasons-bull h4 { color: var(--green); }
        .pm-reasons-bear h4 { color: var(--red); }

        .pm-reasons ul {
            list-style: none;
            padding: 0;
        }
        .pm-reasons li {
            font-size: 10px;
            color: var(--text-secondary);
            padding: 2px 0;
            padding-left: 12px;
            position: relative;
        }
        .pm-reasons-bull li::before {
            content: "▲";
            position: absolute;
            left: 0;
            color: var(--green);
            font-size: 8px;
        }
        .pm-reasons-bear li::before {
            content: "▼";
            position: absolute;
            left: 0;
            color: var(--red);
            font-size: 8px;
        }

        /* Responsive for masters */
        @media (max-width: 1200px) {
            .masters-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        @media (max-width: 800px) {
            .masters-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .pm-reasons {
                grid-template-columns: 1fr;
            }
        }

        /* ── Cards ─────────────────────────── */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            transition: border-color 0.2s;
            min-width: 0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .card:hover { border-color: var(--accent); }

        /* Panel content inside cards fills remaining space */
        .card > [id$="-panel"],
        .card > [id^="calc-"],
        .card > [id$="-list"] {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }

        .card-title {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            flex-shrink: 0;
        }

        .card-body {
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }

        /* ── Color helpers ─────────────────── */
        .up { color: var(--red); }
        .down { color: var(--green); }
        .up-bg { background: var(--red-bg); color: var(--red); }
        .down-bg { background: var(--green-bg); color: var(--green); }

        /* ── Tooltips ──────────────────────── */
        .tip-inline {
            position: relative;
            cursor: help;
            display: inline-flex;
            align-items: center;
            margin-left: 4px;
        }
        .tip-inline .tip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            border: 1px solid var(--text-muted);
            color: var(--text-muted);
            font-size: 9px;
            line-height: 1;
            font-weight: 700;
            transition: all 0.15s;
        }
        .tip-inline:hover .tip-icon {
            border-color: var(--accent);
            color: var(--accent);
            background: var(--accent-glow);
        }
        .tip-inline .tip-text {
            display: none;
            position: fixed;
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 1.5;
            white-space: pre-wrap;
            width: 360px;
            z-index: 999999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            pointer-events: none;
        }
        .tip-inline:hover .tip-text {
            display: block;
        }

        /* Global tooltip overlay — rendered at body level to avoid clipping */
        .tooltip-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 999999;
            pointer-events: none;
            display: none;
        }
        .tooltip-overlay.active {
            display: block;
        }
        .tooltip-overlay .tooltip-content {
            position: absolute;
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 1.5;
            white-space: pre-wrap;
            max-width: 360px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            pointer-events: auto;
            z-index: 999999;
        }

        .sico { display: inline-block; width: 14px; height: 14px; vertical-align: -2px; flex-shrink: 0; }
        .mono { font-family: 'SF Mono', 'SF Mono-Regular', 'Fira Code', 'Consolas', monospace; font-variant-numeric: tabular-nums; }

        /* ── Loading ────────────────────────────── */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 60px;
            color: var(--text-muted);
            font-size: 12px;
        }

        .loading::after {
            content: '';
            display: inline-block;
            width: 16px;
            height: 16px;
            min-width: 16px;
            min-height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-left: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Bubble Score ─────────────────────── */
        .bubble-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            min-height: 0;
            padding: 8px;
        }

        .bubble-score {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 8px;
            width: 100%;
        }

        .score-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 3px solid;
            flex-shrink: 0;
        }

        .score-value {
            font-size: 32px;
            font-weight: 700;
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-variant-numeric: tabular-nums;
        }

        .score-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .bubble-indicators {
            width: 100%;
            max-width: 100%;
        }

        .indicator-bar { margin-bottom: 4px; }

        .indicator-label {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            margin-bottom: 2px;
        }
        .indicator-label span:last-child {
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 9px;
        }

        .indicator-track {
            height: 5px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }

        .indicator-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s;
        }

        /* ── Readiness ────────────────────────── */
        .readiness-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            min-height: 0;
            padding: 8px;
        }

        .readiness-meter {
            text-align: center;
            padding: 6px 0;
            margin-bottom: 8px;
        }

        .meter-value {
            font-size: 42px;
            font-weight: 700;
            line-height: 1;
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-variant-numeric: tabular-nums;
        }

        .meter-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 3px;
        }

        .alert-box {
            padding: 6px 14px;
            border-radius: 5px;
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            margin-bottom: 6px;
        }

        .alert-high { background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }
        .alert-moderate { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
        .alert-low { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
        .alert-info { background: var(--bg-card); color: var(--text-secondary); border: 1px solid var(--border); }

        .sig-table-wrap::-webkit-scrollbar { width: 4px; }
        .sig-table-wrap::-webkit-scrollbar-track { background: transparent; }
        .sig-table-wrap::-webkit-scrollbar-thumb { background: #2a3444; border-radius: 2px; }
        .sig-table-wrap::-webkit-scrollbar-thumb:hover { background: #3a4a5a; }

        .calc-table-wrap::-webkit-scrollbar { width: 4px; }
        .calc-table-wrap::-webkit-scrollbar-track { background: transparent; }
        .calc-table-wrap::-webkit-scrollbar-thumb { background: #2a3444; border-radius: 2px; }
        .calc-table-wrap::-webkit-scrollbar-thumb:hover { background: #3a4a5a; }

        /* ── Signal Table ─────────────────────── */
        .sig-table-wrap {
            overflow-y: auto;
            max-height: 280px;
            width: 100%;
        }

        .sig-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
            table-layout: fixed;
        }

        .sig-table th {
            text-align: right;
            padding: 3px 4px;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            font-size: 8px;
            text-transform: uppercase;
            white-space: nowrap;
            position: sticky;
            top: 0;
            background: var(--bg-card);
            z-index: 1;
        }
        .sig-table th,
        .sig-table td {
            box-sizing: border-box;
        }
        .sig-table th:first-child { text-align: left; }
        .sig-table th:last-child { text-align: right; color: var(--accent); }
        .sig-table th .tip-icon { width: 10px; height: 10px; font-size: 7px; }
        .sig-table th .tip-text { width: 240px; font-size: 10px; }

        .sig-table td {
            text-align: right;
            padding: 3px 4px;
            border-bottom: 1px solid rgba(45, 55, 72, 0.2);
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-size: 9px;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .sig-table td:first-child {
            text-align: left;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .sig-table td:last-child {
            font-weight: 600;
        }
        .sig-table tr:hover { background: var(--bg-hover); }

        /* ── Calculator Table ─────────────────── */
        .calc-table-wrap {
            overflow-y: auto;
            max-height: 180px;
        }

        .calc-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
            table-layout: auto;
        }
        .calc-table th,
        .calc-table td {
            box-sizing: border-box;
        }
        .calc-table th {
            text-align: right;
            padding: 3px 2px;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            font-size: 8px;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .calc-table th:first-child { text-align: left; }
        .calc-table th:last-child { text-align: right; color: var(--accent); }

        .calc-table td {
            text-align: right;
            padding: 3px 2px;
            border-bottom: 1px solid rgba(45, 55, 72, 0.2);
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-size: 9px;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .calc-table td:first-child { text-align: left; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .calc-table tr:hover { background: var(--bg-hover); }

        /* ── Backtest Table ───────────────────── */
        .bt-table-wrap {
            overflow-y: auto;
            max-height: 200px;
        }

        .bt-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 9px;
        }

        .bt-table th {
            text-align: right;
            padding: 3px 3px;
            color: var(--text-muted);
            font-weight: 500;
            border-bottom: 1px solid var(--border);
            font-size: 8px;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .bt-table th:first-child { text-align: left; }

        .bt-table td {
            text-align: right;
            padding: 3px 3px;
            border-bottom: 1px solid rgba(45, 55, 72, 0.2);
            font-family: 'SF Mono', 'SF Mono-Regular', 'Consolas', monospace;
            font-size: 9px;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .bt-table td:first-child { text-align: left; font-family: -apple-system, BlinkMacSystemFont, sans-serif; white-space: normal; word-break: break-word; max-width: 70px; }

        /* ── Chart ────────────────────────────── */
        .chart-container {
            position: relative;
            height: 200px;
        }

        /* ── Scrollbars ─────────────────────────── */
        .main-content::-webkit-scrollbar { width: 6px; }
        .main-content::-webkit-scrollbar-track { background: var(--bg-primary); }
        .main-content::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
        .main-content::-webkit-scrollbar-thumb:hover { background: #4a5568; }

        .scrollable::-webkit-scrollbar { width: 4px; }
        .scrollable::-webkit-scrollbar-track { background: transparent; }
        .scrollable::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }
        .scrollable::-webkit-scrollbar-thumb:hover { background: #4a5568; }

        .ticker-section::-webkit-scrollbar { height: 3px; }
        .ticker-section::-webkit-scrollbar-track { background: transparent; }
        .ticker-section::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

        /* ── Responsive ─────────────────────────── */
        @media (max-width: 1200px) {
            .row-1 { grid-template-columns: 1fr; }
            .right-half { grid-template-columns: 1fr 1fr; }
            .row-2 { grid-template-columns: 1fr; }
        }
        @media (max-width: 800px) {
            .right-half { grid-template-columns: 1fr; }
            .row-3 { grid-template-columns: 1fr; }
        }

        /* ── Master Detail Popup ────────────────── */
        .master-popup-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9999;
            pointer-events: none;
        }
        .master-popup-overlay.active { display: block; }

        .master-popup {
            position: fixed;
            width: 480px;
            max-height: 80vh;
            overflow-y: auto;
            background: #0f1724;
            border: 1px solid #2d3748;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 30px rgba(59, 130, 246, 0.08);
            z-index: 10000;
            pointer-events: auto;
            opacity: 0;
            transform: translateY(8px) scale(0.97);
            transition: opacity 0.2s ease, transform 0.2s ease;
        }
        .master-popup.visible {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
        .master-popup::-webkit-scrollbar { width: 4px; }
        .master-popup::-webkit-scrollbar-track { background: transparent; }
        .master-popup::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }

        .master-popup-header {
            padding: 16px 20px 12px;
            border-bottom: 1px solid rgba(45, 55, 72, 0.5);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .master-popup-avatar {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: #1e3a5f;
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 700;
            flex-shrink: 0;
        }
        .master-popup-name {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }
        .master-popup-name-sub {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .master-popup-signal-badge {
            margin-left: auto;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .master-popup-signal-badge.bullish { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .master-popup-signal-badge.bearish { background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.3); }
        .master-popup-signal-badge.neutral { background: rgba(234, 179, 8, 0.15); color: var(--yellow); border: 1px solid rgba(234, 179, 8, 0.3); }

        .master-popup-section {
            padding: 12px 20px;
            border-bottom: 1px solid rgba(45, 55, 72, 0.3);
        }
        .master-popup-section:last-child { border-bottom: none; }

        .master-popup-section-title {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent);
            font-weight: 600;
            margin-bottom: 8px;
        }
        .master-popup-philosophy {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .master-popup-tags {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        .master-popup-tag {
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .master-popup-analysis {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.7;
            white-space: pre-line;
        }
        .master-popup-analysis strong {
            color: var(--text-primary);
            font-size: 13px;
        }

        .master-popup-track-record {
            list-style: none;
            padding: 0;
        }
        .master-popup-track-record li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(45, 55, 72, 0.2);
        }
        .master-popup-track-record li:last-child { border-bottom: none; }
        .master-popup-record-year {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent);
            background: rgba(59, 130, 246, 0.1);
            padding: 1px 6px;
            border-radius: 3px;
            display: inline-block;
            margin-right: 6px;
        }
        .master-popup-record-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            display: inline;
        }
        .master-popup-record-detail {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 3px;
            line-height: 1.5;
        }

        .master-popup-quote {
            font-style: italic;
            font-size: 12px;
            color: var(--yellow);
            padding: 8px 12px;
            background: rgba(234, 179, 8, 0.05);
            border-left: 2px solid rgba(234, 179, 8, 0.3);
            border-radius: 0 4px 4px 0;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <!-- ── Header ──────────────────────────────── -->
    <header class="header">
        <div class="header-left">
            <div class="logo">AI BUBBLE QUANT</div>
            <div class="status-bar">
                <div class="status-item"><span class="status-dot"></span><span>LIVE</span></div>
                <div class="status-item"><span id="data-time">--:--:--</span></div>
                <div class="status-item"><span>v2.0</span></div>
            </div>
        </div>
        <div class="header-right">
            <div class="clock" id="clock">--:--:--</div>
            <button class="refresh-btn" onclick="refreshDataFresh()">刷新</button>
        </div>
    </header>

    <!-- ── Ticker Bar ─────────────────────────── -->
    <div class="ticker-bar">
        <div class="ticker-section">
            <span class="ticker-label">A股</span>
            <div id="ashare-ticker" class="ticker-items"></div>
        </div>
        <div class="ticker-divider"></div>
        <div class="ticker-section">
            <span class="ticker-label">美股</span>
            <div id="us-index-ticker" class="ticker-items"></div>
            <div id="us-etf-ticker" class="ticker-items" style="margin-top: 2px;"></div>
            <div id="us-mag7-ticker" class="ticker-items" style="margin-top: 2px;"></div>
        </div>
    </div>

    <!-- ── Main Layout ─────────────────────────── -->
    <div class="main-container">
        <main class="main-content">
            <!-- Row 1: Bubble (50%) | Right: Signal (25%) + Readiness (25%) -->
            <div class="row-1">
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><circle cx="17.5" cy="17.5" r="3.5"/></svg>
                        <span>泡沫评估</span>
                        <span class="tip-inline"><span class="tip-icon">?</span><span class="tip-text">综合8项经典泡沫指标对比2000年数据。分数越高越接近泡沫状态。</span></span>
                    </div>
                    <div id="bubble-panel" class="loading">加载中</div>
                </div>
                <div class="right-half">
                    <div class="card">
                        <div class="card-title">
                            <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
                            <span>信号监控</span>
                            <span class="tip-inline"><span class="tip-icon">?</span><span class="tip-text">每行是一个做空信号，右列为当前检测值。红色=已触发。问号查看指标描述。</span></span>
                        </div>
                        <div id="signal-panel">加载中</div>
                    </div>
                    <div class="card">
                        <div class="card-title">
                            <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                            <span>交易就绪度</span>
                            <span class="tip-inline"><span class="tip-icon">?</span><span class="tip-text">综合12个做空信号的整体状态，0-100分。分数越高说明入场时机越好。</span></span>
                        </div>
                        <div id="readiness-panel" class="loading">加载中</div>
                    </div>
                </div>
            </div>

            <!-- Row 2: Calculator 3-way -->
            <div class="row-2">
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>
                        <span>中证1000 MO</span>
                    </div>
                    <div id="calc-1000" class="loading">加载中</div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>
                        <span>沪深300 IF</span>
                    </div>
                    <div id="calc-300" class="loading">加载中</div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>
                        <span>上证50 IH</span>
                    </div>
                    <div id="calc-50" class="loading">加载中</div>
                </div>
            </div>

            <!-- Row 2.5: Investment Masters -->
            <div class="row-masters" id="masters-panel">
                <div class="loading">加载中</div>
            </div>

            <!-- Row 3: Backtest Table | Chart -->
            <div class="row-3">
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        <span>历史回测</span>
                    </div>
                    <div id="backtest-panel" class="loading">加载中</div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <svg class="sico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 16l4-6 4 4 5-8"/></svg>
                        <span>回测收益曲线</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="backtestChart"></canvas>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Global tooltip overlay (outside all containers to avoid clipping) -->
    <div id="global-tooltip" class="tooltip-overlay"></div>

    <!-- ── JavaScript ──────────────────────────── -->
    <script>
        let backtestChart = null;

        function formatNum(n) {
            if (n === null || n === undefined) return 'N/A';
            return Number(n).toLocaleString();
        }

        function formatPrice(n) {
            if (n === null || n === undefined) return 'N/A';
            return Number(n).toFixed(2);
        }

        function updateClock() {
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleTimeString('zh-CN', {hour12: false});
        }
        setInterval(updateClock, 1000);
        updateClock();

        async function fetchWithTimeout(url, timeoutMs = 30000) {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const resp = await fetch(url, { signal: controller.signal });
                clearTimeout(timer);
                return await resp.json();
            } catch (e) {
                clearTimeout(timer);
                throw e;
            }
        }

        async function refreshQuotes() {
            try {
                const quotes = await fetchWithTimeout('/api/quotes', 10000);
                // Only merge quotes-specific fields to avoid overwriting calculator/masters/backtest
                if (window.__fullData) {
                    if (quotes.ashare) window.__fullData.ashare = quotes.ashare;
                    if (quotes.us) window.__fullData.us = quotes.us;
                    if (quotes.us_indices) window.__fullData.us_indices = quotes.us_indices;
                    if (quotes.us_etfs) window.__fullData.us_etfs = quotes.us_etfs;
                    if (quotes.us_mag7) window.__fullData.us_mag7 = quotes.us_mag7;
                    if (quotes.us_stocks) window.__fullData.us_stocks = quotes.us_stocks;
                    if (quotes.bubble) window.__fullData.bubble = quotes.bubble;
                    if (quotes.timestamp) window.__fullData.timestamp = quotes.timestamp;
                    renderDashboard(window.__fullData);
                } else {
                    renderDashboard(quotes);
                }
            } catch (e) {
                console.warn('Quotes refresh failed:', e);
            }
        }

        async function refreshData() {
            try {
                const data = await fetchWithTimeout('/api/data', 30000);
                window.__fullData = data;
                renderDashboard(data);
            } catch (e) {
                console.warn('Full data refresh failed:', e);
            }
        }

        async function refreshDataFresh() {
            try {
                const data = await fetchWithTimeout('/api/data?fresh=1', 120000);
                window.__fullData = data;
                renderDashboard(data);
            } catch (e) {
                console.warn('Fresh data refresh failed:', e);
            }
        }

        function renderDashboard(data) {
            try {
                // Remove loading spinners from all panels once data arrives
                document.querySelectorAll('.loading').forEach(function(el) { el.classList.remove('loading'); });

                // Timestamp
                if (data.timestamp) {
                    const t = new Date(data.timestamp * 1000);
                    document.getElementById('data-time').textContent = t.toLocaleTimeString('zh-CN', {hour12: false});
                }

                // ── Ticker: A-share ──
                let ashareHTML = '';
                const ashare = data.ashare || [];
                if (ashare.length > 0) {
                ashare.forEach(item => {
                const change = item.change_pct || 0;
                const bgCls = change >= 0 ? 'up-bg' : 'down-bg';
                const colorCls = change >= 0 ? 'up' : 'down';
                ashareHTML += '<div class="ticker-item">' +
                    '<span class="ticker-item-name">' + item.name + '</span>' +
                    '<span class="ticker-item-price ' + colorCls + '">' + formatPrice(item.price) + '</span>' +
                    '<span class="ticker-item-change ' + bgCls + '">' + (change >= 0 ? '+' : '') + change.toFixed(2) + '%</span></div>';
            });
            } else {
                ashareHTML = '<span style="color:var(--text-muted)">等待行情数据...</span>';
            }
            document.getElementById('ashare-ticker').innerHTML = ashareHTML;

            // ── Ticker: US indices (row 1) ──
            var usIndices = data.us_indices || [];
            var indexHTML = '';
            usIndices.forEach(function(item) {
                var change = item.change_pct || 0;
                var bgCls = change >= 0 ? 'up-bg' : 'down-bg';
                var colorCls = change >= 0 ? 'up' : 'down';
                indexHTML += '<div class="ticker-item">' +
                    '<span class="ticker-item-name">' + item.name + '</span>' +
                    '<span class="ticker-item-price ' + colorCls + '">' + formatPrice(item.price) + '</span>' +
                    '<span class="ticker-item-change ' + bgCls + '">' + (change >= 0 ? '+' : '') + change.toFixed(2) + '%</span></div>';
            });
            if (!indexHTML) indexHTML = '<span style="color:var(--text-muted)">等待指数数据...</span>';
            document.getElementById('us-index-ticker').innerHTML = indexHTML;

            // ── Ticker: US ETFs + VIX (row 2) ──
            var usEtfs = data.us_etfs || [];
            var etfHTML = '';
            usEtfs.forEach(function(item) {
                var change = item.change_pct || 0;
                var bgCls = change >= 0 ? 'up-bg' : 'down-bg';
                var colorCls = change >= 0 ? 'up' : 'down';
                var isVix = item.name === 'VIX';
                etfHTML += '<div class="ticker-item">' +
                    '<span class="ticker-item-name">' + item.name + '</span>' +
                    '<span class="ticker-item-price ' + colorCls + '">' + (isVix ? '' : '$') + formatPrice(item.price) + '</span>' +
                    '<span class="ticker-item-change ' + bgCls + '">' + (change >= 0 ? '+' : '') + change.toFixed(2) + '%</span></div>';
            });
            if (!etfHTML) etfHTML = '<span style="color:var(--text-muted)">等待行情数据...</span>';
            document.getElementById('us-etf-ticker').innerHTML = etfHTML;

            // ── Ticker: Magnificent 7 (row 3) ──
            var usMag7 = data.us_mag7 || [];
            var mag7HTML = '';
            usMag7.forEach(function(item) {
                var change = item.change_pct || 0;
                var bgCls = change >= 0 ? 'up-bg' : 'down-bg';
                var colorCls = change >= 0 ? 'up' : 'down';
                mag7HTML += '<div class="ticker-item">' +
                    '<span class="ticker-item-name">' + item.name + '</span>' +
                    '<span class="ticker-item-price ' + colorCls + '">' + formatPrice(item.price) + '</span>' +
                    '<span class="ticker-item-change ' + bgCls + '">' + (change >= 0 ? '+' : '') + change.toFixed(2) + '%</span></div>';
            });
            if (!mag7HTML) mag7HTML = '<span style="color:var(--text-muted)">等待行情数据...</span>';
            document.getElementById('us-mag7-ticker').innerHTML = mag7HTML;

            // ── Bubble Score ──
            const bubble = data.bubble || {};
            const score = bubble.overall || 0;
            const verdict = bubble.verdict || '等待数据...';
            const indicators = bubble.indicators || [];
            const hasData = indicators.length > 0;

            if (!hasData) {
                document.getElementById('bubble-panel').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:12px;">等待评估数据...</div>';
            } else {
            const scoreColor = score >= 70 ? 'var(--red)' : score >= 45 ? 'var(--yellow)' : 'var(--green)';

            let bubbleHTML = '<div class="bubble-content">' +
                '<div class="bubble-score">' +
                '<div class="score-circle" style="border-color: ' + scoreColor + '">' +
                '<div class="score-value" style="color: ' + scoreColor + '">' + score.toFixed(0) + '</div>' +
                '<div class="score-label">/ 100</div></div>' +
                '<div class="bubble-indicators">';

            (indicators).forEach(ind => {
                const fill = ind.score || 0;
                const hasData = ind.current !== null && ind.current !== undefined;
                const currentText = hasData ?
                    (typeof ind.current === 'number' ? ind.current.toFixed(2) : ind.current) + ' ' + (ind.unit || '') :
                    '<span style="color:var(--text-muted)">未获取</span>';
                const fillColor = hasData ? (fill >= 50 ? 'var(--red)' : fill >= 25 ? 'var(--yellow)' : 'var(--green)') : 'var(--text-muted)';
                const dotcomPos = ind.dotcom_2000 !== null && ind.dotcom_2000 !== undefined ?
                    Math.min(ind.dotcom_2000 / (ind.extreme_threshold || 1) * 100, 100) : 0;

                let dotcomMarker = '';
                if (ind.dotcom_2000 !== null && ind.dotcom_2000 !== undefined && ind.extreme_threshold) {
                    const pct = Math.min(ind.dotcom_2000 / ind.extreme_threshold * 100, 100);
                    dotcomMarker = '<div style="position:absolute;top:0;height:100%;width:1px;background:var(--yellow);opacity:0.5;left:' + pct + '%"></div>';
                }

                bubbleHTML += '<div class="indicator-bar">' +
                    '<div class="indicator-label">' +
                    '<span>' + ind.name + '</span>' +
                    '<span>' + currentText + '</span></div>' +
                    '<div class="indicator-track" style="position:relative">' + dotcomMarker +
                    '<div class="indicator-fill" style="width: ' + Math.min(fill, 100) + '%; background: ' + fillColor + '"></div></div></div>';
            });

            bubbleHTML += '</div></div>' +
                '<div style="font-size: 11px; color: var(--text-muted); text-align: center; line-height: 1.4;">' + verdict + '</div>' +
                '</div>';
            document.getElementById('bubble-panel').innerHTML = bubbleHTML;
            } // end if hasData for bubble

            // ── Readiness ──
            const signals = data.signals || {};
            const readiness = signals.score || 0;
            const alertLevel = signals.alert_level || 'INFO';
            const triggered = signals.triggered_count || 0;

            const meterColor = readiness >= 70 ? 'var(--red)' : readiness >= 40 ? 'var(--yellow)' : 'var(--green)';
            const alertClass = alertLevel.includes('TRIGGER') ? 'alert-high' :
                              alertLevel.includes('ALERT') ? 'alert-moderate' :
                              alertLevel.includes('WATCH') ? 'alert-low' : 'alert-info';
            const alertExplain = alertLevel.includes('TRIGGER') ? '多个信号已触发，考虑入场' :
                                alertLevel.includes('ALERT') ? '部分信号接近触发，密切关注' :
                                alertLevel.includes('WATCH') ? '少数信号值得关注' : '暂无明显做空信号';

            // Build triggered signals detail list
            let triggeredListHTML = '';
            const triggeredList = signals.triggered_list || [];
            if (triggeredList.length > 0) {
                const dirSymbol = {"above": "↑突破", "below": "↓跌破"};
                triggeredListHTML = '<div style="margin-top: 8px; width: 100%; max-height: 90px; overflow-y: auto; border-top: 1px solid var(--border); padding-top: 6px;">' +
                    '<div style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">已触发指标</div>';
                triggeredList.forEach(ts => {
                    const ds = dirSymbol[ts.direction] || '';
                    const cv = ts.current_value !== null && ts.current_value !== undefined ?
                        (typeof ts.current_value === 'number' ? ts.current_value.toFixed(2) : ts.current_value) : '--';
                    const tv = typeof ts.threshold === 'number' ? ts.threshold.toFixed(2) : ts.threshold;
                    triggeredListHTML += '<div style="font-size: 10px; padding: 2px 0; display: flex; justify-content: space-between; align-items: center;">' +
                        '<span style="color: var(--text-secondary); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="' + (ts.name || '') + '">' + (ts.name || '').substring(0, 18) + '</span>' +
                        '<span class="mono up" style="font-size: 9px;">' + cv + ' ' + ds + ' ' + tv + '</span>' +
                        '</div>';
                });
                triggeredListHTML += '</div>';
            }

            let readinessHTML = '<div class="readiness-content">' +
                '<div class="readiness-meter">' +
                '<div class="meter-value" style="color: ' + meterColor + '">' + readiness.toFixed(0) + '</div>' +
                '<div class="meter-label">就绪分数 / 100</div></div>' +
                '<div class="alert-box ' + alertClass + '" title="' + alertExplain + '">' + alertLevel + '</div>' +
                '<div style="font-size: 12px; color: var(--text-muted); text-align: center;">' +
                triggered + ' / ' + (signals.total_count || 12) + ' 信号触发</div>' +
                triggeredListHTML + '</div>';
            document.getElementById('readiness-panel').innerHTML = readinessHTML;

            // ── Signal Table (simplified: signal name + current value) ──
            const signalList = signals.list || [];

            if (signalList.length > 0) {
            let sigTableHTML = '<div class="sig-table-wrap"><table class="sig-table"><thead><tr><th style="width:70%">信号</th><th style="width:30%">当前值</th></tr></thead><tbody>';

            signalList.forEach(s => {
                const valDisplay = s.current_value !== null && s.current_value !== undefined ?
                    (typeof s.current_value === 'number' ? s.current_value.toFixed(2) : s.current_value) : '--';
                const colorClass = s.triggered ? 'up' : '';
                const safeDesc = (s.description || '').replace(/"/g, '&quot;');

                sigTableHTML += '<tr><td><span class="tip-inline">' +
                    '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block">' + (s.name || '') + '</span>' +
                    '<span class="tip-icon">?</span>' +
                    '<span class="tip-text">' + (s.description || '') + '</span></span></td>' +
                    '<td class="' + colorClass + '">' + valDisplay + '</td></tr>';
            });

            sigTableHTML += '</tbody></table></div>';
            document.getElementById('signal-panel').innerHTML = sigTableHTML;
            } else {
                document.getElementById('signal-panel').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:12px;">等待信号数据...</div>';
            }

            // ── Calculator (3-way: MO 1000, IF 300, IH 50) ──
            try { renderCalcPanel('calc-1000', data.calculator ? data.calculator['1000'] : {}); } catch(e) { console.error('calc-1000 render failed:', e); }
            try { renderCalcPanel('calc-300', data.calculator ? data.calculator['300'] : {}); } catch(e) { console.error('calc-300 render failed:', e); }
            try { renderCalcPanel('calc-50', data.calculator ? data.calculator['50'] : {}); } catch(e) { console.error('calc-50 render failed:', e); }

            // ── Investment Masters ──
            try { renderMastersPanel(data.masters || {}); } catch(e) { console.error('masters panel render failed:', e); }

        // ── Backtest Table ──
            try {
                const bt = data.backtest || {};
                let btHTML = '<div class="bt-table-wrap"><table class="bt-table"><thead><tr>' +
                    '<th>事件</th><th>买入</th><th>行权价</th><th>成本</th><th>暴跌后</th><th>盈亏%</th><th>盈亏</th>' +
                    '</tr></thead><tbody>';
                (bt.results || []).forEach(r => {
                    const pnlCls = r.pnl_pct >= 0 ? 'up' : 'down';
                    const safeDesc = (r.description || r.name || '').replace(/"/g, '&quot;');
                    btHTML += '<tr>' +
                        '<td><span class="tip-inline">' +
                        '<span style="max-width:60px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block">' + r.name.substring(0, 10) + '</span>' +
                        '<span class="tip-icon">?</span>' +
                        '<span class="tip-text">' + safeDesc + '</span></span></td>' +
                        '<td>' + formatNum(r.index_before) + '</td>' +
                        '<td>' + formatNum(r.strike) + '</td>' +
                        '<td>' + r.premium.toFixed(1) + '</td>' +
                        '<td>' + formatNum(r.index_after) + '</td>' +
                        '<td class="' + pnlCls + '">'+(r.pnl_pct >= 0 ? '+' : '') + r.pnl_pct.toFixed(0) + '%</td>' +
                        '<td class="' + pnlCls + '">'+(r.pnl_money >= 0 ? '+' : '') + formatNum(r.pnl_money) + '</td></tr>';
                });
                btHTML += '</tbody></table>' +
                    '<div style="margin-top: 6px; font-size: 10px; text-align: center; color: var(--text-muted);">' +
                    '总计: ' + bt.total + '次 | 盈利' + bt.wins + '次 | 总盈亏: <span class="' + (bt.total_pnl >= 0 ? 'up' : 'down') + '">'+(bt.total_pnl >= 0 ? '+' : '') + formatNum(bt.total_pnl) + '元</span></div></div>';
                document.getElementById('backtest-panel').innerHTML = btHTML;
            } catch(e) { console.error('backtest panel render failed:', e); }

            // ── Backtest Chart ──
            try {
                renderBacktestChart((data.backtest && data.backtest.results) || []);
            } catch(e) { console.error('backtest chart render failed:', e); }
        } catch(e) { console.error('renderDashboard error:', e); }
        }

        // ── Investment Masters Panel ──
        function renderMastersPanel(mastersData) {
            const panel = document.getElementById('masters-panel');
            if (!panel) return;
            panel.classList.remove('loading');
            if (!mastersData || mastersData.error) {
                panel.innerHTML = '<div class="loading">投资大师数据加载失败</div>';
                return;
            }

            const masters = mastersData.masters || [];
            if (masters.length === 0) {
                panel.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:60px;color:var(--text-muted);font-size:12px;">等待评估数据...</div>';
                return;
            }
            const pm = mastersData.portfolio_decision || {};
            const signalClass = pm.overall_signal || 'neutral';
            const signalZh = signalClass === 'bullish' ? '综合看涨' : signalClass === 'bearish' ? '综合看空' : '综合中性';
            const score = pm.consensus_score || 0;
            const barColor = score > 0 ? 'var(--green)' : score < 0 ? 'var(--red)' : 'var(--yellow)';
            const barWidth = Math.min(100, Math.abs(score));

            // Header
            let html = '<div class="masters-header">' +
                '<span class="masters-header-title">🎯 投资大师观点</span>' +
                '<div class="masters-header-signal">' +
                    '<span class="masters-consensus-badge ' + signalClass + '">' + signalZh + '</span>' +
                    '<span class="masters-consensus-score">共识: ' + score.toFixed(1) + '</span>' +
                    '<div class="masters-consensus-bar"><div class="masters-consensus-bar-fill" style="width:' + barWidth + '%;background:' + barColor + '"></div></div>' +
                '</div>' +
            '</div>';

            // Masters grid
            html += '<div class="masters-grid">';
            masters.forEach(m => {
                const sig = m.signal || 'neutral';
                const sigZh = sig === 'bullish' ? '看涨' : sig === 'bearish' ? '看空' : '中性';
                const conf = m.confidence || 0;
                const initials = m.avatar_initials || '??';
                const live = m.is_alive;
                const reasoning = (m.reasoning || '').substring(0, 80);

                html += '<div class="master-card ' + sig + '" data-master-id="' + (m.master_id || '') + '">' +
                    '<div class="master-card-header">' +
                        '<div class="master-avatar">' + initials + '</div>' +
                        '<div class="master-name-group">' +
                            '<div class="master-name-en" title="' + (m.philosophy_zh || '') + '">' + m.master_name_en + '</div>' +
                            '<div class="master-name-zh">' + m.master_name_zh + '</div>' +
                        '</div>' +
                        (live ? '<span class="master-live-badge">LIVE</span>' : '') +
                    '</div>' +
                    '<div class="master-signal-row">' +
                        '<div class="master-signal ' + sig + '"><span>' + sigZh + '</span></div>' +
                    '</div>' +
                    '<div class="master-confidence">' +
                        '<div class="master-confidence-bar"><div class="master-confidence-fill ' + sig + '" style="width:' + conf + '%"></div></div>' +
                        '<span class="master-confidence-value">' + conf + '%</span>' +
                    '</div>' +
                    (reasoning ? '<div class="master-reasoning" title="' + (m.reasoning || '').replace(/"/g, '&quot;') + '">' + reasoning + '</div>' : '') +
                '</div>';
            });
            html += '</div>';

            // Portfolio Manager Summary
            if (pm && pm.overall_signal) {
                const total = (pm.bullish_count || 0) + (pm.bearish_count || 0) + (pm.neutral_count || 0);
                const bPct = total > 0 ? (pm.bullish_count / total * 100) : 0;
                const nPct = total > 0 ? (pm.neutral_count / total * 100) : 0;
                const bePct = total > 0 ? (pm.bearish_count / total * 100) : 0;
                const pmSigZh = pm.overall_signal === 'bullish' ? '看涨' : pm.overall_signal === 'bearish' ? '看空' : '中性';
                const pmConf = pm.overall_confidence || 0;

                html += '<div class="pm-summary">';
                html += '<div class="pm-top">' +
                    '<span class="pm-title">组合经理综合评估</span>' +
                    '<span class="pm-signal-badge ' + pm.overall_signal + '">' + pmSigZh + ' ' + pmConf + '%</span>' +
                    '<span class="pm-action">建议: <strong>' + (pm.recommended_action || '观望') + '</strong> | ' + (pm.suggested_position || '') + '</span>' +
                '</div>';

                // Vote bar
                html += '<div class="pm-vote-bar">';
                if (bPct > 0) html += '<div class="pm-vote-segment bullish" style="width:' + bPct + '%">' + pm.bullish_count + ' 看涨</div>';
                if (nPct > 0) html += '<div class="pm-vote-segment neutral" style="width:' + nPct + '%">' + pm.neutral_count + ' 中性</div>';
                if (bePct > 0) html += '<div class="pm-vote-segment bearish" style="width:' + bePct + '%">' + pm.bearish_count + ' 看空</div>';
                html += '</div>';

                // Reasons
                if ((pm.top_bear_reasons && pm.top_bear_reasons.length) || (pm.top_bull_reasons && pm.top_bull_reasons.length)) {
                    html += '<div class="pm-reasons">';
                    if (pm.top_bull_reasons && pm.top_bull_reasons.length > 0) {
                        html += '<div class="pm-reasons-bull"><h4>看涨理由</h4><ul>';
                        pm.top_bull_reasons.forEach(r => { html += '<li>' + r.substring(0, 60) + '</li>'; });
                        html += '</ul></div>';
                    }
                    if (pm.top_bear_reasons && pm.top_bear_reasons.length > 0) {
                        html += '<div class="pm-reasons-bear"><h4>看空理由</h4><ul>';
                        pm.top_bear_reasons.forEach(r => { html += '<li>' + r.substring(0, 60) + '</li>'; });
                        html += '</ul></div>';
                    }
                    html += '</div>';
                }

                html += '</div>';
            }

            panel.innerHTML = html;
        }

        // ── Master Detail Popup ──
        var __masterProfiles = {};
        var __popupVisible = false;
        var __popupTimer = null;

        function showMasterPopup(masterId, cardEl) {
            if (!masterId) return;
            var popup = document.getElementById('master-popup');
            var overlay = document.getElementById('master-popup-overlay');
            if (!popup || !overlay) return;

            var profile = __masterProfiles[masterId];
            if (!profile) {
                // Profile not loaded yet, fetch it
                fetchMasterProfiles(function() {
                    showMasterPopup(masterId, cardEl);
                });
                return;
            }

            var sigZh = profile.current_signal === 'bullish' ? '看涨' : profile.current_signal === 'bearish' ? '看空' : '中性';
            var sigClass = profile.current_signal || 'neutral';

            var contentHTML = '';

            // Header
            contentHTML += '<div class="master-popup-header">' +
                '<div class="master-popup-avatar">' + (profile.avatar_initials || '?') + '</div>' +
                '<div>' +
                    '<div class="master-popup-name">' + profile.name_zh + '</div>' +
                    '<div class="master-popup-name-sub">' + profile.name_en + (profile.is_alive ? ' · 在世' : ' · 历史') + '</div>' +
                '</div>' +
                '<div class="master-popup-signal-badge ' + sigClass + '">' + sigZh + ' ' + profile.current_confidence + '%</div>' +
            '</div>';

            // Philosophy + Tags
            contentHTML += '<div class="master-popup-section">' +
                '<div class="master-popup-section-title">投资哲学</div>' +
                '<div class="master-popup-philosophy">' + (profile.philosophy_zh || '') + '</div>';
            if (profile.style_tags && profile.style_tags.length) {
                contentHTML += '<div class="master-popup-tags">';
                profile.style_tags.forEach(function(tag) {
                    contentHTML += '<span class="master-popup-tag">' + tag + '</span>';
                });
                contentHTML += '</div>';
            }
            if (profile.famous_quote_zh) {
                contentHTML += '<div class="master-popup-quote">"' + profile.famous_quote_zh + '"</div>';
            }
            contentHTML += '</div>';

            // Detailed Analysis
            if (profile.detailed_analysis) {
                contentHTML += '<div class="master-popup-section">' +
                    '<div class="master-popup-section-title">当前市场观点</div>' +
                    '<div class="master-popup-analysis">' + profile.detailed_analysis + '</div>' +
                '</div>';
            }

            // Track Record
            if (profile.track_record && profile.track_record.length) {
                contentHTML += '<div class="master-popup-section">' +
                    '<div class="master-popup-section-title">经典战绩</div>' +
                    '<ul class="master-popup-track-record">';
                profile.track_record.forEach(function(tr) {
                    contentHTML += '<li>' +
                        '<span class="master-popup-record-year">' + (tr.year || '') + '</span>' +
                        '<span class="master-popup-record-title">' + (tr.title || '') + '</span>' +
                        '<div class="master-popup-record-detail">' + (tr.detail || '') + '</div>' +
                    '</li>';
                });
                contentHTML += '</ul></div>';
            }

            document.getElementById('master-popup-content').innerHTML = contentHTML;

            // Position popup near the card
            var rect = cardEl.getBoundingClientRect();
            var popupW = 480;
            var popupH = Math.min(80 * window.innerHeight / 100, popup.offsetHeight || 600);
            var left = rect.right + 12;
            var top = rect.top;

            // If popup would overflow right edge, show on the left
            if (left + popupW > window.innerWidth - 16) {
                left = rect.left - popupW - 12;
            }
            // If still off-screen, center horizontally
            if (left < 16) {
                left = Math.max(16, (window.innerWidth - popupW) / 2);
            }
            // If popup would overflow bottom, shift up
            if (top + popupH > window.innerHeight - 16) {
                top = window.innerHeight - popupH - 16;
            }
            if (top < 16) top = 16;

            popup.style.left = left + 'px';
            popup.style.top = top + 'px';

            overlay.classList.add('active');
            // Trigger animation
            requestAnimationFrame(function() {
                popup.classList.add('visible');
            });
            __popupVisible = true;
        }

        function hideMasterPopup() {
            if (!__popupVisible) return;
            var popup = document.getElementById('master-popup');
            var overlay = document.getElementById('master-popup-overlay');
            if (popup) popup.classList.remove('visible');
            if (overlay) overlay.classList.remove('active');
            __popupVisible = false;
        }

        function fetchMasterProfiles(callback) {
            fetchWithTimeout('/api/masters-profiles', 15000)
                .then(function(profiles) {
                    __masterProfiles = profiles;
                    if (callback) callback();
                })
                .catch(function(e) {
                    console.warn('Failed to fetch master profiles:', e);
                    if (callback) callback();
                });
        }

        function renderCalcPanel(containerId, calcData) {
            var el = document.getElementById(containerId);
            if (el) el.classList.remove('loading');
            const calc = calcData || {};
            const idx = calc.index || 0;
            let html = '<div class="calc-content" style="width:100%;"><div style="font-size: 9px; color: var(--text-muted); margin-bottom: 6px;">' +
                '指数: <span class="mono" style="color:var(--text-primary)">' + formatPrice(idx) + '</span> | IV: 20% | 到期: 7天</div>';
            html += '<div class="calc-table-wrap"><table class="calc-table"><thead><tr>' +
                '<th>行权价</th><th>虚值%</th><th>成本</th><th>胜率</th><th>跌3%</th><th>跌5%</th><th>R/R</th>' +
                '</tr></thead><tbody>';
            if (!calc.list || calc.list.length === 0) {
                html += '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">无数据</td></tr>';
            } else {
                (calc.list || []).forEach(r => {
                    const drops = {};
                    if (r.scenarios && r.scenarios.length) {
                        r.scenarios.forEach(s => { drops[s.drop_pct] = s.pnl; });
                    }
                    const rr = r.risk_reward || 0;
                    const rrCls = rr >= 3 ? 'up' : rr >= 1 ? '' : 'down';
                    html += '<tr>' +
                        '<td>' + formatPrice(r.strike) + '</td>' +
                        '<td>' + r.otm_pct + '%</td>' +
                        '<td>' + formatNum(r.total_cost) + '</td>' +
                        '<td>' + r.prob_itm + '%</td>' +
                        '<td class="' + ((drops[0.03]||0) >= 0 ? 'up' : 'down') + '">' + ((drops[0.03]||0) >= 0 ? '+' : '') + formatNum(drops[0.03] || 0) + '</td>' +
                        '<td class="' + ((drops[0.05]||0) >= 0 ? 'up' : 'down') + '">' + ((drops[0.05]||0) >= 0 ? '+' : '') + formatNum(drops[0.05] || 0) + '</td>' +
                        '<td class="' + rrCls + '">' + rr.toFixed(1) + 'x</td></tr>';
                });
            }
            html += '</tbody></table></div></div>';
            document.getElementById(containerId).innerHTML = html;
        }

        function renderBacktestChart(results) {
            if (!results || results.length === 0) return;
            const canvas = document.getElementById('backtestChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (backtestChart) backtestChart.destroy();

            const labels = results.map(r => r.name.length > 8 ? r.name.substring(0, 8) : r.name);
            const pnlData = results.map(r => r.pnl_money);

            backtestChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '盈亏 (元)',
                        data: pnlData,
                        backgroundColor: pnlData.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(16, 185, 129, 0.6)'),
                        borderColor: pnlData.map(v => v >= 0 ? '#ef4444' : '#10b981'),
                        borderWidth: 1,
                        borderRadius: 4,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1a2332',
                            borderColor: '#2d3748',
                            borderWidth: 1,
                            titleFont: { family: 'monospace' },
                            bodyFont: { family: 'monospace' },
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#64748b', font: { size: 9, family: 'monospace' } },
                            grid: { color: 'rgba(45, 55, 72, 0.3)' }
                        },
                        y: {
                            ticks: {
                                color: '#64748b',
                                font: { size: 9, family: 'monospace' },
                                callback: function(v) { return '¥' + v.toLocaleString(); }
                            },
                            grid: { color: 'rgba(45, 55, 72, 0.3)' }
                        }
                    }
                }
            });
        }

        // Initial load: parallel quotes + full data
        refreshQuotes();
        refreshData();

        // Auto refresh: quotes every 30s, full data every 1h
        setInterval(refreshQuotes, 30000);
        setInterval(refreshData, 3600000);

        // Master popup hover — pre-fetch profiles on first hover
        document.addEventListener('mouseover', function(e) {
            var card = e.target.closest('.master-card');
            if (card) {
                clearTimeout(__popupTimer);
                var masterId = card.getAttribute('data-master-id');
                if (!masterId) return;
                __popupTimer = setTimeout(function() {
                    showMasterPopup(masterId, card);
                }, 300);
                return;
            }
            // If hovering over popup itself, keep it visible
            if (e.target.closest && e.target.closest('#master-popup')) {
                clearTimeout(__popupTimer);
                return;
            }
            // Otherwise, schedule hide
            if (__popupVisible) {
                clearTimeout(__popupTimer);
                __popupTimer = setTimeout(hideMasterPopup, 200);
            }
        });

        // Pre-fetch profiles once full data is loaded
        var __profilesFetched = false;
        setInterval(function() {
            if (!__profilesFetched && window.__fullData && window.__fullData.masters) {
                __profilesFetched = true;
                fetchMasterProfiles();
            }
        }, 2000);

        // Global tooltip overlay — used for tooltips inside scrollable containers
        // to avoid clipping by parent overflow
        var globalTooltipEl = document.getElementById('global-tooltip');
        var globalContentEl = null;
        var activeTipEl = null;

        function isInsideScrollable(el) {
            while (el && el !== document.body) {
                if (el.classList && (el.classList.contains('sig-table-wrap') || el.classList.contains('calc-table-wrap'))) {
                    return true;
                }
                el = el.parentElement;
            }
            return false;
        }

        function showGlobalTooltip(tipEl, html) {
            hideGlobalTooltip();
            if (!globalContentEl) {
                globalContentEl = document.createElement('div');
                globalContentEl.className = 'tooltip-content';
                globalTooltipEl.appendChild(globalContentEl);
            }
            globalContentEl.innerHTML = html;
            globalTooltipEl.classList.add('active');

            var rect = tipEl.getBoundingClientRect();
            var w = globalContentEl.offsetWidth || 200;
            var h = globalContentEl.offsetHeight || 40;
            var left = rect.left + rect.width / 2;
            var top = rect.top - h - 8;
            if (top < 10) top = rect.bottom + 8;
            // Keep within viewport bounds
            var tw = w / 2 + 8;
            if (left - tw < 0) left = tw;
            if (left + tw > window.innerWidth) left = window.innerWidth - tw;
            globalContentEl.style.left = left + 'px';
            globalContentEl.style.top = top + 'px';
            globalContentEl.style.transform = 'translateX(-50%)';
            activeTipEl = tipEl;
        }

        function hideGlobalTooltip() {
            if (globalTooltipEl) globalTooltipEl.classList.remove('active');
            activeTipEl = null;
        }

        // Tooltip positioning — event delegation for dynamic content
        document.addEventListener('DOMContentLoaded', function() {
            // Static tips that exist at load time
            document.querySelectorAll('.tip-inline').forEach(function(tip) {
                attachTipHandlers(tip);
            });
        });

        // Event delegation: handle dynamically created tips (backtest table, signal table, etc.)
        document.addEventListener('mouseover', function(e) {
            var tip = e.target.closest('.tip-inline');
            if (!tip || tip.__tipAttached) return;
            attachTipHandlers(tip);
            tip.__tipAttached = true;
            // Trigger the mouseenter immediately
            tip.dispatchEvent(new Event('mouseenter', { bubbles: false }));
        });
        document.addEventListener('mouseout', function(e) {
            var tip = e.target.closest('.tip-inline');
            if (!tip) return;
            // Check if we actually left the tip element
            if (!tip.contains(e.relatedTarget)) {
                hideGlobalTooltip();
                // Also hide CSS-based tips
                tip.querySelectorAll('.tip-text').forEach(function(t) { t.style.display = 'none'; });
            }
        });

        function attachTipHandlers(tip) {
            var text = tip.querySelector('.tip-text');
            if (!text) return;
            var useGlobal = isInsideScrollable(tip);
            if (useGlobal) {
                tip.addEventListener('mouseenter', function() {
                    showGlobalTooltip(tip, text.innerHTML);
                });
                tip.addEventListener('mouseleave', function() {
                    hideGlobalTooltip();
                });
            } else {
                tip.addEventListener('mouseenter', function() {
                    text.style.display = 'block';
                    var rect = tip.getBoundingClientRect();
                    var h = text.offsetHeight || 40;
                    var left = rect.left + rect.width / 2;
                    var top = rect.top - h - 8;
                    if (top < 10) top = rect.bottom + 8;
                    if (left < 120) left = 120;
                    if (left > window.innerWidth - 120) left = window.innerWidth - 120;
                    text.style.left = left + 'px';
                    text.style.top = top + 'px';
                    text.style.transform = 'translateX(-50%)';
                });
                tip.addEventListener('mouseleave', function() {
                    text.style.display = 'none';
                });
            }
        }
    </script>

    <!-- Master Detail Popup Container -->
    <div id="master-popup-overlay" class="master-popup-overlay">
        <div id="master-popup" class="master-popup">
            <div id="master-popup-content"></div>
        </div>
    </div>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def api_data():
    fresh = request.args.get('fresh', '0') == '1'
    data = get_all_data(force_fresh=fresh)
    return jsonify(_to_native(data))


@app.route('/api/refresh')
def api_refresh():
    data = get_all_data(force_fresh=True)
    return jsonify(_to_native(data))


@app.route('/api/quotes')
def api_quotes():
    """Fast quotes endpoint — 30s TTL, no historical data requests."""
    data = get_quotes_data()
    return jsonify(_to_native(data))


@app.route('/api/masters-profiles')
def api_masters_profiles():
    """Master profiles with detailed analysis — long TTL cache."""
    global _profiles_cache
    now = time.time()
    if _profiles_cache and _profiles_cache.get("_timestamp") and (now - _profiles_cache["_timestamp"]) < _profiles_ttl:
        return jsonify(_to_native(_profiles_cache))

    # Need to generate profiles — requires current masters data
    full_data = get_all_data()
    masters_data = full_data.get("masters", {})

    # Build market data dict for profile generation
    bubble = full_data.get("bubble", {})
    indicators = bubble.get("indicators", [])
    cape = 30
    for ind in indicators:
        if "CAPE" in str(ind.get("name", "")) or "席勒" in str(ind.get("name", "")):
            cape = ind.get("current") or 30
            break

    market_data = {
        "bubble_score": bubble.get("overall", 50),
        "cape": cape,
        "vix": 20,
        "spy_pe": next((u.get("price", 540) for u in full_data.get("us", []) if u.get("name") == "SPY"), 540) / 22,
        "qqq_rsi": 50,
    }
    # Get QQQ RSI from signals if available
    signals = full_data.get("signals", {})
    for s in signals.get("list", []):
        if "RSI" in s.get("name", ""):
            market_data["qqq_rsi"] = s.get("current_value", 50)
            break

    use_llm = os.environ.get("USE_LLM_FOR_MASTERS", "").lower() == "true"
    api_key = os.environ.get("OPENAI_API_KEY")

    profiles = build_master_profiles(masters_data, market_data, use_llm=use_llm, api_key=api_key)
    profiles["_timestamp"] = now
    _profiles_cache = profiles

    return jsonify(_to_native(profiles))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Bubble Quant Web Terminal")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 AI BUBBLE QUANT — Professional Trading Terminal")
    logger.info("=" * 60)
    logger.info(f"   http://{args.host}:{args.port}")
    logger.info(f"   Press Ctrl+C to stop")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
