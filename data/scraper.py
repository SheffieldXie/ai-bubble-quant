"""
Web scraper module for fetching US stock data from China-accessible sources.
Primary sources: Tencent Finance (腾讯行情), Eastmoney (东方财富)
Fallback: Yahoo Finance via proxy (if available)
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── HTTP Session with retry ────────────────────────────────────────────

def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    })
    return session


# ── Tencent Finance (腾讯行情) ─────────────────────────────────────────

class TencentScraper:
    """
    Fetches US stock data from Tencent Finance API.
    This is the most reliable source for US stock data from China.
    URL: http://qt.gtimg.cn/q=usAAPL
    """

    BASE_URL = "http://qt.gtimg.cn/q="

    def __init__(self):
        self.session = _get_session()
        self._delay = 0.2

    def get_quote(self, ticker: str) -> Optional[dict]:
        """
        Get real-time quote from Tencent Finance.
        Response: v_usSPY="200~name~TICKER~price~prev_close~open~vol~...~date~change~change_pct~high~low~..."
        """
        t = ticker.upper()
        url = f"{self.BASE_URL}us{t}"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=10)
            resp.encoding = "gbk"
            text = resp.text.strip()

            if not text or '""' in text or "v_us" not in text:
                return None

            match = re.search(r'="(.+?)"', text)
            if not match:
                return None

            parts = match.group(1).split("~")
            if len(parts) < 6:
                return None

            def sf(val, default=None):
                """Safe float conversion."""
                try:
                    if not val or val in ("-", "None", "none", ""):
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # Tencent US stock format (VERIFIED from raw data):
            # [3]=price, [4]=prev_close, [5]=open, [6]=volume, [9]=bid
            # [19]=ask, [30]=datetime, [31]=change, [32]=change_pct,
            # [33]=high, [34]=low, [35]=currency, [36]=volume, [37]=turnover
            # [48]=52w_high, [49]=52w_low

            result = {
                "ticker": ticker,
                "name": parts[1] if len(parts) > 1 else "",
                "price": sf(parts[3]),
                "prev_close": sf(parts[4]),
                "open": sf(parts[5]),
                "volume": sf(parts[6]),
                "bid_price": sf(parts[9]),
                "ask_price": sf(parts[19]),
                "date_time": parts[30] if len(parts) > 30 else None,
                "change": sf(parts[31]),
                "change_pct": sf(parts[32]),
                "high": sf(parts[33]),
                "low": sf(parts[34]),
                "currency": parts[35] if len(parts) > 35 else "USD",
                "volume_shares": sf(parts[36]),
                "turnover": sf(parts[37]),
                "market_cap": sf(parts[37]),
                "week_52_high": sf(parts[48]),
                "week_52_low": sf(parts[49]),
                "fetched_at": datetime.now().isoformat(),
                "source": "tencent",
                "raw_fields": len(parts),
            }

            if result["price"] is None:
                return None

            return result

        except Exception as e:
            logger.error(f"Tencent quote fetch failed for {ticker}: {e}")
            return None

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch quotes for multiple tickers in one request."""
        symbols = ",".join([f"us{t.upper()}" for t in tickers])
        url = f"{self.BASE_URL}{symbols}"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=15)
            resp.encoding = "gbk"
            text = resp.text.strip()

            results = {}
            for line in text.split(";"):
                line = line.strip()
                if not line or "=" not in line:
                    continue

                match = re.search(r'v_us(\w+)="(.+?)"', line)
                if not match:
                    continue

                ticker = match.group(1)
                parts = match.group(2).split("~")
                if len(parts) < 10:
                    continue

                results[ticker] = {
                    "ticker": ticker,
                    "name": parts[1] if len(parts) > 1 else "",
                    "price": float(parts[3]) if parts[3] else None,
                    "prev_close": float(parts[4]) if parts[4] else None,
                    "open": float(parts[5]) if parts[5] else None,
                    "volume": float(parts[6]) if parts[6] else None,
                    "high": float(parts[30]) if len(parts) > 30 and parts[30] else None,
                    "low": float(parts[31]) if len(parts) > 31 and parts[31] else None,
                    "market_cap": float(parts[32]) * 1e8 if len(parts) > 32 and parts[32] else None,
                    "pe_ratio": float(parts[38]) if len(parts) > 38 and parts[38] else None,
                    "fetched_at": datetime.now().isoformat(),
                    "source": "tencent",
                }

            return results
        except Exception as e:
            logger.error(f"Tencent batch fetch failed: {e}")
            return {}


# ── Eastmoney (东方财富) ────────────────────────────────────────────────

class EastmoneyScraper:
    """
    Fetches US stock data from Eastmoney API.
    Works from China without proxy.
    """

    BASE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    # Map US ticker to Eastmoney security code
    _market_map = {
        "AAPL": "105.AAPL", "MSFT": "105.MSFT", "GOOGL": "105.GOOGL",
        "AMZN": "105.AMZN", "NVDA": "105.NVDA", "META": "105.META",
        "TSLA": "105.TSLA", "AMD": "105.AMD", "INTC": "105.INTC",
        "NFLX": "105.NFLX", "CRM": "105.CRM", "ADBE": "105.ADBE",
        "PYPL": "105.PYPL", "QCOM": "105.QCOM", "AVGO": "105.AVGO",
        "TXN": "105.TXN", "AMAT": "105.AMAT", "MU": "105.MU",
        "ASML": "105.ASML", "TSM": "105.TSM", "ARM": "105.ARM",
        "SMCI": "105.SMCI", "PLTR": "105.PLTR", "SNOW": "105.SNOW",
        "MDB": "105.MDB", "CRWD": "105.CRWD", "PANW": "105.PANW",
        "SPOT": "105.SPOT", "UBER": "105.UBER", "ABNB": "105.ABNB",
        "COIN": "105.COIN", "MSTR": "105.MSTR",
        "ORCL": "106.ORCL", "IBM": "106.IBM", "BA": "106.BA",
        "JPM": "106.JPM", "V": "106.V", "MA": "106.MA",
        "JNJ": "106.JNJ", "PFE": "106.PFE", "WMT": "106.WMT",
        "PG": "106.PG", "KO": "106.KO", "DIS": "106.DIS",
        "NKE": "106.NKE", "MCD": "106.MCD", "HD": "106.HD",
        "UNH": "106.UNH", "LLY": "106.LLY", "MRK": "106.MRK",
        "DELL": "106.DELL", "VRT": "106.VRT",
        "SPY": "105.SPY", "QQQ": "105.QQQ", "IWM": "105.IWM",
        "XLK": "105.XLK", "XLF": "105.XLF", "XLE": "105.XLE",
        "DIA": "105.DIA", "VOO": "105.VOO", "ARKK": "105.ARKK",
        "SQQQ": "105.SQQQ", "SOXS": "105.SOXS", "TQQQ": "105.TQQQ",
        "SOXL": "105.SOXL", "SPXL": "105.SPXL",
        "^GSPC": "105.SPY", "^DJI": "105.DIA", "^IXIC": "105.QQQ",
    }

    def __init__(self):
        self.session = _get_session()
        self._delay = 0.5

    def _to_eastmoney_code(self, ticker: str) -> Optional[str]:
        t = ticker.upper()
        if t in self._market_map:
            return self._market_map[t]
        return f"105.{t}"

    def get_quote(self, ticker: str) -> Optional[dict]:
        code = self._to_eastmoney_code(ticker)
        if not code:
            return None

        params = {
            "secid": code,
            "fields": "f43,f44,f45,f46,f47,f48,f50,f51,f52,f55,f57,f58,f60,f116,f117,f162,f167,f170,f171,f173,f177,f187,f188,f190,f192",
            "ut": "fa5fd1943c7b386f172d6893dbbd1",
        }

        try:
            time.sleep(self._delay)
            resp = self.session.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("data"):
                return None

            d = data["data"]
            # Eastmoney US stock prices are in cents for some tickers
            # f43 is the latest price, but may need scaling
            raw_price = d.get("f43")
            if raw_price:
                price = raw_price / 100 if raw_price > 100 else raw_price
            else:
                price = None

            return {
                "ticker": ticker,
                "price": price,
                "open": d.get("f46") / 100 if d.get("f46") and d.get("f46", 0) > 100 else d.get("f46"),
                "high": d.get("f44") / 100 if d.get("f44") and d.get("f44", 0) > 100 else d.get("f44"),
                "low": d.get("f45") / 100 if d.get("f45") and d.get("f45", 0) > 100 else d.get("f45"),
                "volume": d.get("f47"),
                "turnover": d.get("f48"),
                "market_cap": d.get("f116"),
                "pe_ratio": d.get("f162"),
                "pb_ratio": d.get("f167"),
                "eps": d.get("f173"),
                "revenue": d.get("f187"),
                "net_profit": d.get("f188"),
                "profit_margin": d.get("f190"),
                "roe": d.get("f192"),
                "fetched_at": datetime.now().isoformat(),
                "source": "eastmoney",
            }
        except Exception as e:
            logger.error(f"Eastmoney quote fetch failed for {ticker}: {e}")
            return None

    def get_history(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "daily",
    ) -> Optional[pd.DataFrame]:
        code = self._to_eastmoney_code(ticker)
        if not code:
            return None

        klt_map = {"daily": "101", "weekly": "102", "monthly": "103"}

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = "20190101"
        if period == "1y":
            start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y%m%d")

        params = {
            "secid": code,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": klt_map.get(interval, "101"),
            "fqt": "1",
            "beg": start_date,
            "end": end_date,
            "lmt": "999999",
            "ut": "fa5fd1943c7b386f172d6893dbbd1",
        }

        try:
            time.sleep(self._delay)
            resp = self.session.get(self.KLINE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("data") or not data["data"].get("klines"):
                return None

            klines = data["data"]["klines"]
            records = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 8:
                    records.append({
                        "date": parts[0],
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": float(parts[5]),
                        "turnover": float(parts[6]),
                        "amplitude": float(parts[7]) if len(parts) > 7 else 0,
                    })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.rename(columns={"close": "adj_close"}, inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Fetched {len(df)} days of history for {ticker} from Eastmoney")
            return df

        except Exception as e:
            logger.error(f"Eastmoney history fetch failed for {ticker}: {e}")
            return None


# ── Combined Scraper ────────────────────────────────────────────────────

class MarketDataScraper:
    """
    Unified scraper that tries multiple sources.
    Priority: Tencent -> Eastmoney -> Sina
    """

    def __init__(self):
        self.tencent = TencentScraper()
        self.eastmoney = EastmoneyScraper()
        logger.info("MarketDataScraper initialized (Tencent + Eastmoney)")

    def get_quote(self, ticker: str) -> Optional[dict]:
        """Get quote, trying Tencent first (most reliable from China)."""
        result = self.tencent.get_quote(ticker)
        if result:
            return result

        result = self.eastmoney.get_quote(ticker)
        if result:
            return result

        logger.warning(f"All sources failed for {ticker} quote")
        return None

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, dict]:
        """Batch fetch quotes from Tencent."""
        return self.tencent.get_quotes_batch(tickers)

    def get_history(self, ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """Get history from Eastmoney."""
        return self.eastmoney.get_history(ticker, period=period)

    def get_multiple_tickers(
        self,
        tickers: list[str],
        period: str = "5y",
    ) -> dict[str, pd.DataFrame]:
        """Fetch history for multiple tickers."""
        results = {}
        for t in tickers:
            try:
                df = self.get_history(t, period=period)
                if df is not None and not df.empty:
                    results[t] = df
                else:
                    results[t] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to fetch {t}: {e}")
                results[t] = pd.DataFrame()
            time.sleep(0.3)
        return results

    def get_fundamentals(self, ticker: str) -> Optional[dict]:
        """Get fundamental data from Tencent (primary) or Eastmoney."""
        quote = self.tencent.get_quote(ticker)
        if quote and quote.get("pe_ratio"):
            return {
                "ticker": ticker,
                "trailing_pe": quote.get("pe_ratio"),
                "forward_pe": None,
                "price_to_sales": None,
                "price_to_book": None,
                "market_cap": quote.get("market_cap"),
                "eps": None,
                "current_price": quote.get("price"),
                "week_52_high": quote.get("week_52_high"),
                "week_52_low": quote.get("week_52_low"),
                "fetched_at": quote.get("fetched_at"),
            }

        quote = self.eastmoney.get_quote(ticker)
        if quote:
            return {
                "ticker": ticker,
                "trailing_pe": quote.get("pe_ratio"),
                "price_to_book": quote.get("pb_ratio"),
                "market_cap": quote.get("market_cap"),
                "eps": quote.get("eps"),
                "revenue": quote.get("revenue"),
                "current_price": quote.get("price"),
                "fetched_at": quote.get("fetched_at"),
            }

        return None
