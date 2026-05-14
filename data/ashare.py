"""
A-Share Data Module.
Fetches real-time A-share index and stock data from China-accessible sources.
Primary: Tencent Finance (腾讯行情)
Fallback: Sina Finance (新浪财经)
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


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    })
    return session


# ── Tencent A-Share (腾讯A股行情) ─────────────────────────────────────

class TencentAShareScraper:
    """
    Tencent A-share realtime data.
    Format: http://qt.gtimg.cn/q=sh000905 (CSI 1000)
    """

    BASE_URL = "http://qt.gtimg.cn/q="

    # Map index/stock names to Tencent codes
    CODE_MAP = {
        "CSI300": "sh000300",
        "CSI500": "sh000905",
        "CSI1000": "sh000852",
        "SSE50": "sh000016",
        "SSE": "sh000001",      # 上证指数
        "SZSE": "sz399001",     # 深证成指
        "CHINEXT": "sz399006",  # 创业板指
        "STAR50": "sh000688",   # 科创50
    }

    def __init__(self):
        self.session = _get_session()
        self._delay = 0.2

    def _sf(self, val, default=None):
        """Safe float."""
        try:
            if not val or val in ("-", "None", "none", ""):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_index(self, name: str) -> Optional[dict]:
        """
        Get realtime A-share index data.

        Tencent format for indices:
        v_sh000905="1~中证1000~000905~6450.12~6430.50~6440.00~...";
        [3]=price, [4]=prev_close, [5]=open, [6]=volume,
        [30]=datetime, [31]=change, [32]=change_pct,
        [33]=high, [34]=low, [35]=currency
        """
        code = self.CODE_MAP.get(name)
        if not code:
            logger.warning(f"Unknown A-share index: {name}")
            return None

        url = f"{self.BASE_URL}{code}"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=10)
            resp.encoding = "gbk"
            text = resp.text.strip()

            if not text or '""' in text:
                return None

            match = re.search(r'="(.+?)"', text)
            if not match:
                return None

            parts = match.group(1).split("~")
            if len(parts) < 6:
                return None

            return {
                "name": name,
                "display_name": parts[1] if len(parts) > 1 else name,
                "price": self._sf(parts[3]),
                "prev_close": self._sf(parts[4]),
                "open": self._sf(parts[5]),
                "volume": self._sf(parts[6]),
                "change": self._sf(parts[31]),
                "change_pct": self._sf(parts[32]),
                "high": self._sf(parts[33]),
                "low": self._sf(parts[34]),
                "datetime": parts[30] if len(parts) > 30 else None,
                "fetched_at": datetime.now().isoformat(),
                "source": "tencent_ashare",
            }
        except Exception as e:
            logger.error(f"Tencent A-share fetch failed for {name}: {e}")
            return None

    def get_batch(self, names: list[str]) -> dict[str, dict]:
        """Fetch multiple indices in one request."""
        codes = []
        for name in names:
            code = self.CODE_MAP.get(name)
            if code:
                codes.append(code)

        if not codes:
            return {}

        url = f"{self.BASE_URL}{','.join(codes)}"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=10)
            resp.encoding = "gbk"
            text = resp.text.strip()

            results = {}
            for line in text.split(";"):
                line = line.strip()
                if not line or "=" not in line:
                    continue

                match = re.search(r'v_(\w+)="(.+?)"', line)
                if not match:
                    continue

                code = match.group(1)
                parts = match.group(2).split("~")
                if len(parts) < 6:
                    continue

                # Find the name for this code
                name = None
                for n, c in self.CODE_MAP.items():
                    if c == code:
                        name = n
                        break

                if not name:
                    continue

                results[name] = {
                    "name": name,
                    "display_name": parts[1] if len(parts) > 1 else name,
                    "price": self._sf(parts[3]),
                    "prev_close": self._sf(parts[4]),
                    "open": self._sf(parts[5]),
                    "volume": self._sf(parts[6]),
                    "change": self._sf(parts[31]),
                    "change_pct": self._sf(parts[32]),
                    "high": self._sf(parts[33]),
                    "low": self._sf(parts[34]),
                    "datetime": parts[30] if len(parts) > 30 else None,
                    "fetched_at": datetime.now().isoformat(),
                    "source": "tencent_ashare",
                }

            return results
        except Exception as e:
            logger.error(f"Tencent A-share batch fetch failed: {e}")
            return {}


# ── Sina A-Share (新浪A股行情) ────────────────────────────────────────

class SinaAShareScraper:
    """
    Sina A-share realtime data (fallback).
    Format: http://hq.sinajs.cn/list=sh000905
    """

    BASE_URL = "http://hq.sinajs.cn/list="

    CODE_MAP = {
        "CSI300": "sh000300",
        "CSI500": "sh000905",
        "CSI1000": "sh000852",
        "SSE50": "sh000016",
        "SSE": "sh000001",
        "SZSE": "sz399001",
        "CHINEXT": "sz399006",
        "STAR50": "sh000688",
    }

    def __init__(self):
        self.session = _get_session()
        self.session.headers.update({
            "Referer": "https://finance.sina.com.cn/",
        })
        self._delay = 0.3

    def _sf(self, val, default=None):
        try:
            if not val or val == "":
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_index(self, name: str) -> Optional[dict]:
        code = self.CODE_MAP.get(name)
        if not code:
            return None

        url = f"{self.BASE_URL}{code}"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=10)
            resp.encoding = "gbk"
            text = resp.text.strip()

            if not text or '""' in text:
                return None

            match = re.search(r'="(.+?)"', text)
            if not match:
                return None

            parts = match.group(1).split(",")
            # Sina format: name, open, prev_close, price, high, low, ...
            if len(parts) < 32:
                return None

            return {
                "name": name,
                "display_name": parts[0],
                "open": self._sf(parts[1]),
                "prev_close": self._sf(parts[2]),
                "price": self._sf(parts[3]),
                "high": self._sf(parts[4]),
                "low": self._sf(parts[5]),
                "volume": self._sf(parts[8]) * 100 if len(parts) > 8 else None,  # 手 -> 股
                "datetime": parts[30] + " " + parts[31] if len(parts) > 31 else None,
                "change": self._sf(parts[3]) - self._sf(parts[2]) if parts[3] and parts[2] else None,
                "change_pct": ((self._sf(parts[3]) - self._sf(parts[2])) / self._sf(parts[2]) * 100) if parts[3] and parts[2] else None,
                "fetched_at": datetime.now().isoformat(),
                "source": "sina_ashare",
            }
        except Exception as e:
            logger.error(f"Sina A-share fetch failed for {name}: {e}")
            return None


# ── Combined A-Share Scraper ──────────────────────────────────────────

class AShareDataFetcher:
    """
    Unified A-share data fetcher.
    Tries Tencent first, falls back to Sina.
    """

    def __init__(self):
        self.tencent = TencentAShareScraper()
        self.sina = SinaAShareScraper()
        logger.info("AShareDataFetcher initialized (Tencent + Sina)")

    def get_index(self, name: str) -> Optional[dict]:
        """Get index data, trying both sources."""
        result = self.tencent.get_index(name)
        if result and result.get("price"):
            return result

        result = self.sina.get_index(name)
        if result and result.get("price"):
            return result

        logger.warning(f"All sources failed for A-share index: {name}")
        return None

    def get_all_indices(self) -> dict[str, dict]:
        """Fetch all major A-share indices."""
        names = ["CSI300", "CSI500", "CSI1000", "SSE50", "SSE", "SZSE", "CHINEXT", "STAR50"]
        results = {}

        # Try batch fetch from Tencent first
        batch = self.tencent.get_batch(names)
        if batch:
            results.update(batch)

        # Fill in any missing from individual fetches
        for name in names:
            if name not in results:
                result = self.get_index(name)
                if result:
                    results[name] = result

        return results

    def get_history(self, name: str, days: int = 120) -> Optional[pd.DataFrame]:
        """
        Get historical daily data for A-share index from Eastmoney.
        """
        code_map = {
            "CSI300": "1.000300",
            "CSI500": "1.000905",
            "CSI1000": "1.000852",
            "SSE50": "1.000016",
            "SSE": "1.000001",
            "SZSE": "0.399001",
            "CHINEXT": "0.399006",
            "STAR50": "1.000688",
        }

        code = code_map.get(name)
        if not code:
            return None

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y%m%d")

        params = {
            "secid": code,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",  # daily
            "fqt": "1",
            "beg": start_date,
            "end": end_date,
            "lmt": "999999",
            "ut": "fa5fd1943c7b386f172d6893dbbd1",
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
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
                        "turnover": float(parts[6]) if parts[6] else 0,
                        "amplitude": float(parts[7]) if parts[7] else 0,
                    })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Fetched {len(df)} days of history for {name}")
            return df

        except Exception as e:
            logger.error(f"Eastmoney history fetch failed for {name}: {e}")
            return None

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI from price history."""
        if df is None or df.empty or "close" not in df.columns:
            return None

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        return round(rsi, 1)

    def calculate_ma(self, df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """Calculate moving average."""
        if df is None or df.empty or "close" not in df.columns:
            return None
        if len(df) < period:
            return None
        return round(df["close"].rolling(window=period).mean().iloc[-1], 1)
