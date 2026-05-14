"""
CFFEX Option Chain Data Module.
Fetches real-time/daily option chain data from:
1. CFFEX official website (daily settlement data)
2. Eastmoney (real-time option quotes)
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
    })
    return session


# ── CFFEX Official Settlement Data ────────────────────────────────────

class CFFEXSettlementScraper:
    """
    Fetches daily settlement data from CFFEX website.
    http://www.cffex.com.cn/sj/jshq/
    """

    BASE_URL = "http://www.cffex.com.cn/sj/jshq/"

    def __init__(self):
        self.session = _get_session()
        self._delay = 1.0

    def get_daily_settlement(self, date: Optional[str] = None) -> dict:
        """
        Fetch daily settlement data for index options.

        Args:
            date: YYYYMMDD format, defaults to today

        Returns:
            dict with IO and MO settlement data
        """
        if date is None:
            date = datetime.now().strftime("%Y%m")

        # CFFEX uses monthly directories
        url = f"{self.BASE_URL}{date}/"

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, timeout=10)
            resp.encoding = "utf-8"

            # Find the latest settlement file
            # Files are usually named like: jshq_MO_YYYYMMDD.csv
            pattern = r'jshq_(IO|MO)_(\d{8})\.(csv|txt)'
            matches = re.findall(pattern, resp.text)

            result = {"IO": None, "MO": None}

            for product, file_date, ext in matches:
                file_url = f"{url}jshq_{product}_{file_date}.{ext}"
                try:
                    time.sleep(0.5)
                    file_resp = self.session.get(file_url, timeout=10)
                    file_resp.encoding = "gbk"  # CFFEX files are usually GBK

                    # Parse the CSV/TXT file
                    lines = file_resp.text.strip().split("\n")
                    if len(lines) > 2:
                        # First line is header, rest is data
                        data_lines = []
                        for line in lines[1:]:
                            parts = line.strip().split(",")
                            if len(parts) >= 10:
                                data_lines.append({
                                    "contract": parts[0].strip(),
                                    "open": self._sf(parts[1]),
                                    "high": self._sf(parts[2]),
                                    "low": self._sf(parts[3]),
                                    "close": self._sf(parts[4]),
                                    "settlement": self._sf(parts[5]),
                                    "change": self._sf(parts[6]),
                                    "change_pct": self._sf(parts[7]),
                                    "volume": self._sf(parts[8]),
                                    "open_interest": self._sf(parts[9]),
                                    "date": file_date,
                                })

                        result[product] = data_lines
                        logger.info(f"Fetched {len(data_lines)} {product} settlement records for {file_date}")

                except Exception as e:
                    logger.warning(f"Failed to fetch {product} settlement: {e}")

            return result

        except Exception as e:
            logger.error(f"CFFEX settlement fetch failed: {e}")
            return {"IO": [], "MO": []}

    @staticmethod
    def _sf(val, default=None):
        try:
            if not val or val.strip() in ("", "-"):
                return default
            return float(val.strip())
        except (ValueError, TypeError):
            return default


# ── Eastmoney Option Quotes ───────────────────────────────────────────

class EastmoneyOptionScraper:
    """
    Fetches option quotes from Eastmoney.
    Note: Real-time option data may have limitations.
    """

    # Eastmoney option codes
    # IO: 沪深300期权
    # MO: 中证1000期权
    # Format: Options are listed under specific codes

    def __init__(self):
        self.session = _get_session()
        self._delay = 0.5

    def get_option_chain(
        self,
        product: str = "MO",
        expiry_month: Optional[str] = None,
    ) -> list[dict]:
        """
        Get option chain for a specific product.

        Args:
            product: "IO" (沪深300) or "MO" (中证1000)
            expiry_month: YYYYMM format for specific expiry

        Returns:
            List of option contracts with quotes
        """
        # Eastmoney API for options
        # The actual API endpoint may vary; this is a best-effort implementation
        url = "https://push2.eastmoney.com/api/qt/clist/get"

        # Eastmoney option market codes
        market_code = "8" if product == "IO" else "9"

        params = {
            "pn": 1,
            "pz": 200,  # Page size
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": f"m:{market_code}",
            "fields": "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18",
            # f2=latest, f3=change%, f4=change, f5=volume, f6=turnover,
            # f12=code, f14=name, f15=high, f16=low, f17=open, f18=prev_close
        }

        try:
            time.sleep(self._delay)
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("data") or not data["data"].get("diff"):
                logger.warning(f"No option data from Eastmoney for {product}")
                return []

            options = []
            for item in data["data"]["diff"]:
                option = {
                    "code": item.get("f12", ""),
                    "name": item.get("f14", ""),
                    "latest": item.get("f2"),
                    "change_pct": item.get("f3"),
                    "change": item.get("f4"),
                    "volume": item.get("f5"),
                    "turnover": item.get("f6"),
                    "amplitude": item.get("f7"),
                    "high": item.get("f15"),
                    "low": item.get("f16"),
                    "open": item.get("f17"),
                    "prev_close": item.get("f18"),
                    "fetched_at": datetime.now().isoformat(),
                    "source": "eastmoney",
                }

                # Filter by product if needed
                if product.upper() in option["name"] or product.upper() in option["code"]:
                    options.append(option)

            logger.info(f"Fetched {len(options)} {product} option quotes from Eastmoney")
            return options

        except Exception as e:
            logger.error(f"Eastmoney option fetch failed for {product}: {e}")
            return []


# ── Combined Option Chain Data ────────────────────────────────────────

class OptionChainData:
    """
    Unified option chain data fetcher.
    Combines settlement data (daily) with quotes (real-time).
    """

    def __init__(self):
        self.cffex = CFFEXSettlementScraper()
        self.em = EastmoneyOptionScraper()
        logger.info("OptionChainData initialized (CFFEX + Eastmoney)")

    def get_settlement_data(self, date: Optional[str] = None) -> dict:
        """Get daily settlement data from CFFEX."""
        return self.cffex.get_daily_settlement(date)

    def get_live_quotes(self, product: str = "MO") -> list[dict]:
        """Get live option quotes from Eastmoney."""
        return self.em.get_option_chain(product)

    def analyze_liquidity(self, product: str = "MO") -> dict:
        """
        Analyze option liquidity for a product.
        Returns dict with liquidity metrics.
        """
        quotes = self.get_live_quotes(product)

        if not quotes:
            return {"error": "No quote data available", "product": product}

        volumes = [q.get("volume", 0) or 0 for q in quotes]
        turnover = [q.get("turnover", 0) or 0 for q in quotes]

        # Sort by volume to find most liquid contracts
        liquid_contracts = sorted(
            quotes,
            key=lambda q: q.get("volume", 0) or 0,
            reverse=True
        )[:10]

        return {
            "product": product,
            "total_contracts": len(quotes),
            "contracts_with_volume": sum(1 for v in volumes if v > 0),
            "total_volume": sum(volumes),
            "total_turnover": sum(turnover),
            "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
            "most_liquid": [
                {
                    "code": q.get("code"),
                    "name": q.get("name"),
                    "volume": q.get("volume"),
                    "latest": q.get("latest"),
                }
                for q in liquid_contracts[:5]
            ],
            "fetched_at": datetime.now().isoformat(),
        }

    def format_liquidity_report(self, product: str = "MO") -> str:
        """Generate a liquidity report."""
        analysis = self.analyze_liquidity(product)

        if "error" in analysis:
            return f"Error analyzing {product} liquidity: {analysis['error']}"

        lines = []
        lines.append("=" * 70)
        lines.append(f"📊 {product} 期权流动性分析")
        lines.append("=" * 70)
        lines.append(f"总合约数: {analysis['total_contracts']}")
        lines.append(f"有成交合约: {analysis['contracts_with_volume']}")
        lines.append(f"总成交量: {analysis['total_volume']:,.0f}")
        lines.append(f"总成交额: {analysis['total_turnover']:,.0f}")
        lines.append(f"平均成交量: {analysis['avg_volume']:,.0f}")
        lines.append("")
        lines.append("🔥 最活跃合约 Top 5:")
        lines.append(f"  {'合约代码':<15} {'合约名称':<25} {'成交量':>10} {'最新价':>8}")
        lines.append(f"  {'-'*58}")

        for c in analysis.get("most_liquid", []):
            lines.append(
                f"  {c['code']:<15} {c['name']:<25} {c['volume']:>10,} {c['latest']:>8}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def main():
    """Test option chain data."""
    logging.basicConfig(level=logging.INFO)

    oc = OptionChainData()

    # Test settlement data
    settlement = oc.get_settlement_data()
    if settlement.get("MO"):
        logger.info(f"MO settlement: {len(settlement['MO'])} records")

    # Test live quotes
    quotes = oc.get_live_quotes("MO")
    logger.info(f"MO live quotes: {len(quotes)} contracts")

    # Test liquidity analysis
    print(oc.format_liquidity_report("MO"))


if __name__ == "__main__":
    main()
