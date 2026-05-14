"""
Masters News Scanner.

Scrapes recent news and opinions for living investment masters from
web sources accessible from China (Tencent, Bing News, RSS feeds).
Results are cached to avoid repeated scraping.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Sentiment Keywords ────────────────────────────────────────────────

BULLISH_KEYWORDS = [
    "bullish", "opportunity", "buy", "optimistic", "undervalued", "attractive",
    "upgrade", "positive", "growth", "invest", "accumulate", "long",
    "看好", "买入", "增持", "乐观", "低估", "机会",
]

BEARISH_KEYWORDS = [
    "bearish", "bubble", "crash", "overvalued", "warning", "caution", "short",
    "risk", "downgrade", "negative", "decline", "sell", "recession", "correction",
    "看空", "卖出", "减持", "谨慎", "高估", "风险", "泡沫", "崩盘",
]


# ── Master Search Queries ─────────────────────────────────────────────

MASTER_QUERIES = {
    "warren_buffett": "Warren Buffett Berkshire market outlook",
    "bill_ackman": "Bill Ackman Pershing Square market view",
    "cathie_wood": "Cathie Wood ARK Invest AI stocks",
    "aswath_damodaran": "Damodaran market valuation overvalued",
    "stanley_druckenmiller": "Druckenmiller market outlook portfolio",
    "nassim_taleb": "Nassim Taleb market bubble risk",
    "michael_burry": "Michael Burry Scion market short",
    "ray_dalio": "Ray Dalio Bridgewater economic cycle outlook",
}

# ── News Sources ──────────────────────────────────────────────────────

NEWS_SOURCES = [
    {
        "name": "bing_news",
        "url_template": "https://www.bing.com/news/search?q={query}&format=rss",
    },
    {
        "name": "google_news",
        "url_template": "https://news.google.com/rss/search?q={query}&hl=en&gl=us&ceid=us:en",
    },
]


class MastersNewsScanner:
    """
    Scrapes recent news/opinions for living investment masters.

    Uses Bing News RSS as primary source (more accessible from China),
    falls back to Google News RSS. Results cached in data/cache/masters/.
    """

    CACHE_TTL = 3600 * 4  # 4 hours

    def __init__(self, cache_dir: Path = None):
        self.session = self._get_session()
        base = Path(__file__).parent.parent
        self.cache_dir = cache_dir or base / "data" / "cache" / "masters"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/rss+xml, application/xml, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return session

    def scan(self, master_id: str) -> Optional[dict]:
        """
        Scan for recent news about a master.

        Returns:
            dict with keys: headline, sentiment, summary, source, date
            or None if no results / error.
        """
        # Check cache
        cached = self._load_cache(master_id)
        if cached and not self._is_stale(cached):
            return cached

        # Fetch from sources
        query = MASTER_QUERIES.get(master_id, master_id)
        result = None

        for source in NEWS_SOURCES:
            try:
                result = self._fetch_rss(source["url_template"].format(query=query))
                if result:
                    break
            except Exception as e:
                logger.debug(f"Source {source['name']} failed for {master_id}: {e}")
                continue

        if result:
            result["_cached_at"] = time.time()
            self._save_cache(master_id, result)

        return result

    def _fetch_rss(self, url: str) -> Optional[dict]:
        """Fetch and parse RSS feed, return first relevant item."""
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"RSS fetch failed: {e}")
            return None

        # Simple RSS parsing: extract items
        text = resp.text
        items = []
        current = {}

        for line in text.split("\n"):
            line = line.strip()
            if "<item>" in line:
                current = {}
            elif "</item>" in line:
                if current.get("title"):
                    items.append(current)
                current = {}
            elif "<title>" in line and current is not None:
                current["title"] = self._strip_tags(line)
            elif "<link>" in line and current is not None and "link" not in current:
                current["link"] = self._strip_tags(line)
            elif "<pubDate>" in line and current is not None:
                current["date"] = self._strip_tags(line)
            elif "<description>" in line and current is not None:
                current["description"] = self._strip_tags(line)[:300]

        if not items:
            return None

        # Take the most recent item
        item = items[0]
        title = item.get("title", "")
        sentiment = self._classify_sentiment(title + " " + item.get("description", ""))

        return {
            "headline": title,
            "sentiment": sentiment,
            "summary": item.get("description", "")[:200],
            "source": item.get("link", ""),
            "date": item.get("date", ""),
        }

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove XML tags from text."""
        import re
        return re.sub(r"<[^>]+>", "", text).strip()

    @staticmethod
    def _classify_sentiment(text: str) -> str:
        """Rule-based sentiment classification from text."""
        text_lower = text.lower()
        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw.lower() in text_lower)

        if bullish_count > bearish_count and bullish_count >= 1:
            return "bullish"
        elif bearish_count > bullish_count and bearish_count >= 1:
            return "bearish"
        return "neutral"

    def _cache_path(self, master_id: str) -> Path:
        return self.cache_dir / f"{master_id}.json"

    def _load_cache(self, master_id: str) -> Optional[dict]:
        path = self._cache_path(master_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_cache(self, master_id: str, data: dict):
        path = self._cache_path(master_id)
        try:
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.debug(f"Cache save failed for {master_id}: {e}")

    def _is_stale(self, data: dict) -> bool:
        cached_at = data.get("_cached_at", 0)
        return (time.time() - cached_at) > self.CACHE_TTL
