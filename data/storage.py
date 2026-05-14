"""
Data storage module.
Handles caching and persistence using SQLite for structured data
and Parquet files for large price/volume datasets.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """Manages data persistence and caching."""

    def __init__(self, base_dir: str = "./data/cache"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "market_data.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    as_of_date TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, metric_name, as_of_date)
                )
            """)
            conn.commit()

    # ── Cache (simple key-value with TTL) ───────────────────────────────

    def has(self, key: str, ttl_hours: int = 24) -> bool:
        """Check if a cache entry exists and is not expired."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT created_at FROM cache WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            created = datetime.fromisoformat(row[0])
            return datetime.now() - created < timedelta(hours=ttl_hours)

    def get(self, key: str) -> Any:
        """Retrieve a cached value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if not row:
                return None
            return json.loads(row[0])

    def set(self, key: str, value: Any):
        """Store a value in cache (JSON serializable)."""
        serialized = json.dumps(value, default=str)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at)
                VALUES (?, ?, ?)
                """,
                (key, serialized, datetime.now().isoformat()),
            )

    # ── Price data persistence (Parquet) ────────────────────────────────

    def save_price_data(self, ticker: str, df: pd.DataFrame):
        """Save OHLCV data to Parquet file."""
        path = self.base_dir / f"prices_{ticker}.parquet"
        df.to_parquet(path, index=True)
        logger.info(f"Saved price data for {ticker}: {path}")

    def load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data from Parquet file."""
        path = self.base_dir / f"prices_{ticker}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        logger.info(f"Loaded price data for {ticker}: {path}")
        return df

    # ── Metrics storage (SQLite) ────────────────────────────────────────

    def save_metric(self, ticker: str, metric_name: str, value: float, as_of_date: str):
        """Save a computed metric (e.g. Shiller PE, bubble score) to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metrics (ticker, metric_name, metric_value, as_of_date)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, metric_name, value, as_of_date),
            )

    def get_metric_history(
        self, ticker: str, metric_name: str
    ) -> pd.DataFrame:
        """Retrieve historical values of a metric."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT as_of_date, metric_value
                FROM metrics
                WHERE ticker = ? AND metric_name = ?
                ORDER BY as_of_date ASC
                """,
                conn,
                params=(ticker, metric_name),
            )
        if not df.empty:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df

    def get_all_metrics(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """Get all metrics, optionally filtered by date."""
        with sqlite3.connect(self.db_path) as conn:
            if as_of_date:
                df = pd.read_sql_query(
                    "SELECT * FROM metrics WHERE as_of_date = ?",
                    conn,
                    params=(as_of_date,),
                )
            else:
                df = pd.read_sql_query("SELECT * FROM metrics", conn)
        return df
