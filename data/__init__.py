"""
Data acquisition module.
Handles fetching market data, financial statements, macro indicators, and sentiment data.
"""

from data.fetcher import MarketDataFetcher
from data.storage import DataStorage

__all__ = ["MarketDataFetcher", "DataStorage"]
