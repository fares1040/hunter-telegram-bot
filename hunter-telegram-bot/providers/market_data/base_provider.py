"""Hunter Bot — Abstract Market Data Provider"""
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from models.ticker import TickerData


class MarketDataProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def fetch_ticker(self, ticker: str, timestamp: Optional[datetime] = None) -> TickerData:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_realtime(self) -> bool:
        pass
