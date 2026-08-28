"""Hunter Bot — Abstract Market Data Provider"""
from abc import ABC, abstractmethod
from typing import Optional, List
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

    # Additive realtime capability — optional, defaults to not supported.

    @property
    def supports_realtime_quotes(self) -> bool:
        return False

    @property
    def supports_realtime_trades(self) -> bool:
        return False

    async def fetch_quotes(self, ticker: str, limit: int = 1):
        """Return List[Quote]. Raises NotImplementedError if unsupported."""
        raise NotImplementedError(f"{self.name} does not support fetch_quotes")

    async def fetch_trades(self, ticker: str, limit: int = 1):
        """Return List[Trade]. Raises NotImplementedError if unsupported."""
        raise NotImplementedError(f"{self.name} does not support fetch_trades")

    async def fetch_history(self, ticker: str, period: str = "3mo", interval: str = "1d"):
        """Optional history fetch via provider abstraction. Default: not implemented."""
        raise NotImplementedError(f"{self.name} does not support fetch_history")

    @property
    def supports_options_realtime(self) -> bool:
        return False

    async def fetch_option_quotes(self, ticker: str, limit: int = 20):
        raise NotImplementedError

    async def fetch_option_trades(self, ticker: str, limit: int = 50):
        raise NotImplementedError
