"""Hunter Bot — Abstract News Provider"""
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from models.news import NewsItem


class NewsProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def fetch_news(self, ticker: str, since: datetime) -> List[NewsItem]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass
