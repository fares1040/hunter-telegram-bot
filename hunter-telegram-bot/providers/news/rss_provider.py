"""Hunter Bot — RSS News Provider (STUB / FUTURE)"""
from datetime import datetime
from typing import List
from providers.news.base_provider import NewsProvider
from models.news import NewsItem


class RSSNewsProvider(NewsProvider):
    name = "rss"

    async def fetch_news(self, ticker: str, since: datetime) -> List[NewsItem]:
        return []

    async def health_check(self) -> bool:
        return True
