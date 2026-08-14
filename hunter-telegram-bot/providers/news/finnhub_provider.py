"""Hunter Bot — Finnhub News Provider"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List
import aiohttp

from providers.news.base_provider import NewsProvider
from models.news import NewsItem, SourceTier
from core.exceptions import ProviderError
from utils.retry import async_retry
from utils.logger import LOGGER
from config.settings import SETTINGS


SOURCE_TIER_MAP = {
    "Reuters": SourceTier.TIER_2_MAJOR,
    "Bloomberg": SourceTier.TIER_2_MAJOR,
    "WSJ": SourceTier.TIER_2_MAJOR,
    "CNBC": SourceTier.TIER_3_FINANCIAL,
    "Benzinga": SourceTier.TIER_3_FINANCIAL,
    "MarketWatch": SourceTier.TIER_3_FINANCIAL,
    "Seeking Alpha": SourceTier.TIER_3_FINANCIAL,
    "Yahoo Finance": SourceTier.TIER_3_FINANCIAL,
    "Twitter": SourceTier.TIER_4_UNVERIFIED,
    "StockTwits": SourceTier.TIER_4_UNVERIFIED,
}

DEFAULT_TIER = SourceTier.TIER_3_FINANCIAL


class FinnhubNewsProvider(NewsProvider):
    name = "finnhub"

    def __init__(self):
        self.api_key = SETTINGS.finnhub_api_key
        self.base_url = "https://finnhub.io/api/v1"

    def _map_tier(self, source: str) -> SourceTier:
        return SOURCE_TIER_MAP.get(source, DEFAULT_TIER)

    @async_retry(max_retries=2, delay=1.0, exceptions=(Exception,))
    async def fetch_news(self, ticker: str, since: datetime) -> List[NewsItem]:
        if not self.api_key:
            LOGGER.warning("[Finnhub] No API key configured")
            return []

        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        from_date = since.strftime("%Y-%m-%d")

        url = f"{self.base_url}/company-news"
        params = {
            "symbol": ticker.upper(),
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        LOGGER.warning(f"[Finnhub] HTTP {resp.status} for {ticker}")
                        return []
                    data = await resp.json()
            except Exception as e:
                LOGGER.warning(f"[Finnhub] Request failed: {e}")
                return []

        items = []
        for article in data:
            published = article.get("datetime")
            pub_dt = None
            if published:
                try:
                    pub_dt = datetime.fromtimestamp(published, tz=timezone.utc)
                except (OSError, ValueError):
                    pass

            source_name = article.get("source", "Unknown")
            item = NewsItem(
                id=f"finnhub_{article.get('id', '')}",
                ticker=ticker.upper(),
                headline=article.get("headline", ""),
                source=source_name,
                source_tier=self._map_tier(source_name),
                url=article.get("url"),
                summary=article.get("summary", ""),
                published_at=pub_dt,
            )
            items.append(item)

        LOGGER.info(f"[Finnhub] Fetched {len(items)} articles for {ticker}")
        return items

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            await self.fetch_news("AAPL", datetime.now(timezone.utc) - timedelta(days=1))
            return True
        except Exception:
            return False
