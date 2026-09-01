"""Hunter Bot — YFinance News Provider (REAL, no API key required).

Uses the Yahoo Finance news endpoint exposed by the installed yfinance package.
Response shape (verified live, yfinance 1.2.x): a list of items shaped
{"id": ..., "content": {"title", "summary", "pubDate" (ISO-8601 Z),
"provider": {"displayName"}, "canonicalUrl": {"url"}}}.

LIMITATIONS: Yahoo news is an aggregator; publisher mix skews retail. Tier
mapping mirrors Finnhub conventions. Malformed/missing fields are skipped —
never fabricated.
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from providers.news.base_provider import NewsProvider
from models.news import NewsItem, SourceTier, ensure_utc
from utils.logger import LOGGER
from providers.market_data.yfinance_concurrency import get_yfinance_semaphore


SOURCE_TIER_MAP = {
    "Reuters": SourceTier.TIER_2_MAJOR,
    "Bloomberg": SourceTier.TIER_2_MAJOR,
    "WSJ": SourceTier.TIER_2_MAJOR,
    "Barron's": SourceTier.TIER_2_MAJOR,
    "Associated Press": SourceTier.TIER_2_MAJOR,
    "CNBC": SourceTier.TIER_3_FINANCIAL,
    "Benzinga": SourceTier.TIER_3_FINANCIAL,
    "MarketWatch": SourceTier.TIER_3_FINANCIAL,
    "Seeking Alpha": SourceTier.TIER_3_FINANCIAL,
    "Yahoo Finance": SourceTier.TIER_3_FINANCIAL,
    "Yahoo Finance Video": SourceTier.TIER_3_FINANCIAL,
    "Investor's Business Daily": SourceTier.TIER_3_FINANCIAL,
    "Insider Monkey": SourceTier.TIER_3_FINANCIAL,
    "Motley Fool": SourceTier.TIER_4_UNVERIFIED,
}
DEFAULT_TIER = SourceTier.TIER_3_FINANCIAL


def _parse_pub_date(raw) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_utc(parsed)


class YFinanceNewsProvider(NewsProvider):
    name = "yfinance_news"

    def __init__(self, news_fn=None):
        self._news_fn = news_fn or self._default_news_fn

    @staticmethod
    def _default_news_fn(ticker: str, count: int):
        import yfinance as yf
        return yf.Ticker(ticker).get_news(count=count)

    def _map_tier(self, source: str) -> SourceTier:
        return SOURCE_TIER_MAP.get(source, DEFAULT_TIER)

    async def fetch_news(self, ticker: str, since) -> List[NewsItem]:
        async with get_yfinance_semaphore():
            try:
                raw_items = await asyncio.to_thread(self._news_fn, ticker, 20)
            except Exception as e:
                LOGGER.warning(f"[{self.name}] fetch failed for {ticker}: {e}")
                return []

        items: List[NewsItem] = []
        if not isinstance(raw_items, list):
            return []
        for raw in raw_items:
            item = self._map_item(ticker, raw)
            if item is None:
                continue
            if item.published_at is not None and ensure_utc(item.published_at) < since:
                continue
            items.append(item)
        return items

    def _map_item(self, ticker: str, raw) -> Optional[NewsItem]:
        if not isinstance(raw, dict):
            return None
        content = raw.get("content") if isinstance(raw.get("content"), dict) else {}
        headline = content.get("title") or raw.get("title")
        if not headline or not isinstance(headline, str):
            return None
        item_id = raw.get("id") or headline[:80]
        provider = content.get("provider") or {}
        source = provider.get("displayName") if isinstance(provider, dict) else None
        canonical = content.get("canonicalUrl") or {}
        url = canonical.get("url") if isinstance(canonical, dict) else None
        published = _parse_pub_date(content.get("pubDate") or content.get("displayTime"))
        return NewsItem(
            id=f"{self.name}:{item_id}",
            ticker=ticker,
            headline=headline.strip(),
            source=(source or "Yahoo Finance"),
            source_tier=self._map_tier(source or ""),
            url=url if isinstance(url, str) else None,
            summary=content.get("summary") if isinstance(content.get("summary"), str) else None,
            published_at=published,
        )

    async def health_check(self) -> bool:
        async with get_yfinance_semaphore():
            try:
                items = await asyncio.to_thread(self._news_fn, "AAPL", 1)
                return isinstance(items, list)
            except Exception as e:
                LOGGER.warning(f"[{self.name}] health check failed: {e}")
                return False
