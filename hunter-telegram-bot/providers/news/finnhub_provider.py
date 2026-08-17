"""Hunter Bot — Finnhub News Provider"""
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set
import aiohttp

from providers.news.base_provider import NewsProvider
from models.news import NewsItem, SourceTier
from core.exceptions import ProviderError
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

# HTTP statuses that are transient — retry with backoff
RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}

# HTTP statuses that are permanent client errors — never retry
NON_RETRYABLE_STATUSES: Set[int] = {400, 401, 403, 404}

MAX_RETRIES: int = 3
INITIAL_DELAY: float = 1.0
BACKOFF_MULTIPLIER: float = 2.0
MIN_DELAY: float = 0.5
MAX_DELAY: float = 4.0
PER_REQUEST_TIMEOUT: float = 10.0
TOTAL_RETRY_BUDGET: float = 20.0


class FinnhubNewsProvider(NewsProvider):
    name = "finnhub"

    def __init__(self):
        self.api_key = SETTINGS.finnhub_api_key
        self.base_url = "https://finnhub.io/api/v1"

    def _map_tier(self, source: str) -> SourceTier:
        return SOURCE_TIER_MAP.get(source, DEFAULT_TIER)

    # ------------------------------------------------------------------
    # Internal: make a single HTTP GET and return parsed JSON.
    # Raises ProviderError for any non-2xx response or on timeout/connection error.
    # ------------------------------------------------------------------
    async def _fetch_json(self, url: str, params: dict) -> dict:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=PER_REQUEST_TIMEOUT)
        ) as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception as e:
                        raise ProviderError(
                            f"Finnhub malformed JSON response",
                            provider=self.name,
                            retryable=False,
                        ) from e

                if resp.status in NON_RETRYABLE_STATUSES:
                    raise ProviderError(
                        f"Finnhub HTTP {resp.status}",
                        provider=self.name,
                        retryable=False,
                    )

                if resp.status in RETRYABLE_STATUSES:
                    retry_after: Optional[float] = None
                    try:
                        raw = resp.headers.get("Retry-After")
                        if raw:
                            retry_after = float(raw)
                    except (ValueError, TypeError):
                        pass
                    raise ProviderError(
                        f"Finnhub HTTP {resp.status}",
                        provider=self.name,
                        retryable=True,
                        retry_after=retry_after,
                    )

                # Any other non-200 status
                raise ProviderError(
                    f"Finnhub HTTP {resp.status}",
                    provider=self.name,
                    retryable=False,
                )

    # ------------------------------------------------------------------
    # Internal: GET with bounded exponential-backoff retry.
    # Respects TOTAL_RETRY_BUDGET so one request cannot block indefinitely.
    # Handles HTTP 429 with Retry-After when available and within budget.
    # ------------------------------------------------------------------
    async def _fetch_json_with_retry(self, url: str, params: dict) -> dict:
        delay = INITIAL_DELAY
        last_error: Optional[ProviderError] = None
        start_time = time.monotonic()
        attempts_made = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts_made += 1
            try:
                return await self._fetch_json(url, params)
            except ProviderError as e:
                last_error = e

                if not e.retryable:
                    LOGGER.warning(
                        "[Finnhub] Non-retryable HTTP error — giving up after %d attempt(s)",
                        attempts_made,
                    )
                    raise

                if attempt == MAX_RETRIES:
                    LOGGER.warning(
                        "[Finnhub] Max retries (%d) exhausted",
                        MAX_RETRIES,
                    )
                    raise

                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)

                if e.retry_after is not None:
                    # Retry-After is server-specified — use it directly, cap only by remaining budget.
                    # Do NOT apply BACKOFF_MULTIPLIER here; the server already told us the exact wait.
                    delay = min(e.retry_after, max(remaining, 0))
                else:
                    # No Retry-After: use exponential backoff
                    delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Finnhub] Retry budget exhausted after %d attempt(s) (elapsed=%.1fs)",
                        attempts_made,
                        time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Finnhub] Retry %d/%d in %.1fs (HTTP error, elapsed=%.1fs/%.1fs budget)%s",
                    attempt,
                    MAX_RETRIES,
                    delay,
                    time.monotonic() - start_time,
                    TOTAL_RETRY_BUDGET,
                    " [Retry-After]" if e.retry_after is not None else "",
                )
                await asyncio.sleep(delay)
                # After sleeping on a Retry-After, revert to exponential for subsequent retries
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)

            except asyncio.TimeoutError:
                last_error = ProviderError(
                    f"Finnhub timeout after {PER_REQUEST_TIMEOUT}s",
                    provider=self.name,
                    retryable=True,
                )
                if attempt == MAX_RETRIES:
                    LOGGER.warning("[Finnhub] Max retries exhausted — timeout")
                    raise last_error

                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Finnhub] Retry budget exhausted after %d attempt(s) — timeout (elapsed=%.1fs)",
                        attempts_made,
                        time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Finnhub] Retry %d/%d — timeout, sleeping %.1fs (elapsed=%.1fs/%.1fs budget)",
                    attempt,
                    MAX_RETRIES,
                    delay,
                    time.monotonic() - start_time,
                    TOTAL_RETRY_BUDGET,
                )
                await asyncio.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)

            except aiohttp.ClientError as e:
                last_error = ProviderError(
                    f"Finnhub connection error: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt == MAX_RETRIES:
                    LOGGER.warning("[Finnhub] Max retries exhausted — connection error")
                    raise last_error

                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Finnhub] Retry budget exhausted after %d attempt(s) — connection error (elapsed=%.1fs)",
                        attempts_made,
                        time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Finnhub] Retry %d/%d — connection error, sleeping %.1fs",
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)

        raise last_error

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

        data = await self._fetch_json_with_retry(url, params)

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
        except ProviderError:
            return False
        except Exception:
            return False
