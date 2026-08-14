"""Polygon.io real-time-capable stock market data provider."""
import aiohttp, pandas as pd, asyncio, logging, time
from datetime import datetime, timedelta
from typing import Optional, Set
from providers.market_data.base_provider import MarketDataProvider
from models.ticker import TickerData
from models.session import SessionSnapshot
from core.session_clock import SessionClock, MarketSession
from core.exceptions import ProviderError, DataInsufficientError

LOGGER = logging.getLogger("hunter")

# HTTP statuses that are retryable (transient server-side or rate-limit)
RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}

# HTTP statuses that are permanent client errors — never retry
NON_RETRYABLE_STATUSES: Set[int] = {400, 401, 403, 404}

MAX_RETRIES: int = 3
INITIAL_DELAY: float = 1.0          # seconds
BACKOFF_MULTIPLIER: float = 2.0
MIN_DELAY: float = 0.5               # minimum delay between retries
MAX_DELAY: float = 4.0               # hard cap per backoff interval
PER_REQUEST_TIMEOUT: float = 10.0     # seconds per HTTP attempt
TOTAL_RETRY_BUDGET: float = 20.0     # total budget for all attempts + backoffs


class PolygonProvider(MarketDataProvider):
    name = "polygon"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def is_realtime(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal: make a single HTTP GET and return parsed JSON.
    # Raises ProviderError for any non-2xx response.
    # ------------------------------------------------------------------
    async def _get_json(self, url: str, params: dict) -> dict:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=PER_REQUEST_TIMEOUT)
        ) as session:
            async with session.get(url, params=params) as r:
                if r.status == 200:
                    return await r.json()
                raise ProviderError(
                    f"Polygon HTTP {r.status} for {url.split('/')[-1]}",
                    provider=self.name,
                    retryable=r.status in RETRYABLE_STATUSES,
                )

    # ------------------------------------------------------------------
    # Internal: GET with bounded exponential-backoff retry.
    # Respects TOTAL_RETRY_BUDGET so one request cannot block indefinitely.
    # Handles HTTP 429 with Retry-After when available and within budget.
    # ------------------------------------------------------------------
    async def _get_json_with_retry(self, url: str, params: dict) -> dict:
        delay = INITIAL_DELAY
        last_error: Optional[ProviderError] = None
        start_time = time.monotonic()
        attempts_made = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts_made += 1
            try:
                return await self._get_json(url, params)
            except ProviderError as e:
                last_error = e

                if not e.retryable:
                    LOGGER.warning(
                        "[Polygon] Non-retryable HTTP %s — giving up after %d attempt(s)",
                        e, attempts_made,
                    )
                    raise

                if attempt == MAX_RETRIES:
                    LOGGER.warning(
                        "[Polygon] Max retries (%d) exhausted for %s",
                        MAX_RETRIES, url.split("/")[-1],
                    )
                    raise

                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)
                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)
                delay = min(delay, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Polygon] Retry budget exhausted after %d attempt(s) for %s (elapsed=%.1fs)",
                        attempts_made, url.split("/")[-1], time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Polygon] Retry %d/%d for %s in %.1fs (HTTP %s, elapsed=%.1fs/%.1fs budget)",
                    attempt, MAX_RETRIES, url.split("/")[-1], delay, e,
                    time.monotonic() - start_time, TOTAL_RETRY_BUDGET,
                )
                await asyncio.sleep(delay)

            except asyncio.TimeoutError:
                last_error = ProviderError(
                    f"Polygon timeout after {PER_REQUEST_TIMEOUT}s",
                    provider=self.name,
                    retryable=True,
                )
                if attempt == MAX_RETRIES:
                    LOGGER.warning("[Polygon] Max retries (%d) exhausted — timeout", MAX_RETRIES)
                    raise last_error

                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Polygon] Retry budget exhausted after %d attempt(s) — timeout (elapsed=%.1fs)",
                        attempts_made, time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Polygon] Retry %d/%d — timeout, sleeping %.1fs (elapsed=%.1fs/%.1fs budget)",
                    attempt, MAX_RETRIES, delay, time.monotonic() - start_time, TOTAL_RETRY_BUDGET,
                )
                await asyncio.sleep(delay)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY)

            except aiohttp.ClientError as e:
                last_error = ProviderError(
                    f"Polygon connection error: {e}",
                    provider=self.name,
                    retryable=True,
                )
                if attempt == MAX_RETRIES:
                    LOGGER.warning("[Polygon] Max retries (%d) exhausted — connection error", MAX_RETRIES)
                    raise last_error

                remaining = TOTAL_RETRY_BUDGET - (time.monotonic() - start_time)
                delay = min(delay * BACKOFF_MULTIPLIER, MAX_DELAY, max(remaining, 0))

                if delay < MIN_DELAY:
                    LOGGER.warning(
                        "[Polygon] Retry budget exhausted after %d attempt(s) — connection error (elapsed=%.1fs)",
                        attempts_made, time.monotonic() - start_time,
                    )
                    raise last_error

                LOGGER.warning(
                    "[Polygon] Retry %d/%d — connection error, sleeping %.1fs",
                    attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

        raise last_error

    async def fetch_ticker(self, ticker: str, timestamp: Optional[datetime] = None) -> TickerData:
        anchor = SessionClock.localize(timestamp or SessionClock.now())
        day = anchor.strftime("%Y-%m-%d")
        start = (anchor - timedelta(days=5)).strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{start}/{day}"
        data = await self._get_json_with_retry(
            url,
            {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key},
        )
        rows = data.get("results") or []
        if not rows:
            raise DataInsufficientError(f"No Polygon minute data for {ticker}")
        df = pd.DataFrame([{
            "Open": r.get("o"), "High": r.get("h"), "Low": r.get("l"),
            "Close": r.get("c"), "Volume": r.get("v"),
        } for r in rows], index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert(SessionClock._tz))
        df = df[df.index <= anchor]
        if df.empty:
            raise DataInsufficientError(f"No Polygon bars at anchor for {ticker}")
        regular_all = df[
            (df.index.time >= datetime.strptime("09:30", "%H:%M").time()) &
            (df.index.time < datetime.strptime("16:00", "%H:%M").time())
        ]
        prior = regular_all[regular_all.index.date < anchor.date()]
        if prior.empty:
            raise DataInsufficientError(f"No previous regular close for {ticker}")
        previous_close = float(prior.groupby(prior.index.date)["Close"].last().iloc[-1])
        today = df[df.index.date == anchor.date()]
        pre = today[
            (today.index.time >= datetime.strptime("04:00", "%H:%M").time()) &
            (today.index.time < datetime.strptime("09:30", "%H:%M").time())
        ]
        reg = today[
            (today.index.time >= datetime.strptime("09:30", "%H:%M").time()) &
            (today.index.time < datetime.strptime("16:00", "%H:%M").time())
        ]
        ah = today[
            (today.index.time >= datetime.strptime("16:00", "%H:%M").time()) &
            (today.index.time <= datetime.strptime("20:00", "%H:%M").time())
        ]
        current = float(df.iloc[-1]["Close"])
        gap = ((pre.iloc[0]["Open"] - previous_close) / previous_close * 100) if not pre.empty else \
              ((current - previous_close) / previous_close * 100)
        regvol = int(reg["Volume"].sum()) if not reg.empty else int(pre["Volume"].sum())
        avg = None
        if prior.groupby(prior.index.date)["Volume"].sum().shape[0] >= 3:
            avg = int(prior.groupby(prior.index.date)["Volume"].sum().tail(20).mean())
        return TickerData(
            ticker=ticker.upper(), timestamp=anchor, previous_close=round(previous_close, 2),
            premarket=self._snapshot(pre, MarketSession.PREMARKET),
            regular=self._snapshot(reg, MarketSession.REGULAR),
            after_hours=self._snapshot(ah, MarketSession.AFTER_HOURS),
            current_price=round(current, 2),
            change_percent=round((current - previous_close) / previous_close * 100, 2),
            gap_percent=round(gap, 2),
            avg_volume_20d=avg, provider_name=self.name, intraday_bars=df,
        )

    def _snapshot(self, df, kind):
        if df.empty:
            return SessionSnapshot(session_type=kind)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = int(df["Volume"].sum())
        vwap = float((tp * df["Volume"]).sum() / df["Volume"].sum()) if vol else None
        return SessionSnapshot(
            session_type=kind, high=float(df.High.max()), low=float(df.Low.min()),
            open=float(df.Open.iloc[0]), close=float(df.Close.iloc[-1]),
            volume=vol, vwap=vwap,
            timestamp_start=df.index[0].to_pydatetime(),
            timestamp_end=df.index[-1].to_pydatetime(),
        )

    async def health_check(self) -> bool:
        try:
            await self.fetch_ticker("SPY")
            return True
        except Exception:
            return False
