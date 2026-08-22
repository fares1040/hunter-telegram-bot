"""YFinance screener universe provider — REAL live market discovery.

Source: Yahoo Finance screener API exposed through the installed yfinance package
(`yf.screen` + `EquityQuery`). Data is delayed 15-20 min and screener coverage of
extended-hours moves is not guaranteed; treat metrics as indicative.

KNOWN LIMITATIONS (documented, not worked around):
- OTC/Pink symbols are deliberately excluded (exchange filter NMS/NYQ) until a
  reliable data source for them exists.
- Yahoo's predefined queries carry a >= $2B intraday-market-cap floor; custom
  "hunter" queries below use lower thresholds to reach small caps.
- Polygon snapshot/gainer endpoints (real-time universe) are paid-tier; when a
  key with snapshot access is available a PolygonUniverseProvider can slot in
  behind the same MarketUniverseProvider interface without touching this engine.
"""
import asyncio
from typing import Callable, List, Optional

from config.settings import SETTINGS
from core.session_clock import MarketSession
from providers.universe.base_provider import MarketUniverseProvider, UniverseEntry, UniverseResult
from utils.logger import LOGGER


def _default_screen(screen_query, size):
    from yfinance import screen as yf_screen
    return yf_screen(screen_query, size=size)


class YFinanceScreenerUniverseProvider(MarketUniverseProvider):
    name = "yfinance_screener"

    def __init__(
        self,
        screen_fn: Optional[Callable] = None,
        min_abs_change: float = SETTINGS.discovery_min_abs_change,
        max_price: float = SETTINGS.max_watchlist_price,
    ):
        self.min_abs_change = min_abs_change
        self.max_price = max_price
        try:
            from yfinance.screener.query import EquityQuery  # noqa: F401
            self._equity_query_factory = EquityQuery
            self.available = True
        except Exception:
            self._equity_query_factory = None
            self.available = False
        if screen_fn is not None:
            self._screen = screen_fn
            self.available = True
        else:
            self._screen = _default_screen

    async def fetch_universe(self, session: MarketSession, limit: int = 25) -> UniverseResult:
        result = UniverseResult(source=self.name)
        if not self.available:
            result.success = False
            result.warnings.append("yfinance screener API unavailable in installed version")
            return result

        per_query = max(1, min(limit, SETTINGS.discovery_max_candidates_per_source))
        queries = self._session_queries(session)
        loop = asyncio.get_event_loop()
        for label, screen_query in queries:
            try:
                payload = await asyncio.to_thread(self._screen, screen_query, per_query)
                quotes = self._extract_quotes(payload)
                mapped = 0
                seen_in_payload = set()
                for q in quotes:
                    entry = self._map_quote(q, label)
                    if entry is None:
                        continue
                    if entry.symbol in seen_in_payload:
                        continue
                    seen_in_payload.add(entry.symbol)
                    result.entries.append(entry)
                    mapped += 1
                if mapped == 0:
                    result.warnings.append(f"{label}: no usable quotes returned")
            except Exception as e:
                result.warnings.append(f"{label} failed: {type(e).__name__}")
                LOGGER.warning(f"[{self.name}] {label} failed: {e}")
        result.success = bool(result.entries)
        if not result.entries and result.warnings:
            result.success = False
        return result

    def _session_queries(self, session: MarketSession) -> List[tuple]:
        eq = self._equity_query_factory

        def hunter(direction: str):
            # Custom low-threshold movers across NASDAQ/NYSE. OTC excluded on purpose.
            pct = ("percentchange", direction, self.min_abs_change)
            return eq("and", [
                eq("is-in", ["exchange", "NMS", "NYQ"]),
                eq("eq", ["region", "us"]),
                eq("gte", ["intradayprice", 1.0]),
                eq("lte", ["intradayprice", float(self.max_price)]),
                eq(pct[1], [pct[0], pct[2]]),
                eq("gt", ["dayvolume", 100000]),
            ])

        plans = {
            MarketSession.PREMARKET: [("HUNTER_GAINERS", hunter("gt")), ("HUNTER_LOSERS", hunter("lt"))],
            MarketSession.REGULAR: [
                ("HUNTER_GAINERS", hunter("gt")),
                ("HUNTER_LOSERS", hunter("lt")),
                ("MOST_ACTIVES", "most_actives"),
                ("SMALL_CAP_GAINERS", "small_cap_gainers"),
            ],
            MarketSession.AFTER_HOURS: [("HUNTER_GAINERS", hunter("gt")), ("HUNTER_LOSERS", hunter("lt"))],
            MarketSession.CLOSED: [
                ("DAY_GAINERS", "day_gainers"),
                ("DAY_LOSERS", "day_losers"),
                ("MOST_ACTIVES", "most_actives"),
            ],
        }
        built = []
        for label, spec in plans.get(session, plans[MarketSession.CLOSED]):
            if isinstance(spec, str):
                built.append((label, spec))
            else:
                built.append((label, spec))
        return built

    @staticmethod
    def _extract_quotes(payload) -> List[dict]:
        if isinstance(payload, dict):
            quotes = payload.get("quotes")
            if isinstance(quotes, list):
                return [q for q in quotes if isinstance(q, dict)]
        if isinstance(payload, list):
            return [q for q in payload if isinstance(q, dict)]
        return []

    def _map_quote(self, q: dict, reason: str) -> Optional[UniverseEntry]:
        symbol = (q.get("symbol") or q.get("ticker") or "").strip().upper()
        if not symbol:
            return None
        price = self._first_float(q, ["regularMarketPrice", "intradayprice", "postMarketPrice", "preMarketPrice"])
        change = self._first_float(q, ["regularMarketChangePercent", "percentchange"])
        volume = self._first_float(q, ["regularMarketVolume", "dayvolume"])
        mcap = self._first_float(q, ["marketCap", "intradaymarketcap"])
        return UniverseEntry(
            symbol=symbol,
            source=self.name,
            reason=reason,
            price=price,
            change_percent=change,
            volume=int(volume) if volume is not None else None,
            market_cap=mcap,
        )

    @staticmethod
    def _first_float(q: dict, keys: List[str]) -> Optional[float]:
        for k in keys:
            v = q.get(k)
            if isinstance(v, dict):
                v = v.get("raw") or v.get("value")
            if isinstance(v, (int, float)):
                return float(v)
        return None

    async def health_check(self) -> bool:
        if not self.available:
            return False
        try:
            await asyncio.to_thread(self._screen, "most_actives", 1)
            return True
        except Exception as e:
            LOGGER.warning(f"[{self.name}] health check failed: {e}")
            return False
