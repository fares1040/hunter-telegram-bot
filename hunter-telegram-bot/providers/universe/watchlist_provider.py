"""Watchlist universe provider — the user-curated, always-available universe source."""
from core.session_clock import MarketSession
from core.watchlist import WatchlistStore
from providers.universe.base_provider import MarketUniverseProvider, UniverseEntry, UniverseResult
from utils.logger import LOGGER


class WatchlistUniverseProvider(MarketUniverseProvider):
    name = "watchlist"

    def __init__(self, watchlist: WatchlistStore):
        self.watchlist = watchlist

    async def fetch_universe(self, session: MarketSession, limit: int = 25) -> UniverseResult:
        result = UniverseResult(source=self.name)
        for ticker in self.watchlist.list()[:limit]:
            # Metrics are intentionally None: enrichment happens downstream via
            # the normal market-data provider pipeline, never invented here.
            result.entries.append(UniverseEntry(symbol=ticker, source=self.name, reason="WATCHLIST"))
        return result

    async def health_check(self) -> bool:
        try:
            self.watchlist.list()
            return True
        except Exception as e:
            LOGGER.error(f"[{self.name}] health check failed: {e}")
            return False
