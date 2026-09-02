"""Market Discovery Engine.

Pipeline:
  universe providers → normalization → deduplication → basic filters
  → transparent discovery score → ranked CandidatePool

The engine never sends alerts and never fabricates data. Missing metrics stay
None, are listed in each candidate's missing_fields, and simply contribute 0 to
the score. The pool feeds the existing Hunter engines (CandidateGate → … →
DecisionEngine); it does not duplicate their logic.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config.settings import SETTINGS
from core.session_clock import SessionClock, MarketSession
from core.watchlist import normalize_ticker
from models.discovery import CandidatePool, DiscoveryCandidate, merge_entries_to_candidates
from providers.universe.base_provider import MarketUniverseProvider
from utils.logger import LOGGER

SCORE_WEIGHTS = {
    "move": 40,
    "volume": 25,
    "size_fit": 20,
    "corroboration": 10,
    "watchlist": 5,
}


class DiscoveryEngine:
    def __init__(
        self,
        universe_providers: List[MarketUniverseProvider],
        pool_size: int = SETTINGS.discovery_pool_size,
        cache_ttl: float = SETTINGS.discovery_cache_ttl,
        clock=SessionClock,
    ):
        if not universe_providers:
            raise ValueError("DiscoveryEngine requires at least one universe provider")
        self.universe_providers = universe_providers
        self.pool_size = pool_size
        self.cache_ttl = timedelta(seconds=cache_ttl)
        self.clock = clock
        self._cache: Optional[tuple] = None
        # Created lazily inside refresh(): on Python 3.9 an asyncio.Lock bound at
        # import/construction time attaches to whichever loop exists then.
        self._refresh_lock = None

    async def refresh(self, session: Optional[MarketSession] = None, force: bool = False) -> CandidatePool:
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            if not force and self._cache is not None:
                generated_at, cached = self._cache
                if datetime.now(timezone.utc) - generated_at < self.cache_ttl:
                    return cached
            session = session or self.clock.get_session()
            entries = await self._collect_entries(session)
            pool = self._build_pool(session, entries)
            self._cache = (pool.generated_at, pool)
            return pool

    async def _collect_entries(self, session: MarketSession) -> tuple:
        semaphore = asyncio.Semaphore(3)

        async def guarded(provider):
            async with semaphore:
                try:
                    return await provider.fetch_universe(session, limit=self.pool_size * 3)
                except Exception as e:
                    LOGGER.error(f"[Discovery] provider {provider.name} failed: {e}")
                    return None

        results = await asyncio.gather(*(guarded(p) for p in self.universe_providers))
        return results

    def _build_pool(self, session: MarketSession, results) -> CandidatePool:
        warnings: List[str] = []
        raw_entries = []
        invalid = 0
        for res in results:
            if res is None:
                continue
            warnings.extend(res.warnings)
            if not res.success:
                warnings.append(f"{res.source}: unavailable")
            for entry in res.entries:
                try:
                    entry.symbol = normalize_ticker(entry.symbol)
                except ValueError:
                    invalid += 1
                    continue
                raw_entries.append(entry)

        candidates = merge_entries_to_candidates(raw_entries)
        duplicate_count = len(raw_entries) - len(candidates)

        scored = []
        for cand in candidates:
            self._score(cand)
            scored.append(cand)

        scored.sort(key=lambda c: (-c.discovery_score, c.symbol))
        pool = CandidatePool(
            session=session,
            candidates=scored[: self.pool_size],
            raw_count=len(raw_entries),
            invalid_count=invalid,
            duplicate_count=max(0, duplicate_count),
            warnings=warnings,
        )
        LOGGER.info(
            f"[Discovery] session={session.value} raw={pool.raw_count} unique={len(scored)} "
            f"invalid={invalid} dupes={duplicate_count} pool={len(pool.candidates)} "
            f"warnings={len(warnings)}"
        )
        return pool

    def _score(self, cand: DiscoveryCandidate) -> None:
        breakdown = {}
        missing = []
        from_watchlist = "watchlist" in cand.sources

        change = cand.change_percent
        if change is None:
            missing.append("change_percent")
            breakdown["move"] = 0
        else:
            magnitude = abs(change)
            if magnitude >= 20:
                breakdown["move"] = SCORE_WEIGHTS["move"]
            elif magnitude >= 10:
                breakdown["move"] = 32
            elif magnitude >= 5:
                breakdown["move"] = 24
            elif magnitude >= 2:
                breakdown["move"] = 14
            else:
                breakdown["move"] = 6

        volume = cand.volume
        if volume is None:
            missing.append("volume")
            breakdown["volume"] = 0
        elif volume >= 10_000_000:
            breakdown["volume"] = SCORE_WEIGHTS["volume"]
        elif volume >= 5_000_000:
            breakdown["volume"] = 20
        elif volume >= 1_000_000:
            breakdown["volume"] = 14
        elif volume >= 300_000:
            breakdown["volume"] = 8
        else:
            breakdown["volume"] = 3

        mcap = cand.market_cap
        if mcap is None:
            missing.append("market_cap")
            breakdown["size_fit"] = 0
        elif mcap <= 300_000_000:
            breakdown["size_fit"] = SCORE_WEIGHTS["size_fit"]
        elif mcap <= 2_000_000_000:
            breakdown["size_fit"] = 15
        elif mcap <= 10_000_000_000:
            breakdown["size_fit"] = 10
        elif mcap <= 50_000_000_000:
            breakdown["size_fit"] = 5
        else:
            breakdown["size_fit"] = 2

        extra_sources = max(0, len(set(cand.sources)) - 1)
        breakdown["corroboration"] = min(extra_sources * 5, SCORE_WEIGHTS["corroboration"])
        breakdown["watchlist"] = SCORE_WEIGHTS["watchlist"] if from_watchlist else 0
        # Adaptive Hunt (Stage 6): bounded additive adjustment from Track Record, cached per-scan
        breakdown["adaptive"] = 0
        try:
            from core.adaptive import adjustment_for
            from core.memory import SignalMemory
            # Use default memory path; if DB missing or insufficient history, adjustment stays 0
            mem = SignalMemory()
            # Discovery has no Swing setup yet, so we check UNKNOWN/setup-agnostic: overall not per-candidate setup
            # For determinism, use UNKNOWN bucket which covers no-setup candidates
            adj = adjustment_for("UNKNOWN", mem)
            # Also try candidate-specific if it had setup hint (future)
            breakdown["adaptive"] = max(-5, min(5, adj))
        except Exception:
            breakdown["adaptive"] = 0

        cand.score_breakdown = breakdown
        cand.missing_fields = sorted(missing)
        cand.discovery_score = int(sum(breakdown.values()))
        # Quality prioritization: deprioritize noisy micro-caps without hard rejection
        # price and volume are real fields; low price alone does not force IGNORE
        price = cand.price
        if price is not None and volume is not None:
            if price < 2.0 and volume < 500_000:
                cand.discovery_score = max(0, cand.discovery_score - 10)
            elif price < 5.0 and volume < 300_000:
                cand.discovery_score = max(0, cand.discovery_score - 8)
        # apply adaptive bounded adjustment additively
        if breakdown.get("adaptive"):
            cand.discovery_score = max(0, min(100, cand.discovery_score + breakdown["adaptive"]))
