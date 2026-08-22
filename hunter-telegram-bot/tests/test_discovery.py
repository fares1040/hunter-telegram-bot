"""Phase 2.5 — Market Discovery Engine tests (fully offline; no network calls)."""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from core.session_clock import MarketSession
from core.scheduler import ScanScheduler
from core.watchlist import WatchlistStore
from engines.discovery import DiscoveryEngine, SCORE_WEIGHTS
from models.discovery import CandidatePool, DiscoveryCandidate
from providers.universe.base_provider import MarketUniverseProvider, UniverseEntry, UniverseResult
from providers.universe.watchlist_provider import WatchlistUniverseProvider
from providers.universe.yfinance_screener_provider import YFinanceScreenerUniverseProvider


def _run(coro):
    return asyncio.run(coro)


class FakeUniverseProvider(MarketUniverseProvider):
    def __init__(self, name="fake", entries=None, exc=None, success=True):
        self.name = name
        self.entries = entries or []
        self.exc = exc
        self.success = success
        self.calls = 0

    async def fetch_universe(self, session, limit=25):
        self.calls += 1
        if self.exc:
            raise self.exc
        return UniverseResult(source=self.name, success=self.success, entries=list(self.entries))

    async def health_check(self):
        return True


def _entry(symbol, source="src", reason="MOVER", **kw):
    return UniverseEntry(symbol=symbol, source=source, reason=reason, **kw)


class TestDiscoveryNormalizationAndDedup:
    def _engine(self, providers, pool_size=10):
        return DiscoveryEngine(providers, pool_size=pool_size, cache_ttl=0)

    def test_symbols_normalized_uppercase(self):
        p = FakeUniverseProvider(entries=[_entry("aapl"), _entry("Nvda")])
        pool = _run(self._engine([p]).refresh())
        assert set(pool.symbols()) == {"AAPL", "NVDA"}

    def test_dollar_prefix_stripped(self):
        p = FakeUniverseProvider(entries=[_entry("$tsla")])
        pool = _run(self._engine([p]).refresh())
        assert pool.symbols() == ["TSLA"]

    def test_invalid_symbols_dropped_and_counted(self):
        p = FakeUniverseProvider(entries=[_entry("GOOD"), _entry("BAD!1"), _entry("WAYTOOLONG"), _entry("")])
        pool = _run(self._engine([p]).refresh())
        assert pool.symbols() == ["GOOD"]
        assert pool.invalid_count == 3

    def test_cross_provider_dedup_merges_sources_and_fills_missing_metrics(self):
        a = FakeUniverseProvider("alpha", [_entry("XYZ", source="alpha", reason="GAINER", change_percent=8.5)])
        b = FakeUniverseProvider("beta", [_entry("$xyz", source="beta", reason="ACTIVE", volume=4_000_000)])
        pool = _run(self._engine([a, b]).refresh())
        assert len(pool.candidates) == 1
        c = pool.candidates[0]
        assert c.symbol == "XYZ"
        assert set(c.sources) == {"alpha", "beta"}
        assert set(c.reasons) == {"GAINER", "ACTIVE"}
        assert c.change_percent == 8.5
        assert c.volume == 4_000_000
        assert pool.duplicate_count == 1

    def test_within_payload_duplicate_symbol_single_entry(self):
        p = FakeUniverseProvider(entries=[
            _entry("DUP", source="s1", reason="A"),
            _entry("dup", source="s1", reason="B"),
            _entry("DUP", source="s1", reason="C"),
        ])
        pool = _run(self._engine([p]).refresh())
        assert pool.symbols() == ["DUP"]
        assert len(pool.candidates[0].reasons) == 3


class TestDiscoveryFiltersAndPool:
    def test_empty_universe_produces_empty_pool(self):
        p = FakeUniverseProvider()
        pool = _run(DiscoveryEngine([p], cache_ttl=0).refresh())
        assert pool.candidates == [] and pool.raw_count == 0

    def test_pool_size_cap_and_ranking(self):
        entries = []
        for i in range(15):
            entries.append(_entry(f"{chr(65 + i)}SYM", source="s", reason="R", change_percent=float(i)))
        pool = _run(DiscoveryEngine([FakeUniverseProvider(entries=entries)], pool_size=5, cache_ttl=0).refresh())
        assert len(pool.candidates) == 5
        scores = [c.discovery_score for c in pool.candidates]
        assert scores == sorted(scores, reverse=True)
        # change>=10 ties at the same move tier; deterministic tie-break is alphabetical
        assert pool.candidates[0].symbol == "KSYM"

    def test_engine_requires_provider(self):
        with pytest.raises(ValueError):
            DiscoveryEngine([])

    def test_session_label_recorded(self):
        pool = _run(DiscoveryEngine([FakeUniverseProvider()], cache_ttl=0).refresh(session=MarketSession.REGULAR))
        assert pool.session == MarketSession.REGULAR


class TestDiscoveryScoring:
    def _score_one(self, **kw):
        cand = DiscoveryCandidate(symbol="X", **kw)
        DiscoveryEngine([FakeUniverseProvider()], cache_ttl=0)._score(cand)
        return cand

    def test_full_metrics_scored_with_breakdown(self):
        cand = self._score_one(change_percent=22.0, volume=12_000_000, market_cap=150_000_000)
        b = cand.score_breakdown
        assert b["move"] == SCORE_WEIGHTS["move"]
        assert b["volume"] == SCORE_WEIGHTS["volume"]
        assert b["size_fit"] == SCORE_WEIGHTS["size_fit"]
        assert "change_percent" not in cand.missing_fields
        assert cand.discovery_score >= 85

    def test_missing_metrics_are_neutral_not_fake(self):
        cand = self._score_one()
        assert cand.missing_fields == ["change_percent", "market_cap", "volume"]
        b = cand.score_breakdown
        assert b["move"] == 0 and b["volume"] == 0 and b["size_fit"] == 0
        assert cand.discovery_score == sum(b.values())

    def test_negative_moves_score_like_positive(self):
        up = self._score_one(change_percent=-18.0)
        down = self._score_one(change_percent=18.0)
        assert up.score_breakdown["move"] == down.score_breakdown["move"]

    def test_small_move_low_volume_scores_modestly(self):
        cand = self._score_one(change_percent=1.0, volume=100_000, market_cap=80_000_000_000)
        assert cand.discovery_score < 20

    def test_corroboration_bonus_capped(self):
        one = self._score_one(sources=["a"], change_percent=6.0)
        many = self._score_one(sources=["a", "b", "c", "d"], change_percent=6.0)
        assert many.score_breakdown["corroboration"] == SCORE_WEIGHTS["corroboration"]
        assert many.discovery_score - one.discovery_score == SCORE_WEIGHTS["corroboration"]

    def test_watchlist_bonus_applies_only_to_watchlist_source(self):
        wl = self._score_one(sources=["watchlist"], change_percent=6.0)
        sc = self._score_one(sources=["yfinance_screener"], change_percent=6.0)
        assert wl.score_breakdown["watchlist"] == SCORE_WEIGHTS["watchlist"]
        assert sc.score_breakdown["watchlist"] == 0


class TestDiscoveryResilience:
    def test_provider_failure_keeps_other_results(self):
        bad = FakeUniverseProvider("bad", exc=RuntimeError("network down"))
        good = FakeUniverseProvider("good", entries=[_entry("OKAY", source="good")])
        pool = _run(DiscoveryEngine([bad, good], cache_ttl=0).refresh())
        assert pool.symbols() == ["OKAY"]

    def test_all_failures_yield_empty_pool_no_crash(self):
        bad = FakeUniverseProvider("bad", exc=RuntimeError("down"))
        pool = _run(DiscoveryEngine([bad], cache_ttl=0).refresh())
        assert pool.candidates == []

    def test_unsuccessful_result_flags_warning(self):
        p = FakeUniverseProvider(success=False)
        pool = _run(DiscoveryEngine([p], cache_ttl=0).refresh())
        assert any("unavailable" in w for w in pool.warnings)

    def test_partial_data_candidate_survives_into_pool(self):
        p = FakeUniverseProvider(entries=[_entry("PART", source="s")])  # no metrics at all
        pool = _run(DiscoveryEngine([p], cache_ttl=0).refresh())
        assert pool.symbols() == ["PART"]
        assert pool.candidates[0].missing_fields


class TestDiscoveryCacheTTL:
    def test_cached_refresh_avoids_second_provider_call(self):
        p = FakeUniverseProvider(entries=[_entry("A")])
        eng = DiscoveryEngine([p], cache_ttl=300)
        _run(eng.refresh())
        _run(eng.refresh())
        assert p.calls == 1

    def test_force_bypasses_cache(self):
        p = FakeUniverseProvider(entries=[_entry("A")])
        eng = DiscoveryEngine([p], cache_ttl=300)
        _run(eng.refresh())
        _run(eng.refresh(force=True))
        assert p.calls == 2

    def test_concurrent_refreshes_share_single_pass(self):
        class SlowProvider(FakeUniverseProvider):
            async def fetch_universe(self, session, limit=25):
                await asyncio.sleep(0.05)
                return await super().fetch_universe(session, limit)

        p = SlowProvider(entries=[_entry("A")])
        eng = DiscoveryEngine([p], cache_ttl=300)

        async def scenario():
            return await asyncio.gather(eng.refresh(), eng.refresh(), eng.refresh())
        pools = _run(scenario())
        assert p.calls == 1
        assert all(isinstance(p_, CandidatePool) for p_ in pools)


class TestWatchlistUniverseProvider:
    def test_returns_stored_symbols_without_invented_metrics(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"))
        e = WatchlistUniverseProvider(wl)
        res = _run(e.fetch_universe(MarketSession.REGULAR))
        assert res.success and [x.symbol for x in res.entries] == ["AAPL", "NVDA", "TSLA"]
        for entry in res.entries:
            assert entry.reason == "WATCHLIST"
            assert entry.price is None and entry.volume is None

    def test_limit_respected(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"))
        res = _run(WatchlistUniverseProvider(wl).fetch_universe(MarketSession.REGULAR, limit=2))
        assert len(res.entries) == 2


class TestYFinanceScreenerProvider:
    def _provider(self, screen_fn=None, min_abs_change=3.0):
        return YFinanceScreenerUniverseProvider(screen_fn=screen_fn, min_abs_change=min_abs_change)

    @staticmethod
    def _quote(**overrides):
        q = {
            "symbol": "MOV",
            "regularMarketPrice": 9.99,
            "regularMarketChangePercent": 7.4,
            "regularMarketVolume": 2_500_000,
            "intradaymarketcap": 450_000_000,
        }
        q.update(overrides)
        return q

    def test_capability_off_returns_failure_result(self):
        prov = self._provider(screen_fn=lambda *a, **k: {})
        prov.available = False
        res = _run(prov.fetch_universe(MarketSession.REGULAR))
        assert not res.success and res.entries == []
        assert any("unavailable" in w for w in res.warnings)

    def test_maps_real_quote_fields_defensively(self):
        payload = {"quotes": [self._quote()]}
        prov = self._provider(screen_fn=lambda q, size: payload)
        res = _run(prov.fetch_universe(MarketSession.REGULAR))
        assert res.success
        assert {e.symbol for e in res.entries} == {"MOV"}
        e = res.entries[0]
        assert e.price == 9.99 and e.change_percent == 7.4
        assert e.volume == 2_500_000 and e.market_cap == 450_000_000

    def test_missing_fields_map_to_none_never_zero(self):
        payloads = iter([
            {"quotes": [{"symbol": "BARE"}]},
            {"quotes": [{"symbol": "BARE"}]},
            {"quotes": [{"symbol": "BARE"}]},
            {"quotes": [{"symbol": "BARE"}]},
        ])
        prov = self._provider(screen_fn=lambda q, size: next(payloads))
        res = _run(prov.fetch_universe(MarketSession.REGULAR))
        e = res.entries[0]
        assert e.price is None and e.change_percent is None and e.volume is None

    def test_raw_dict_values_unwrapped(self):
        payload = {"quotes": [{"symbol": "RAW", "regularMarketPrice": {"raw": 5.5}, "regularMarketChangePercent": {"raw": -4.2}}]}
        prov = self._provider(screen_fn=lambda q, size: payload)
        res = _run(prov.fetch_universe(MarketSession.AFTER_HOURS))
        e = res.entries[0]
        assert e.price == 5.5 and e.change_percent == -4.2

    def test_symbolless_quotes_skipped(self):
        payload = {"quotes": [{"regularMarketPrice": 1.0}, {}, {"symbol": ""}]}
        prov = self._provider(screen_fn=lambda q, size: payload)
        res = _run(prov.fetch_universe(MarketSession.PREMARKET))
        assert res.entries == []

    def test_cross_query_duplicate_observation_merged_by_engine(self):
        # Same symbol observed by two screener queries (e.g. gainer + most active):
        # the provider emits one observation per query; the engine merges reasons.
        payload = {"quotes": [self._quote()]}
        prov = self._provider(screen_fn=lambda q, size: payload)
        res = _run(prov.fetch_universe(MarketSession.PREMARKET))
        assert [e.reason for e in res.entries] == ["HUNTER_GAINERS", "HUNTER_LOSERS"]

        eng = DiscoveryEngine([prov], cache_ttl=0)
        pool = _run(eng.refresh(session=MarketSession.PREMARKET))
        assert pool.symbols() == ["MOV"]
        assert set(pool.candidates[0].reasons) == {"HUNTER_GAINERS", "HUNTER_LOSERS"}

    def test_query_failure_records_warning_but_other_queries_continue(self):
        calls = {"n": 0}

        def flaky(q, size):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("rate limited")
            return {"quotes": [self._quote(symbol=f"S{calls['n']}")]}

        prov = self._provider(screen_fn=flaky)
        res = _run(prov.fetch_universe(MarketSession.CLOSED))  # 3 planned queries
        assert calls["n"] == 3
        assert {e.symbol for e in res.entries} == {"S2", "S3"}
        assert any("failed" in w for w in res.warnings)

    def test_list_payload_supported(self):
        prov = self._provider(screen_fn=lambda q, size: [self._quote(symbol="LST")])
        res = _run(prov.fetch_universe(MarketSession.PREMARKET))
        assert res.entries[0].symbol == "LST"

    def test_session_aware_query_plan_shape(self):
        seen = []

        def recorder(q, size):
            seen.append(q)
            return {"quotes": []}
        prov = self._provider(screen_fn=recorder)
        _run(prov.fetch_universe(MarketSession.REGULAR))
        regular_calls = len(seen)
        seen.clear()
        _run(prov.fetch_universe(MarketSession.PREMARKET))
        premarket_calls = len(seen)
        seen.clear()
        _run(prov.fetch_universe(MarketSession.CLOSED))
        closed_queries = list(seen)
        assert regular_calls == 4  # gainers, losers, most_actives, small_cap_gainers
        assert premarket_calls == 2  # hunter gainers + losers
        assert closed_queries == ["day_gainers", "day_losers", "most_actives"]

    def test_custom_hunter_query_is_equity_query_object(self):
        seen = []

        def recorder(q, size):
            seen.append(q)
            return {"quotes": []}
        prov = self._provider(screen_fn=recorder)
        _run(prov.fetch_universe(MarketSession.PREMARKET))
        assert type(seen[0]).__name__ == "EquityQuery"
        assert isinstance(seen[0], object)

    def test_per_source_limit_bounds_each_screen_call(self):
        sizes = []

        def recorder(q, size):
            sizes.append(size)
            return {"quotes": []}
        prov = self._provider(screen_fn=recorder)
        _run(prov.fetch_universe(MarketSession.REGULAR, limit=50))
        assert all(s <= 25 for s in sizes)


class TestSchedulerDiscoveryIntegration:
    def _scheduler(self, tmp_path, discovery_engine=None, batch=15):
        orch = MagicMock()
        orch.process_ticker = AsyncMock()
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        wl.add("WLIST")
        return ScanScheduler(orch, wl, ticker_pause=0, discovery_engine=discovery_engine, max_scan_batch=batch), wl

    def test_scan_targets_merge_watchlist_then_discovered(self, tmp_path):
        pool = CandidatePool(session=MarketSession.REGULAR, candidates=[
            DiscoveryCandidate(symbol="DISC1"),
            DiscoveryCandidate(symbol="DISC2"),
        ])
        eng = MagicMock()
        eng.refresh = AsyncMock(return_value=pool)
        sched, wl = self._scheduler(tmp_path, eng)
        targets = _run(sched._scan_targets())
        assert targets[0] == "WLIST"
        assert "DISC1" in targets and "DISC2" in targets
        assert "discovered=2" in sched.last_pool_summary

    def test_discovery_symbols_do_not_duplicate_watchlist(self, tmp_path):
        pool = CandidatePool(session=MarketSession.REGULAR, candidates=[DiscoveryCandidate(symbol="WLIST")])
        eng = MagicMock()
        eng.refresh = AsyncMock(return_value=pool)
        sched, _ = self._scheduler(tmp_path, eng)
        targets = _run(sched._scan_targets())
        assert targets.count("WLIST") == 1

    def test_batch_cap_respected(self, tmp_path):
        pool = CandidatePool(session=MarketSession.REGULAR, candidates=[DiscoveryCandidate(symbol=f"D{i}") for i in range(10)])
        eng = MagicMock()
        eng.refresh = AsyncMock(return_value=pool)
        sched, _ = self._scheduler(tmp_path, eng, batch=5)
        targets = _run(sched._scan_targets())
        assert len(targets) == 5

    def test_discovery_failure_falls_back_to_watchlist(self, tmp_path):
        eng = MagicMock()
        eng.refresh = AsyncMock(side_effect=RuntimeError("boom"))
        sched, wl = self._scheduler(tmp_path, eng)
        targets = _run(sched._scan_targets())
        assert targets == ["WLIST"]
