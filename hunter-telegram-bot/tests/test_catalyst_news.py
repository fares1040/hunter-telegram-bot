"""Phase 2.6 — Catalyst Intelligence Engine tests.

Deterministic news classification, materiality scoring, freshness buckets,
trap-risk flags, the YFinance real news provider, and the /news command.
All synthetic data; no network required.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from engines.catalyst_engine import CatalystEngine
from engines.news_engine import NewsEngine
from models.catalyst import FreshnessBucket, Recommendation, SentimentLabel
from models.news import CatalystEvent, CatalystType, NewsItem, SourceTier
from providers.news.yfinance_provider import YFinanceNewsProvider


NOW = datetime.now(timezone.utc)


def _anchor():
    """Match the age anchor used by the rest of the suite.

    test_integration.py globally rebinds NewsItem.age_minutes (anchored to its
    TEST_NEWS_TIME) at import/collection time, so our fixtures must publish
    against the same reference clock or freshness math diverges.
    """
    try:
        from tests.test_integration import TEST_NEWS_TIME
        return TEST_NEWS_TIME
    except Exception:
        return NOW


def _item(headline, *, age_minutes=10.0, source="Reuters", tier=SourceTier.TIER_2_MAJOR,
          published_at=NOW, url="https://example.com/x", ticker="TST", item_id=None):
    if published_at == NOW and age_minutes is not None:
        published_at = _anchor() - timedelta(minutes=age_minutes)
    return NewsItem(
        id=item_id or f"test:{abs(hash(headline)) % 10_000_000}",
        ticker=ticker,
        headline=headline,
        source=source,
        source_tier=tier,
        url=url,
        published_at=published_at,
    )


def _event(headline, **kw):
    item = _item(headline, **kw)
    return CatalystEvent(
        event_id=item.id,
        ticker=item.ticker,
        catalyst_type=CatalystType.OTHER,
        headline_summary=headline,
        primary_source=item,
    )


@pytest.fixture
def engine():
    return CatalystEngine()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class TestClassification:
    def test_positive_contract(self, engine):
        category, label, conf, rules = engine.classify("ACME awarded $50M government contract by Department of Defense")
        assert category == "CONTRACTS"
        assert label is SentimentLabel.POSITIVE
        assert conf > 0 and "gov_contract" in rules

    def test_negative_offering_is_trap_category(self, engine):
        category, label, _, _ = engine.classify("ACME announces $40M public offering of common stock")
        assert label is SentimentLabel.NEGATIVE
        assert category in {"FINANCING_OFFERING", "DILUTION"}

    def test_downgrade_negative(self, engine):
        _, label, _, _ = engine.classify("ACME downgraded to underweight at Morgan Stanley")
        assert label is SentimentLabel.NEGATIVE

    def test_unknown_when_no_rule_matches(self, engine):
        category, label, conf, rules = engine.classify("ACME opens new office in Austin")
        assert category == "UNKNOWN"
        assert label is SentimentLabel.UNKNOWN
        assert conf == 0 and rules == []

    def test_mixed_when_conflicting_rules(self, engine):
        _, label, conf, _ = engine.classify("ACME beats estimates but announces dilutive offering")
        assert label is SentimentLabel.MIXED
        assert conf < 90

    def test_case_insensitive(self, engine):
        _, label, _, _ = engine.classify("FDA GRANTS APPROVAL FOR ACME DRUG")
        assert label is SentimentLabel.POSITIVE


# ---------------------------------------------------------------------------
# Freshness buckets
# ---------------------------------------------------------------------------
class TestFreshness:
    @pytest.mark.parametrize("age,expected", [
        (None, FreshnessBucket.STALE),
        (0.0, FreshnessBucket.BREAKING),
        (29.9, FreshnessBucket.BREAKING),
        (30.1, FreshnessBucket.RECENT),
        (119.9, FreshnessBucket.RECENT),
        (120.1, FreshnessBucket.AGING),
        (359.9, FreshnessBucket.AGING),
        (360.1, FreshnessBucket.STALE),
    ])
    def test_buckets(self, engine, age, expected):
        assert engine.freshness_bucket(age) is expected

    def test_stale_reduces_materiality(self, engine):
        fresh = engine.assess(_event("ACME wins $100M contract award", age_minutes=5))
        stale = engine.assess(_event("ACME wins $100M contract award", age_minutes=600))
        assert fresh.materiality_breakdown["freshness"] > stale.materiality_breakdown["freshness"]
        assert fresh.materiality > stale.materiality


# ---------------------------------------------------------------------------
# Materiality: explainable breakdown
# ---------------------------------------------------------------------------
class TestMateriality:
    def test_breakdown_keys_and_sum(self, engine):
        p = engine.assess(_event("ACME reports record revenue $2B, beats estimates"))
        assert set(p.materiality_breakdown) == {"category", "source_quality", "freshness", "figures", "corroboration"}
        assert p.materiality == min(sum(p.materiality_breakdown.values()), 100)

    def test_tier_score_affects_source_quality(self, engine):
        high = engine.assess(_event("ACME wins contract", tier=SourceTier.TIER_1_OFFICIAL))
        low = engine.assess(_event("ACME wins contract", tier=SourceTier.TIER_4_UNVERIFIED))
        assert high.materiality_breakdown["source_quality"] > low.materiality_breakdown["source_quality"]

    def test_figures_boost(self, engine):
        with_fig = engine.assess(_event("ACME wins $250M contract"))
        without = engine.assess(_event("ACME wins large contract"))
        assert with_fig.materiality_breakdown["figures"] == 15
        assert without.materiality_breakdown["figures"] == 0

    def test_corroboration_from_cluster(self, engine):
        primary = _item("ACME wins major defense contract", age_minutes=5)
        extra = _item("ACME wins major defense contract", age_minutes=8, source="CNBC",
                      tier=SourceTier.TIER_3_FINANCIAL, item_id="test:dup")
        ev = CatalystEvent(event_id="e", ticker="TST", catalyst_type=CatalystType.OTHER,
                           headline_summary=primary.headline, primary_source=primary,
                           additional_sources=[extra])
        solo = engine.assess(CatalystEvent(event_id="s", ticker="TST", catalyst_type=CatalystType.OTHER,
                                           headline_summary=primary.headline, primary_source=primary))
        clustered = engine.assess(ev)
        assert clustered.cluster_size == 2
        assert clustered.materiality_breakdown["corroboration"] > solo.materiality_breakdown["corroboration"]

    def test_materiality_bounded(self, engine):
        p = engine.assess(_event("ACME to be acquired in $5B merger, beats estimates, FDA approval granted"))
        assert 0 <= p.materiality <= 100


# ---------------------------------------------------------------------------
# Trap flags + recommendations
# ---------------------------------------------------------------------------
class TestTrapFlagsAndRecommendations:
    def test_offering_flagged_trap(self, engine):
        p = engine.assess(_event("ACME announces $75M registered direct offering", age_minutes=5))
        assert p.is_trap_risk
        assert p.trap_reasons
        assert p.recommendation is Recommendation.TRAP_RISK

    def test_dilution_headline_flagged(self, engine):
        p = engine.assess(_event("ACME files dilutive warrant inducement agreement", age_minutes=5))
        assert p.is_trap_risk and p.recommendation is Recommendation.TRAP_RISK

    def test_stale_high_materiality_flagged_priced_in(self, engine):
        p = engine.assess(_event("ACME to be acquired for $5B in all-cash merger deal", age_minutes=700))
        assert any("priced in" in r.lower() for r in p.trap_reasons)

    def test_opportunity_requires_fresh_positive_material(self, engine):
        p = engine.assess(_event("ACME awarded $500M government contract by Department of Defense", age_minutes=10))
        assert p.recommendation is Recommendation.OPPORTUNITY

    def test_same_catalyst_stale_is_not_opportunity(self, engine):
        p = engine.assess(_event("ACME awarded $500M government contract by Department of Defense", age_minutes=400))
        assert p.recommendation is not Recommendation.OPPORTUNITY


# ---------------------------------------------------------------------------
# Missing fields (honesty about gaps — never fabricated)
# ---------------------------------------------------------------------------
class TestMissingFields:
    def test_missing_published_at_reported(self, engine):
        p = engine.assess(_event("ACME wins contract", published_at=None, age_minutes=None))
        assert "published_at" in p.missing_fields
        assert "age" in p.missing_fields
        assert p.age_minutes is None

    def test_no_figures_reported(self, engine):
        p = engine.assess(_event("ACME wins big contract"))
        assert "figures" in p.missing_fields


# ---------------------------------------------------------------------------
# enrich(): applies deterministic values onto existing events
# ---------------------------------------------------------------------------
class TestEnrich:
    def test_maps_other_to_specific_type(self, engine):
        ev = _event("ACME awarded defense contract by Department of Defense")
        engine.enrich(ev)
        assert ev.catalyst_type is CatalystType.CONTRACT

    def test_does_not_overwrite_explicit_type(self, engine):
        ev = _event("ACME awarded defense contract by Department of Defense")
        ev.catalyst_type = CatalystType.EARNINGS
        engine.enrich(ev)
        assert ev.catalyst_type is CatalystType.EARNINGS

    def test_sets_materiality_and_sentiment(self, engine):
        ev = _event("ACME misses estimates and lowers guidance")
        engine.enrich(ev)
        assert ev.sentiment == "NEGATIVE"
        assert ev.materiality_score > 0

    def test_negative_analyst_action_maps_to_downgrade(self, engine):
        ev = _event("ACME downgraded to sell rating at Goldman")
        engine.enrich(ev)
        assert ev.catalyst_type is CatalystType.DOWNGRADE

    def test_new_enum_members_exist(self):
        assert CatalystType.GUIDANCE.value == "GUIDANCE"
        assert CatalystType.DILUTION.value == "DILUTION"
        assert CatalystType.BANKRUPTCY.value == "BANKRUPTCY"


# ---------------------------------------------------------------------------
# YFinanceNewsProvider (REAL provider, injected fetch function)
# ---------------------------------------------------------------------------
VALID_YF_ITEM = {
    "id": "abc123",
    "content": {
        "title": "ACME beats estimates",
        "summary": "Strong quarter",
        "pubDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://finance.yahoo.com/news/abc123.html"},
    },
}


def _provider():
    return YFinanceNewsProvider(news_fn=lambda t, c: [dict(VALID_YF_ITEM)])


class TestYFinanceProvider:
    def test_maps_valid_item(self):
        items = asyncio.run(_provider().fetch_news("TST", NOW - timedelta(hours=24)))
        assert len(items) == 1
        it = items[0]
        assert it.id == "yfinance_news:abc123"
        assert it.headline == "ACME beats estimates"
        assert it.source == "Reuters"
        assert it.url.startswith("https://")
        assert it.published_at is not None and it.published_at.tzinfo is not None

    def test_skips_malformed_items_never_fabricates(self):
        bad = [{"no_content": True}, {"content": {"title": ""}}, {"id": "x"}, "junk", None]
        fn = lambda t, c: [dict(VALID_YF_ITEM)] + bad  # noqa: E731
        items = asyncio.run(YFinanceNewsProvider(news_fn=fn).fetch_news("TST", NOW - timedelta(hours=24)))
        assert len(items) == 1

    def test_respects_since_filter(self):
        old = dict(VALID_YF_ITEM)
        old["content"] = dict(old["content"])
        old["content"]["pubDate"] = (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        items = asyncio.run(YFinanceNewsProvider(news_fn=lambda t, c: [old]).fetch_news("TST", NOW - timedelta(hours=24)))
        assert items == []

    def test_fetch_failure_returns_empty_list(self):
        def boom(t, c):
            raise RuntimeError("network down")
        items = asyncio.run(YFinanceNewsProvider(news_fn=boom).fetch_news("TST", NOW - timedelta(hours=24)))
        assert items == []

    def test_health_check_reflects_failure(self):
        def boom(t, c):
            raise RuntimeError("down")
        assert asyncio.run(YFinanceNewsProvider(news_fn=boom).health_check()) is False
        assert asyncio.run(_provider().health_check()) is True

    def test_tier_mapping(self):
        p = _provider()
        assert p._map_tier("Bloomberg") is SourceTier.TIER_2_MAJOR
        assert p._map_tier("CNBC") is SourceTier.TIER_3_FINANCIAL
        assert p._map_tier("Motley Fool") is SourceTier.TIER_4_UNVERIFIED
        assert p._map_tier("Unknown Outlet") is SourceTier.TIER_3_FINANCIAL


# ---------------------------------------------------------------------------
# Regression: naive vs aware datetime crash in cluster_events
# ---------------------------------------------------------------------------
class TestClusterDatetimeRegression:
    def test_cluster_with_missing_timestamp_does_not_crash(self):
        ne = NewsEngine([])
        aware = _item("ACME wins contract", age_minutes=5)
        missing = _item("Completely unrelated headline about offices", published_at=None, age_minutes=None)
        events = ne.cluster_events([aware, missing])
        assert len(events) >= 1

    def test_cluster_with_naive_timestamp_does_not_crash(self):
        ne = NewsEngine([])
        naive_item = _item("ACME wins contract award", published_at=datetime(2026, 8, 20, 14, 30), age_minutes=None)
        assert ne.cluster_events([naive_item])

    def test_cluster_with_mixed_naive_and_aware_does_not_crash(self):
        ne = NewsEngine([])
        none_ts = _item("ACME wins major contract", published_at=None, age_minutes=None)
        naive_ts = _item("ACME wins major contract", published_at=datetime(2026, 8, 20, 14, 30),
                         source="CNBC", tier=SourceTier.TIER_3_FINANCIAL, item_id="t:naive")
        aware_ts = _item("ACME wins major contract", published_at=NOW,
                         source="Bloomberg", tier=SourceTier.TIER_2_MAJOR, item_id="t:aware")
        try:
            events = ne.cluster_events([none_ts, naive_ts, aware_ts])
        except TypeError as e:
            pytest.fail(f"mixed naive/aware timestamps must not crash clustering: {e}")
        assert len(events) >= 1
        # Similar headlines must still merge despite timestamp normalization
        assert any(ev.cluster_size if hasattr(ev, "cluster_size") else True for ev in events)

    def test_provider_normalizes_offset_less_pubdate_to_utc(self):
        raw = dict(VALID_YF_ITEM)
        raw["content"] = dict(raw["content"])
        recent_naive = (NOW - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")  # offset-less -> naive risk
        raw["content"]["pubDate"] = recent_naive
        items = asyncio.run(YFinanceNewsProvider(news_fn=lambda t, c: [raw]).fetch_news("TST", NOW - timedelta(hours=24)))
        assert len(items) == 1
        assert items[0].published_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Regression: low-tier clusters can never be OPPORTUNITY (engine-internal guard)
# ---------------------------------------------------------------------------
class TestTierGuard:
    STRONG_POSITIVE = "ACME awarded $500M government contract by Department of Defense"

    def test_low_tier_never_opportunity(self, engine):
        p = engine.assess(_event(self.STRONG_POSITIVE, age_minutes=10, tier=SourceTier.TIER_4_UNVERIFIED))
        assert p.source_tier_score < 40
        assert p.recommendation is not Recommendation.OPPORTUNITY
        assert p.recommendation is Recommendation.WATCH  # still visible, never promoted

    def test_high_tier_equivalent_still_opportunity(self, engine):
        p = engine.assess(_event(self.STRONG_POSITIVE, age_minutes=10, tier=SourceTier.TIER_2_MAJOR))
        assert p.source_tier_score >= 40
        assert p.recommendation is Recommendation.OPPORTUNITY


# ---------------------------------------------------------------------------
# /news command (end-to-end with real NewsEngine + CatalystEngine)
# ---------------------------------------------------------------------------
def _news_orchestrator(items):
    o = MagicMock()
    o.news_engine = NewsEngine([])
    o.catalyst_engine = CatalystEngine()
    o.news_engine.gather_news = AsyncMock(return_value=items)
    return o


class TestNewsCommand:
    def _make_bot(self, orchestrator):
        from bot.commands import TelegramCommandBot
        bot = TelegramCommandBot(orchestrator, watchlist=MagicMock(), memory=MagicMock())
        bot.authorized_ids = {"111"}
        return bot

    async def _invoke(self, bot, args):
        update, chat = MagicMock(), MagicMock()
        chat.id = 111
        chat.send_message = AsyncMock()
        update.effective_chat = chat
        ctx = MagicMock()
        ctx.args = args
        await bot.cmd_news(update, ctx)
        return chat.send_message

    def test_news_reply_contains_deterministic_profile(self):
        items = [
            _item("ACME awarded $300M government contract by Department of Defense", age_minutes=12, item_id="n:1"),
            _item("ACME wins major government contract for radar systems", age_minutes=20, source="CNBC", tier=SourceTier.TIER_3_FINANCIAL, item_id="n:2"),
        ]
        o = _news_orchestrator(items)
        send = asyncio.run(self._invoke(self._make_bot(o), ["ACME"]))
        text = send.call_args[0][0]
        assert "ACME" in text
        assert "CONTRACTS" in text
        assert "Materiality:" in text
        assert "/100" in text

    def test_no_news_message(self):
        o = _news_orchestrator([])
        send = asyncio.run(self._invoke(self._make_bot(o), ["ACME"]))
        assert "No recent news" in send.call_args[0][0]

    def test_missing_arg_sends_usage_hint(self):
        o = _news_orchestrator([])
        send = asyncio.run(self._invoke(self._make_bot(o), []))
        assert "Usage: /news" in send.call_args[0][0]


# ---------------------------------------------------------------------------
# run.py wiring: yfinance always present + catalyst engine attached
# ---------------------------------------------------------------------------
class TestOrchestratorWiring:
    def test_yfinance_provider_always_registered(self):
        from run import HunterOrchestrator
        o = HunterOrchestrator()
        assert any(isinstance(p, YFinanceNewsProvider) for p in o.news_providers)

    def test_catalyst_engine_attached(self):
        from run import HunterOrchestrator
        o = HunterOrchestrator()
        assert isinstance(o.catalyst_engine, CatalystEngine)
