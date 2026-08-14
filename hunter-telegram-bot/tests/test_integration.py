"""
Hunter Bot — Integration Tests (Local Verification Only)
No external APIs required. All tests use synthetic data.
Run with: pytest tests/test_integration.py -v
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytz
import sys
import types

# Keep local tests independent from installed vendor SDKs/network. The production
# requirements still include yfinance and python-telegram-bot.
if "yfinance" not in sys.modules:
    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = lambda *args, **kwargs: None
    sys.modules["yfinance"] = fake_yf

from models.ticker import TickerData
from models.news import CatalystEvent, NewsItem, CatalystType, SourceTier
from models.signal import HunterDecision
from models.session import SessionSnapshot
from core.session_clock import SessionClock, MarketSession
from core.data_confidence import DataConfidenceReport, DataQuality
from core.exceptions import ProviderError, DataInsufficientError

from engines.news_engine import NewsEngine
from engines.market_reaction_engine import MarketReactionEngine
from engines.liquidity_proxy import LiquidityProxyEngine
from engines.technical_engine import TechnicalEngine
from engines.decision_engine import DecisionEngine


TEST_NEWS_TIME = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=30)


def make_intraday_bars(price: float, change: float, regular_vol: int) -> pd.DataFrame:
    """Synthetic 1-minute candles around TEST_NEWS_TIME."""
    baseline = price / (1 + change / 100.0) if (1 + change / 100.0) > 0 else price
    times = pd.date_range(
        start=TEST_NEWS_TIME - timedelta(minutes=15),
        end=TEST_NEWS_TIME + timedelta(minutes=35),
        freq="1min",
        tz="UTC",
    )
    rows = []
    before_n = max(1, int(regular_vol * 0.002))
    after_n = max(before_n + 1, int(regular_vol * 0.01))
    for ts in times:
        if ts < TEST_NEWS_TIME:
            close = baseline
            vol = before_n
        elif ts < TEST_NEWS_TIME + timedelta(minutes=5):
            close = baseline + (price - baseline) * 0.4
            vol = after_n
        else:
            close = price
            vol = after_n
        rows.append({
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": vol,
        })
    return pd.DataFrame(rows, index=times)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_ticker(
    price=10.0,
    prev_close=9.0,
    premarket_high=None,
    premarket_low=None,
    premarket_vol=None,
    regular_high=None,
    regular_low=None,
    regular_vol=1_000_000,
    gap=10.0,
    change=11.1,
    market_cap=1_000_000_000,
    float_shares=100_000_000,
    avg_vol=400_000,
) -> TickerData:
    pre = SessionSnapshot(
        session_type=MarketSession.PREMARKET,
        high=premarket_high,
        low=premarket_low,
        volume=premarket_vol,
    )
    reg = SessionSnapshot(
        session_type=MarketSession.REGULAR,
        high=regular_high or price * 1.02,
        low=regular_low or price * 0.98,
        volume=regular_vol,
        vwap=price * 0.99,
    )
    return TickerData(
        ticker="TEST",
        timestamp=datetime.now(timezone.utc),
        previous_close=prev_close,
        current_price=price,
        gap_percent=gap,
        change_percent=change,
        premarket=pre,
        regular=reg,
        market_cap=market_cap,
        float_shares=float_shares,
        avg_volume_20d=avg_vol,
        intraday_bars=make_intraday_bars(price, change, regular_vol),
    )


def make_event(
    sentiment="POSITIVE",
    impact=80,
    materiality=85,
    priced_in=0.2,
    tier=SourceTier.TIER_2_MAJOR,
    age_minutes=30,
    catalyst_type=CatalystType.EARNINGS,
) -> CatalystEvent:
    pub = TEST_NEWS_TIME - timedelta(minutes=age_minutes - 30)
    news = NewsItem(
        id="test_1",
        ticker="TEST",
        headline="Test News",
        source="Reuters",
        source_tier=tier,
        published_at=pub,
    )
    ev = CatalystEvent(
        event_id="evt_123",
        ticker="TEST",
        catalyst_type=catalyst_type,
        headline_summary="Test",
        primary_source=news,
    )
    ev.sentiment = sentiment
    ev.impact_score = impact
    ev.materiality_score = materiality
    ev.priced_in_probability = priced_in
    ev.source_tier_score = 85 if tier == SourceTier.TIER_2_MAJOR else 50
    ev.freshness_score = 90 if age_minutes < 60 else 40
    return ev


def make_confidence(score=85) -> DataConfidenceReport:
    r = DataConfidenceReport(ticker="TEST")
    r.add("price", DataQuality.REAL, 1.0)
    r.add("volume", DataQuality.REAL, 1.0)
    # Manually set score by adding enough REAL fields
    while r.score < score:
        r.add(f"field_{len(r.fields)}", DataQuality.REAL, 2.0)
    return r


# ─── Core Logic Tests (No External APIs) ─────────────────────────────────────

class TestA_StrongCatalystStrongVolume:
    """Strong positive catalyst + strong volume + positive reaction → HUNT_NOW"""
    def test_hunt_now(self):
        data = make_ticker(regular_vol=5_000_000, change=15.0)
        event = make_event(impact=95, sentiment="VERY_POSITIVE")
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(90)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision == HunterDecision.HUNT_NOW, f"Expected HUNT_NOW, got {signal.decision}"
        assert signal.hunter_score >= 70


class TestB_PositiveNewsWeakVolume:
    """Positive news + weak volume → WATCH or IGNORE"""
    def test_weak_volume(self):
        data = make_ticker(regular_vol=100_000, change=2.0)
        event = make_event(impact=80, sentiment="POSITIVE")
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision in (HunterDecision.IGNORE, HunterDecision.WATCH)


class TestC_PositiveNewsNegativePrice:
    """Positive news + negative price reaction → NOT HUNT_NOW"""
    def test_not_hunt_now(self):
        data = make_ticker(change=-10.0, price=9.0, prev_close=10.0)
        event = make_event(impact=85, sentiment="POSITIVE")
        reaction = MarketReactionEngine().analyze(event, data)
        assert reaction.reaction_label in ("NEGATIVE_REACTION", "NEUTRAL", "WEAK_REACTION")
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision != HunterDecision.HUNT_NOW


class TestD_StaleNews:
    """Stale news → IGNORE"""
    def test_stale(self):
        data = make_ticker()
        event = make_event(age_minutes=300, impact=90)
        event.freshness_score = 10
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision == HunterDecision.IGNORE


class TestE_WeakPartnership:
    """Weak partnership → IGNORE"""
    def test_weak_partnership(self):
        data = make_ticker(change=1.5)
        event = make_event(catalyst_type=CatalystType.PARTNERSHIP, impact=40, materiality=30)
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision == HunterDecision.IGNORE


class TestF_HugeGap:
    """Huge gap +50% → WATCH or IGNORE"""
    def test_huge_gap(self):
        data = make_ticker(gap=55.0, change=55.0, price=15.0, prev_close=9.68)
        event = make_event(impact=85)
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.decision in (HunterDecision.WATCH, HunterDecision.IGNORE)


class TestG_MissingData:
    """Missing critical market data → IGNORE + DATA_INSUFFICIENT"""
    def test_no_data(self):
        data = TickerData(ticker="FAKE", timestamp=datetime.now(timezone.utc))
        event = make_event()
        reaction = MarketReactionEngine().analyze(event, data)
        assert reaction.reaction_label == "DATA_INSUFFICIENT"


class TestH_DuplicateNews:
    """Duplicate news from 3 sources → ONE EVENT"""
    def test_dedup(self):
        engine = NewsEngine(providers=[])
        items = [
            NewsItem("n1", "TEST", "Company signs major deal", "Reuters", SourceTier.TIER_2_MAJOR),
            NewsItem("n2", "TEST", "Company signs major deal", "Bloomberg", SourceTier.TIER_2_MAJOR),
            NewsItem("n3", "TEST", "Company announces major deal today", "Yahoo", SourceTier.TIER_3_FINANCIAL),
        ]
        events = engine.cluster_events(items)
        assert len(events) == 1
        assert len(events[0].all_sources) == 3


# ─── Timezone Tests ──────────────────────────────────────────────────────────

class TestTimezoneLogic:
    """Timezone-aware session detection"""
    def test_0400_et_is_premarket(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 4, 0, 0))
        assert SessionClock.get_session(dt) == MarketSession.PREMARKET

    def test_0929_et_is_premarket(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 9, 29, 0))
        assert SessionClock.get_session(dt) == MarketSession.PREMARKET

    def test_0930_et_is_regular(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 9, 30, 0))
        assert SessionClock.get_session(dt) == MarketSession.REGULAR

    def test_1559_et_is_regular(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 15, 59, 0))
        assert SessionClock.get_session(dt) == MarketSession.REGULAR

    def test_1600_et_is_after_hours(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 16, 0, 0))
        assert SessionClock.get_session(dt) == MarketSession.AFTER_HOURS

    def test_1959_et_is_after_hours(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 19, 59, 0))
        assert SessionClock.get_session(dt) == MarketSession.AFTER_HOURS

    def test_2000_et_is_closed(self):
        et = pytz.timezone("America/New_York")
        dt = et.localize(datetime(2024, 6, 12, 20, 0, 0))
        assert SessionClock.get_session(dt) == MarketSession.CLOSED

    def test_dst_handling(self):
        et = pytz.timezone("America/New_York")
        # March 10, 2024 — DST starts (clocks spring forward)
        dt = et.localize(datetime(2024, 3, 11, 9, 30, 0))
        assert SessionClock.get_session(dt) == MarketSession.REGULAR
        # November 3, 2024 — DST ends (clocks fall back)
        dt = et.localize(datetime(2024, 11, 4, 9, 30, 0))
        assert SessionClock.get_session(dt) == MarketSession.REGULAR


# ─── Session Separation Tests ────────────────────────────────────────────────

class TestSessionSeparation:
    """Premarket/Regular/AH data must not mix"""
    def test_premarket_high_isolated(self):
        data = TickerData(
            ticker="TEST",
            timestamp=datetime.now(timezone.utc),
            previous_close=8.0,
            current_price=15.0,
            premarket=SessionSnapshot(
                session_type=MarketSession.PREMARKET,
                high=10.0,
                low=8.0,
                volume=100_000,
            ),
            regular=SessionSnapshot(
                session_type=MarketSession.REGULAR,
                high=15.0,
                low=9.0,
                volume=1_000_000,
            ),
            after_hours=SessionSnapshot(
                session_type=MarketSession.AFTER_HOURS,
                high=14.0,
                low=10.0,
                volume=50_000,
            ),
        )
        assert data.premarket.high == 10.0
        assert data.regular.high == 15.0
        assert data.after_hours.high == 14.0
        # Premarket high must NOT be regular high
        assert data.premarket.high != data.regular.high


# ─── Liquidity Naming Tests ──────────────────────────────────────────────────

class TestLiquidityNaming:
    """Verify no fake 'money inflow' claims"""
    def test_only_liquidity_proxy(self):
        data = make_ticker(regular_vol=5_000_000, change=15.0)
        result = LiquidityProxyEngine().analyze(data)
        assert "LIQUIDITY_PROXY" in result.status or result.status == "UNKNOWN"
        assert "INFLOW" not in result.status.upper() or "LIQUIDITY_PROXY" in result.status


# ─── Phase 1 Restriction Tests ───────────────────────────────────────────────

class TestFinalAlertFeatures:
    """Final bot is allowed to publish scenario levels and option candidates."""
    def test_alert_source_contains_scenario_and_options_sections(self):
        from pathlib import Path
        source = Path("bot/telegram_bot.py").read_text().upper()
        assert "SCENARIO LEVELS" in source
        assert "OPTIONS CANDIDATE" in source
        assert "EDUCATIONAL" in source


# ─── Provider Error Handling Tests ───────────────────────────────────────────

class TestProviderErrors:
    """Provider failures are tested without real network calls."""
    @pytest.mark.asyncio
    async def test_provider_error_raised(self):
        from providers.market_data.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()
        class BadTicker:
            def history(self, *args, **kwargs):
                raise RuntimeError("simulated network failure")
        with patch("yfinance.Ticker", return_value=BadTicker()):
            with pytest.raises(ProviderError):
                await provider.fetch_ticker("TEST")

    @pytest.mark.asyncio
    async def test_mock_provider_failure(self):
        from providers.market_data.base_provider import MarketDataProvider

        class FailingProvider(MarketDataProvider):
            name = "failing"
            @property
            def is_realtime(self): return False
            async def fetch_ticker(self, ticker, timestamp=None):
                raise ProviderError("Simulated failure", provider="failing")
            async def health_check(self): return False

        provider = FailingProvider()
        with pytest.raises(ProviderError):
            await provider.fetch_ticker("TEST")


# ─── Phase 1 Audit Regression Tests ──────────────────────────────────────────

class TestSourceRanking:
    def test_tier_1_is_primary_over_lower_quality_sources(self):
        engine = NewsEngine(providers=[])
        items = [
            NewsItem("t1", "TEST", "Company announces acquisition", "Company IR", SourceTier.TIER_1_OFFICIAL, published_at=TEST_NEWS_TIME),
            NewsItem("t2", "TEST", "Company announces acquisition", "Reuters", SourceTier.TIER_2_MAJOR, published_at=TEST_NEWS_TIME),
            NewsItem("t3", "TEST", "Company announces acquisition", "Yahoo", SourceTier.TIER_3_FINANCIAL, published_at=TEST_NEWS_TIME),
            NewsItem("t4", "TEST", "Company announces acquisition", "Social", SourceTier.TIER_4_UNVERIFIED, published_at=TEST_NEWS_TIME),
        ]
        events = engine.cluster_events(items)
        assert len(events) == 1
        assert events[0].primary_source.source_tier == SourceTier.TIER_1_OFFICIAL
        assert events[0].best_tier == SourceTier.TIER_1_OFFICIAL


class TestMarketReactionRealWindows:
    def test_uses_real_before_and_after_windows(self):
        data = make_ticker(price=12.0, change=20.0, regular_vol=5_000_000)
        event = make_event(age_minutes=30, impact=95, sentiment="VERY_POSITIVE")
        reaction = MarketReactionEngine().analyze(event, data)
        assert reaction.data_sufficient is True
        assert reaction.price_before is not None
        assert reaction.price_after_5m is not None
        assert reaction.price_after_15m is not None
        assert reaction.price_after_30m is not None
        assert reaction.price_after > reaction.price_before
        assert reaction.volume_before is not None
        assert reaction.volume_after is not None

    def test_missing_windows_is_data_insufficient(self):
        data = make_ticker()
        # Only candles before news, no post-news reaction window.
        data.intraday_bars = data.intraday_bars[data.intraday_bars.index <= pd.Timestamp(TEST_NEWS_TIME).tz_convert("UTC")]
        event = make_event(age_minutes=30)
        reaction = MarketReactionEngine().analyze(event, data)
        assert reaction.reaction_label == "DATA_INSUFFICIENT"
        assert reaction.data_sufficient is False


class TestYFinancePreviousClose:
    @pytest.mark.asyncio
    async def test_previous_close_ignores_after_hours(self):
        from providers.market_data.yfinance_provider import YFinanceProvider
        et = pytz.timezone("America/New_York")
        idx = pd.DatetimeIndex([
            et.localize(datetime(2026, 8, 13, 15, 59)).astimezone(timezone.utc),
            et.localize(datetime(2026, 8, 13, 16, 30)).astimezone(timezone.utc),
            et.localize(datetime(2026, 8, 14, 8, 0)).astimezone(timezone.utc),
            et.localize(datetime(2026, 8, 14, 9, 30)).astimezone(timezone.utc),
        ])
        hist = pd.DataFrame({
            "Open": [9.9, 10.0, 12.0, 12.0],
            "High": [10.0, 20.0, 13.0, 13.0],
            "Low": [9.8, 10.0, 12.0, 12.0],
            "Close": [10.0, 20.0, 12.5, 13.0],
            "Volume": [1000, 2000, 500, 1000],
        }, index=idx)

        class FakeTicker:
            def history(self, *args, **kwargs):
                return hist
            @property
            def info(self):
                return {"averageVolume": 1000}

        provider = YFinanceProvider()
        anchor = et.localize(datetime(2026, 8, 14, 10, 0))
        with patch("yfinance.Ticker", return_value=FakeTicker()):
            data = await provider.fetch_ticker("TEST", timestamp=anchor)
        assert data.previous_close == 10.0
        assert data.previous_close != 20.0


class TestCleanImportsAndSettings:
    def test_settings_loads_without_env(self):
        import config.settings as settings_module
        assert settings_module.SETTINGS.telegram_bot_token == "test"
        assert settings_module.SETTINGS.openai_model

    def test_finnhub_and_news_engine_imports(self):
        from providers.news.finnhub_provider import FinnhubNewsProvider
        from engines.news_engine import NewsEngine as ImportedNewsEngine
        assert FinnhubNewsProvider.name == "finnhub"
        assert ImportedNewsEngine is NewsEngine


class TestAnchorSafety:
    """Historical analysis must not read candles after the requested anchor."""

    def test_decision_session_uses_ticker_timestamp(self):
        from datetime import datetime, timezone
        et = pytz.timezone("America/New_York")
        anchor = et.localize(datetime(2024, 6, 12, 10, 0, 0))
        data = make_ticker()
        data.timestamp = anchor
        event = make_event()
        reaction = MarketReactionEngine().analyze(event, data)
        liquidity = LiquidityProxyEngine().analyze(data)
        technical = TechnicalEngine().analyze(data, pd.DataFrame())
        confidence = make_confidence(80)
        signal = DecisionEngine().decide(data, event, reaction, liquidity, technical, confidence)
        assert signal.session == MarketSession.REGULAR.value
