from datetime import datetime, timezone, timedelta
from models.discovery import DiscoveryCandidate
from engines.discovery import DiscoveryEngine
from engines.candidate_gate import CandidateGate
from models.ticker import TickerData
from models.session import SessionSnapshot, MarketSession
from models.news import NewsItem, SourceTier, CatalystEvent, CatalystType
from engines.news_engine import NewsEngine

def _cand(price, volume, change=10):
    c=DiscoveryCandidate(symbol="TST", price=price, volume=volume, change_percent=change, market_cap=500_000_000, sources=["a"])
    c.sources=["a"]
    return c

def test_quality_ranking():
    eng=DiscoveryEngine.__new__(DiscoveryEngine)
    high=_cand(50, 5_000_000, 10)
    low=_cand(1.5, 200_000, 10)
    eng._score(high); eng._score(low)
    assert high.discovery_score > low.discovery_score

def test_low_price_not_auto_ignore():
    gate=CandidateGate()
    data=TickerData(ticker="CHEAP", timestamp=datetime.now(timezone.utc), current_price=3.0, previous_close=2.9, premarket=SessionSnapshot(session_type=MarketSession.REGULAR, high=3.1, low=2.8, volume=500000))
    data.market_cap=100_000_000
    # price <5 alone should not be rejected if dollar_volume sufficient
    res=gate.evaluate(data)
    # gate only rejects NO_PRICE, price above limit is not <5, so should pass
    assert res.passed

def test_no_fabricated_negative():
    gate=CandidateGate()
    data=TickerData(ticker="LOWVOL", timestamp=datetime.now(timezone.utc), current_price=50, previous_close=49, premarket=SessionSnapshot(session_type=MarketSession.REGULAR, high=51, low=48, volume=100000))
    # LOW_DOLLAR_VOLUME only when dollar_volume <1M - this is real field, not fabricated from volume alone
    # volume 100k * price 50 =5M -> not low
    res=gate.evaluate(data)
    assert res.passed

def test_tier1_contextual_240():
    eng=NewsEngine([])
    now=datetime.now(timezone.utc)
    src1=NewsItem(id="1", ticker="T", headline="t", source="Reuters", source_tier=SourceTier.TIER_1_OFFICIAL, published_at=now-timedelta(minutes=200))
    ev=CatalystEvent(event_id="e1", ticker="T", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src1)
    ev.source_tier_score=100
    filtered=eng.filter_material_events([ev])
    assert len(filtered)==1  # tier1 200m kept
    assert filtered[0].is_fresh(240)
    assert not filtered[0].is_fresh(180)

def test_tier1_over_240_filtered():
    eng=NewsEngine([])
    now=datetime.now(timezone.utc)
    src=NewsItem(id="1", ticker="T", headline="t", source="Reuters", source_tier=SourceTier.TIER_1_OFFICIAL, published_at=now-timedelta(minutes=250))
    ev=CatalystEvent(event_id="e1", ticker="T", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src)
    ev.source_tier_score=100
    assert len(eng.filter_material_events([ev]))==0

def test_lowtier_stale_filtered():
    eng=NewsEngine([])
    now=datetime.now(timezone.utc)
    src=NewsItem(id="1", ticker="T", headline="t", source="Reuters", source_tier=SourceTier.TIER_3_FINANCIAL, published_at=now-timedelta(minutes=200))
    ev=CatalystEvent(event_id="e1", ticker="T", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src)
    ev.source_tier_score=60
    assert len(eng.filter_material_events([ev]))==0

def test_stale_never_confirmed():
    src=NewsItem(id="1", ticker="T", headline="t", source="Reuters", source_tier=SourceTier.TIER_1_OFFICIAL, published_at=datetime.now(timezone.utc)-timedelta(minutes=200))
    ev=CatalystEvent(event_id="e1", ticker="T", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src)
    ev.source_tier_score=100
    # is_fresh 120 false, so WhyNow should be UNKNOWN
    assert not ev.is_fresh(120)
    assert ev.is_fresh(240)

def test_hunt_unchanged():
    from engines.decision_engine import DecisionEngine
    # Ensure DecisionEngine still requires hunter_score etc - regression of stage5
    import inspect
    src=inspect.getsource(DecisionEngine.decide)
    assert "hunter_score >= SETTINGS.hunter_min_score" in src
