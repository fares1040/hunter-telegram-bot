"""Stage 4 Decision 2.0 tests."""
from engines.decision_support import WhyNowBuilder, ConflictDetector, ConvictionEngine, OpportunityQualityEngine
from datetime import datetime, timezone, timedelta

def _now(): return datetime.now(timezone.utc)

def test_why_now_confirmed():
    c = _now() - timedelta(minutes=10)
    r = _now() - timedelta(minutes=5)
    w = WhyNowBuilder.build(c, r, "STRONG_POSITIVE_REACTION", True)
    assert w.status in ("CONFIRMED","PARTIAL")
    assert w.elapsed_seconds == 300
def test_why_now_unknown_when_missing():
    w = WhyNowBuilder.build(None, None, "UNKNOWN")
    assert w.status=="UNKNOWN"
def test_no_fabricated_causality_future():
    c = _now() + timedelta(minutes=10)
    r = _now()
    w = WhyNowBuilder.build(c, r, "STRONG_POSITIVE_REACTION")
    assert w.status=="UNKNOWN"
def test_catalyst_after_reaction_unknown():
    c = _now() - timedelta(minutes=2)
    r = _now() - timedelta(minutes=10)
    w = WhyNowBuilder.build(c, r, "POSITIVE_REACTION")
    assert w.status=="UNKNOWN"
def test_data_insufficient_unknown():
    c = _now() - timedelta(minutes=10)
    r = _now() - timedelta(minutes=5)
    w = WhyNowBuilder.build(c, r, "DATA_INSUFFICIENT")
    assert w.status=="UNKNOWN"
def test_stale_window_unknown():
    c = _now() - timedelta(hours=5)
    r = _now() - timedelta(hours=4)
    w = WhyNowBuilder.build(c, r, "STRONG_POSITIVE_REACTION")
    assert w.status=="UNKNOWN"
def test_conflict_detection():
    c = ConflictDetector.detect("POSITIVE", "WEAK_REACTION", "STRONG_LIQUIDITY_PROXY", "CALL_LEAN", "NEUTRAL", 0, True, True)
    assert any(x.type=="catalyst_vs_reaction" for x in c)
def test_unknown_not_bearish():
    c = ConflictDetector.detect("POSITIVE", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", 0, False, False)
    assert not any(x.type=="price_vs_options_flow" for x in c)
def test_conviction_stale_penalty():
    conv = ConvictionEngine.build(80, 80, 20, 80, [])
    assert conv.score < 70
def test_conviction_independent_from_hunter():
    # high hunter-style alignment but low freshness should still penalize
    conv_low = ConvictionEngine.build(80, 90, 20, 90, [])
    conv_high = ConvictionEngine.build(80, 90, 80, 90, [])
    assert conv_high.score > conv_low.score
def test_opportunity_quality_tiers():
    from models.decision import Conviction
    conv = Conviction(score=80, level="HIGH")
    q = OpportunityQualityEngine.build(conv, 80, 80, True, 10, "NEUTRAL", True)
    assert q.tier in ("HIGH_QUALITY","ACTIONABLE")
def test_opportunity_low_when_risk_invalid():
    from models.decision import Conviction
    conv = Conviction(score=80, level="HIGH")
    q = OpportunityQualityEngine.build(conv, 80, 80, False, 10, "NEUTRAL", True)
    assert q.tier=="UNAVAILABLE"
def test_signal_backward_compat():
    from models.signal import HunterSignal, HunterDecision
    s = HunterSignal(ticker="AAPL", decision=HunterDecision.IGNORE)
    assert hasattr(s, "decision2")
    assert s.decision2 is None
    d = s.to_dict()
    assert d["decision"]=="IGNORE"
def test_decision_engine_still_authority():
    from engines.decision_engine import DecisionEngine
    import inspect
    src = inspect.getsource(DecisionEngine.decide)
    assert "HUNT_NOW" in src
def test_realtime_trueflow_not_bypass():
    # Ensure decision_support never imports Telegram
    import engines.decision_support as m
    import inspect
    src = inspect.getsource(m)
    assert "Telegram" not in src
def test_deterministic_output():
    c = _now() - timedelta(minutes=10)
    r = _now() - timedelta(minutes=5)
    w1 = WhyNowBuilder.build(c, r, "POSITIVE_REACTION")
    w2 = WhyNowBuilder.build(c, r, "POSITIVE_REACTION")
    assert w1.status==w2.status and w1.elapsed_seconds==w2.elapsed_seconds
def test_provenance_preserved():
    from models.decision import DecisionEvidence
    e = DecisionEvidence(name="test", quality="REAL", source="polygon_realtime_trade", freshness="FRESH")
    assert e.source=="polygon_realtime_trade"
    assert e.freshness=="FRESH"
