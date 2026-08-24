"""Phase 2.10 integration — TargetEngine wired into DecisionEngine/HunterSignal.

Deterministic (no network). Proves:
- TargetResult produced by TargetEngine is attached to HunterSignal via decide()
- decision authority is unchanged when target_result is supplied
- target zone ordering is mathematically correct (LONG/SHORT)
- missing entry/invalidation yields target_result=None without breaking decide()
"""
import types
from datetime import datetime
from models.swing import SwingIntelligence, SwingLevel, SwingEntry
from models.technical import TechnicalIntelligence
from models.target import TargetResult
from engines.target_engine import TargetEngine
from engines.decision_engine import DecisionEngine
from models.signal import HunterSignal, HunterDecision


# ---------------------------------------------------------------------------
# fakes for the non-target decide() inputs (kept minimal + deterministic)
# ---------------------------------------------------------------------------
def _event():
    e = types.SimpleNamespace()
    e.catalyst_type = types.SimpleNamespace(value="EARNINGS")
    e.sentiment = "POSITIVE"
    e.source_tier_score = 80
    e.impact_score = 80
    e.priced_in_probability = 0.1
    e.primary_source = types.SimpleNamespace(published_at="2025-01-01T00:00:00Z")
    e.is_fresh = lambda max_age_minutes=120: True
    return e


def _reaction():
    r = types.SimpleNamespace()
    r.reaction_score = 70
    r.reaction_label = "POSITIVE_REACTION"
    return r


def _liquidity():
    l = types.SimpleNamespace()
    l.score = 70
    l.status = "NORMAL"
    return l


def _technical_profile():
    t = types.SimpleNamespace()
    t.setup_score = 70
    t.warnings = []
    return t


def _confidence():
    c = types.SimpleNamespace()
    c.score = 80
    return c


def _ticker_data():
    d = types.SimpleNamespace()
    d.ticker = "TEST"
    d.current_price = 134.5
    d.change_percent = 1.0
    d.relative_volume = 1.2
    d.timestamp = datetime(2025, 1, 1, 14, 30)
    return d


def _build_swing(levels, entry_zone_low, invalidation, side="LONG"):
    sw = SwingIntelligence(ticker="TEST")
    sw.levels = levels
    sw.entry = SwingEntry(
        status="READY", setup="BREAKOUT", side=side,
        entry_zone_low=entry_zone_low, entry_zone_high=entry_zone_low + 0.5,
        invalidation_price=invalidation, confidence=70,
    )
    return sw


def _decide_with(target_result):
    eng = DecisionEngine()
    return eng.decide(
        ticker_data=_ticker_data(), event=_event(), reaction=_reaction(),
        liquidity=_liquidity(), technical=_technical_profile(),
        confidence_report=_confidence(), options=None, risk_plan=None,
        trap_risk=0, trap_warnings=[], market_context=None,
        technical_intelligence=TechnicalIntelligence(ticker="TEST"),
        intraday_intelligence=None, swing_intelligence=_build_swing([], 134.5, 132.0),
        target_result=target_result,
    )


def _target_for(levels, entry, inv):
    sw = _build_swing(levels, entry, inv)
    tech = TechnicalIntelligence(ticker="TEST")
    return TargetEngine().build(swing=sw, technical=tech, intraday=None,
                               entry_price=entry, invalidation=inv)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_target_result_attached_to_signal():
    levels = [SwingLevel(price=145.0, level_type="RESISTANCE", strength=85,
                         distance_pct=7.8, evidence="pivot high", role="MAJOR")]
    tr = _target_for(levels, 134.5, 132.0)
    sig = _decide_with(tr)
    assert sig.target_result is tr
    assert any("TARGETS[LONG]" in w for w in sig.warnings)


def test_decision_authority_unchanged_by_target():
    levels = [SwingLevel(price=145.0, level_type="RESISTANCE", strength=85,
                         distance_pct=7.8, evidence="pivot high", role="MAJOR")]
    tr = _target_for(levels, 134.5, 132.0)
    sig_with = _decide_with(tr)
    sig_without = _decide_with(None)
    # target_result must NOT alter the final decision
    assert sig_with.decision == sig_without.decision
    assert sig_with.hunter_score == sig_without.hunter_score


def test_long_target_ordering_math():
    levels = [
        SwingLevel(price=145.0, level_type="RESISTANCE", strength=85, distance_pct=7.8, evidence="r1", role="MAJOR"),
        SwingLevel(price=150.0, level_type="RESISTANCE", strength=70, distance_pct=11.5, evidence="r2", role="MAJOR"),
    ]
    entry, inv = 134.5, 132.0
    tr = _target_for(levels, entry, inv)
    # invalidation < entry < TP1 < TP2
    assert inv < entry
    assert tr.direction == "LONG"
    assert tr.tp1.zone.zone_low > entry
    assert tr.tp2.zone.zone_low > tr.tp1.zone.zone_low
    # R:R positive and mathematically consistent
    assert tr.risk_reward > 0
    assert abs(tr.risk_reward - (tr.tp1.zone.zone_low - entry) / (entry - inv)) < 1e-6


def test_short_target_ordering_math():
    levels = [SwingLevel(price=128.0, level_type="SUPPORT", strength=80,
                         distance_pct=-4.8, evidence="s1", role="MAJOR")]
    entry, inv = 134.5, 137.0
    tr = _target_for(levels, entry, inv)
    assert tr.direction == "SHORT"
    assert tr.tp1.zone.zone_high < entry
    assert inv > entry
    assert tr.risk_reward > 0


def test_missing_entry_yields_none_target_safely():
    # swing entry with no zone -> run.py would pass target_result=None
    sig = _decide_with(None)
    assert sig.target_result is None


def test_target_result_is_none_when_no_structural_level():
    # entry valid but only a support below for a LONG -> no LONG target
    levels = [SwingLevel(price=128.0, level_type="SUPPORT", strength=80,
                         distance_pct=-4.8, evidence="s1", role="MAJOR")]
    tr = _target_for(levels, 134.5, 132.0)
    assert tr.status == "UNAVAILABLE"
    assert tr.tp1 is None
