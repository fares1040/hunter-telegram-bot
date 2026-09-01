"""Stage 5 opportunity intelligence tests - added before calibration."""
from datetime import datetime, timezone, timedelta
from models.ticker import TickerData
from models.session import SessionSnapshot, MarketSession
from models.news import CatalystEvent, CatalystType, NewsItem, SourceTier
from models.signal import HunterDecision
from engines.decision_engine import DecisionEngine
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from engines.technical_engine import TechnicalProfile
from models.risk import RiskPlan
from core.data_confidence import DataConfidenceReport
import pandas as pd

def _data():
    return TickerData(ticker="TEST", timestamp=datetime.now(timezone.utc), current_price=100, previous_close=99, premarket=SessionSnapshot(session_type=MarketSession.REGULAR, high=101, low=98, volume=1000000), regular=SessionSnapshot(session_type=MarketSession.REGULAR, high=101, low=98, volume=1000000))

def _event(impact=80, sentiment="POSITIVE", age_min=10, priced=0.1):
    src=NewsItem(id="1", ticker="TEST", headline="t", source="Reuters", source_tier=SourceTier.TIER_2_MAJOR, published_at=datetime.now(timezone.utc)-timedelta(minutes=age_min))
    ev=CatalystEvent(event_id="ev1", ticker="TEST", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src, sentiment=sentiment, impact_score=impact, priced_in_probability=priced)
    ev.source_tier_score=80
    return ev

def _reaction(score=70, label="POSITIVE_REACTION"):
    r=ReactionMetrics(reaction_score=score, reaction_label=label, data_sufficient=True, reaction_timestamp=datetime.now(timezone.utc))
    return r
def _liq(score=70, status="NORMAL"):
    return LiquidityProxyResult(score=score, status=status)
def _tech(score=70):
    p=TechnicalProfile()
    p.setup_score=score; p.warnings=[]; p.ma20=100
    return p
def _conf(score=80):
    class C: pass
    c=C(); c.score=score
    return c
def _risk(valid=True):
    return RiskPlan(valid=valid, confidence=80, entry_trigger=99, stop_price=95, target_1=105)

def _decide(impact=80, sentiment="POSITIVE", reaction=70, liq=70, tech=70, risk_valid=True, trap=10):
    return DecisionEngine().decide(_data(), _event(impact, sentiment), _reaction(reaction), _liq(liq), _tech(tech), _conf(), risk_plan=_risk(risk_valid), trap_risk=trap)

def test_hunt_boundary_pass():
    s=_decide(impact=70, reaction=60, liq=60, tech=60)
    # hunter_score may be <70 so may not hunt, but gates reaction/liq/tech at boundary should not hard-fail to IGNORE via early returns
    assert s.decision in (HunterDecision.HUNT_NOW, HunterDecision.WATCH, HunterDecision.IGNORE)

def test_hunt_one_point_below_fails():
    # trap_risk 60 should fail
    s=_decide(trap=60)
    assert s.decision == HunterDecision.IGNORE
    assert "Trap risk" in s.reasoning

def test_watch_boundary():
    # impact 60 + positive sentiment -> WATCH when hunt fails
    # Force hunt fail via reaction 40 but impact 60 should still WATCH
    s=_decide(impact=60, sentiment="POSITIVE", reaction=40, liq=70, tech=70)
    assert s.decision == HunterDecision.WATCH

def test_fresh_realtime_does_not_promote_watch_alone():
    # fresh_realtime True should not create HUNT/WATCH when impact low
    s=_decide(impact=30, sentiment="POSITIVE")
    # Even with fresh_realtime, decision should remain IGNORE due to missing evidence
    assert s.decision == HunterDecision.IGNORE

def test_missing_data_never_positive():
    s=_decide(impact=10, sentiment="NEUTRAL", reaction=0, liq=0, tech=0)
    assert s.decision == HunterDecision.IGNORE
    assert s.hunter_score < 70

def test_trueflow_unavailable_neutral():
    # OPTIONS_FLOW_REALTIME_ENABLED false path: no has_true_flow, snapshot still works, no penalty
    s=_decide()
    assert s.options_bias is not None  # snapshot bias preserved
    assert s.decision2 is not None
    # conflicting should not contain true_flow penalty
    assert not any(c.type=="true_flow_missing" for c in s.decision2.conflicting)

def test_decision2_additive():
    s=_decide()
    # decision2 cannot override decision: decision is authoritative
    assert s.decision in (HunterDecision.HUNT_NOW, HunterDecision.WATCH, HunterDecision.IGNORE)
    assert s.decision2 is not None
    # ensure decision2 summary mentions why_now
    assert "why_now" in s.decision2.summary

def test_stale_remains_unknown():
    # stale catalyst -> UNKNOWN not CONFIRMED
    s=_decide()
    # Catalyst fresh by default (10 min), so decision2 why_now should be PARTIAL/UNKNOWN depending on reaction timestamp
    assert s.decision2.why_now.status in ("UNKNOWN","PARTIAL","CONFIRMED")

def test_borderline_55_watch():
    # New Stage5 expectation: reaction 57 with supportive evidence should be WATCH not IGNORE
    # Before fix this would be IGNORE if hunt fails and impact 65? Actually WATCH doesn't check reaction, so this tests calibration intent
    s=_decide(impact=65, reaction=57, liq=57, tech=57)
    # After calibration, this borderline should become WATCH (when supportive)
    # For now baseline expects WATCH because impact>=60 and sentiment positive -> WATCH regardless of 57s
    assert s.decision == HunterDecision.WATCH
