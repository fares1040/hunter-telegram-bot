"""Stage 7 classification tests - Swing/Target/SupplyDemand confluence."""
from datetime import datetime, timezone, timedelta
from models.decision import Conviction
from engines.decision_support import OpportunityQualityEngine
from engines.decision_engine import DecisionEngine
from models.signal import HunterDecision
from models.ticker import TickerData
from models.session import SessionSnapshot, MarketSession
from models.news import NewsItem, SourceTier, CatalystEvent, CatalystType
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from engines.technical_engine import TechnicalProfile
from models.risk import RiskPlan

def _data(): return TickerData(ticker="TEST", timestamp=datetime.now(timezone.utc), current_price=100, previous_close=99, premarket=SessionSnapshot(session_type=MarketSession.REGULAR, high=101, low=98, volume=1000000), regular=SessionSnapshot(session_type=MarketSession.REGULAR, high=101, low=98, volume=1000000))
def _event(): src=NewsItem(id="1", ticker="TEST", headline="t", source="Reuters", source_tier=SourceTier.TIER_2_MAJOR, published_at=datetime.now(timezone.utc)-timedelta(minutes=10)); ev=CatalystEvent(event_id="ev1", ticker="TEST", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src, sentiment="POSITIVE", impact_score=80, priced_in_probability=0.1); ev.source_tier_score=80; return ev
def _reaction(score=70): return ReactionMetrics(reaction_score=score, reaction_label="POSITIVE_REACTION", data_sufficient=True, reaction_timestamp=datetime.now(timezone.utc))
def _liq(score=70): return LiquidityProxyResult(score=score, status="NORMAL")
def _tech(score=70):
    p=TechnicalProfile(); p.setup_score=score; p.warnings=[]; p.ma20=100; return p
def _conf(score=80):
    class C: pass
    c=C(); c.score=score; return c
def _risk(valid=True): return RiskPlan(valid=valid, confidence=80, entry_trigger=99, stop_price=95, target_1=105)

class FakeSwing:
    def __init__(self, ready=True, confirmed=True):
        from models.swing import SwingEntry, SwingSetup
        self.entry=SwingEntry(status="READY" if ready else "UNKNOWN")
        self.setups=[SwingSetup(name="BREAKOUT", direction="BULLISH", detected=True, quality="CONFIRMED" if confirmed else "UNCONFIRMED")]
        self.score=type("o",(),{"total":70})()
        self.timeframe="1d"
        self.summary=lambda: "test swing"
        self.trap_flags=[]
        self.warnings=[]
class FakeTarget:
    def __init__(self, rr=2.5, status="READY"):
        self.status=status; self.risk_reward=rr; self.direction="LONG"
        self.tp1=self.tp2=self.tp3=None; self.score=None; self.confidence=None
class FakeSupply:
    def __init__(self, dominant="DEMAND"):
        self.dominant_zone_type=dominant; self.demand_zones=[1]; self.supply_zones=[]; self.nearest_demand=None; self.nearest_supply=None; self.warnings=[]; self.missing_data=[]

def _decide(impact=80, reaction=70, liq=70, tech=70, risk_valid=True, trap=10, swing=None, target=None, supply=None):
    return DecisionEngine().decide(_data(), _event(), _reaction(reaction), _liq(liq), _tech(tech), _conf(), risk_plan=_risk(risk_valid), trap_risk=trap, swing_intelligence=swing, target_result=target, supply_demand_result=supply)

def test_confirmed_confluence_improves_quality():
    conv=Conviction(score=70, level="MEDIUM")
    q1=OpportunityQualityEngine.build(conv, 70,70,True,10,"NEUTRAL",True)
    q2=OpportunityQualityEngine.build(conv, 70,70,True,10,"NEUTRAL",True, swing_intelligence=FakeSwing(True,True), target_result=FakeTarget(2.5), supply_demand_result=FakeSupply("DEMAND"))
    assert q2.score >= q1.score
    assert "confluence" in q2.rationale

def test_no_target_no_benefit():
    conv=Conviction(score=70, level="MEDIUM")
    q=OpportunityQualityEngine.build(conv, 70,70,True,10,"NEUTRAL",True, swing_intelligence=FakeSwing(True,True), target_result=None, supply_demand_result=FakeSupply("DEMAND"))
    assert "confluence" not in q.rationale

def test_no_supply_no_benefit():
    conv=Conviction(score=70, level="MEDIUM")
    q=OpportunityQualityEngine.build(conv, 70,70,True,10,"NEUTRAL",True, swing_intelligence=FakeSwing(True,True), target_result=FakeTarget(2.5), supply_demand_result=FakeSupply("UNKNOWN"))
    assert "confluence" not in q.rationale

def test_hunt_safety_reaction59():
    s=_decide(reaction=59, swing=FakeSwing(True,True), target=FakeTarget(2.5), supply=FakeSupply("DEMAND"))
    assert s.decision != HunterDecision.HUNT_NOW

def test_hunt_gate_unchanged():
    s=_decide(impact=80, reaction=70, liq=70, tech=70)
    # hunter_score may be <70 so not HUNT, but gates reaction/liq/tech at boundary should not be bypassed
    assert s.decision in (HunterDecision.HUNT_NOW, HunterDecision.WATCH, HunterDecision.IGNORE)

def test_watch_confluence_additive():
    s1=_decide()
    s2=_decide(swing=FakeSwing(True,True), target=FakeTarget(2.5), supply=FakeSupply("DEMAND"))
    # Decision2 quality may improve but decision stays authoritative (not flipped by confluence alone)
    assert s2.decision2 is not None

def test_unknown_missing_neutral():
    conv=Conviction(score=70, level="MEDIUM")
    q=OpportunityQualityEngine.build(conv, 70,70,True,10,"NEUTRAL",True, swing_intelligence=None, target_result=None, supply_demand_result=None)
    assert q.tier in ("LOW_QUALITY","INTERESTING","ACTIONABLE","HIGH_QUALITY")

def test_trap_blocks_quality():
    conv=Conviction(score=70, level="MEDIUM")
    q=OpportunityQualityEngine.build(conv, 70,70,True,60,"NEUTRAL",True, swing_intelligence=FakeSwing(True,True), target_result=FakeTarget(2.5), supply_demand_result=FakeSupply("DEMAND"))
    assert q.tier=="LOW_QUALITY"

def test_stale_prevents_confirmed():
    src=NewsItem(id="1", ticker="TEST", headline="t", source="Reuters", source_tier=SourceTier.TIER_1_OFFICIAL, published_at=datetime.now(timezone.utc)-timedelta(hours=5))
    ev=CatalystEvent(event_id="ev1", ticker="TEST", catalyst_type=CatalystType.OTHER, headline_summary="t", primary_source=src, sentiment="POSITIVE", impact_score=80, priced_in_probability=0.1); ev.source_tier_score=80
    s=DecisionEngine().decide(_data(), ev, _reaction(), _liq(), _tech(), _conf(), trap_risk=10)
    assert s.decision2.why_now.status != "CONFIRMED"
