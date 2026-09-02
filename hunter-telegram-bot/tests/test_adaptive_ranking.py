import tempfile, os
from models.discovery import DiscoveryCandidate
from engines.discovery import DiscoveryEngine
from core.memory import SignalMemory
from models.signal import HunterSignal, HunterDecision
from core.adaptive import get_adjustments, adjustment_for, clear_cache, BASELINE_WIN_RATE

def _seed_memory(path, setup, wins, losses):
    m=SignalMemory(path)
    for i in range(wins):
        s=HunterSignal(ticker=f"T{i}", decision=HunterDecision.HUNT_NOW, hunter_score=80)
        s.technical_setup=setup; s.catalyst_type="OTHER"
        sid=m.create_signal(s)
        m.update_outcome(sid, outcome_price=110, status="WON", forward_return=0.1)
        # update setup_name via direct DB
        import sqlite3
        with sqlite3.connect(path) as db:
            db.execute("UPDATE signals SET setup_name=? WHERE signal_id=?", (setup, sid))
    for i in range(losses):
        s=HunterSignal(ticker=f"L{i}", decision=HunterDecision.HUNT_NOW, hunter_score=80)
        s.technical_setup=setup
        sid=m.create_signal(s)
        m.update_outcome(sid, outcome_price=90, status="LOST", forward_return=-0.1)
        import sqlite3
        with sqlite3.connect(path) as db:
            db.execute("UPDATE signals SET setup_name=? WHERE signal_id=?", (setup, sid))
    return m

def test_high_performing_positive_bounded():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"a.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 18, 2)  # 90% win rate, dev +40pp -> +5
        # patch default path by monkey patching SignalMemory default? Instead test via direct call with custom path
        # Use get_adjustments with custom memory
        adj=get_adjustments(m, force=True)
        assert adj.get("BREAKOUT")==5
        # ranking test: equal base scores, high setup ahead
        eng=DiscoveryEngine.__new__(DiscoveryEngine)
        c1=DiscoveryCandidate(symbol="AAA", price=50, volume=5_000_000, change_percent=10, market_cap=500_000_000, sources=["a"])
        c2=DiscoveryCandidate(symbol="BBB", price=50, volume=5_000_000, change_percent=10, market_cap=500_000_000, sources=["a"])
        # manually set scores equal then apply adaptive
        eng._score(c1); eng._score(c2)
        # Simulate high performing BREAKOUT vs low: we test adjustment directly
        assert adjustment_for("BREAKOUT", m)==5

def test_low_performing_negative():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"b.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 2, 18)  # 10% win rate -> -5
        adj=get_adjustments(m, force=True)
        assert adj.get("BREAKOUT")==-5

def test_sample_lt20_zero():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"c.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 10, 5)  # 15 total <20
        adj=get_adjustments(m, force=True)
        assert adj.get("BREAKOUT") is None or adj.get("BREAKOUT")==0
        assert adjustment_for("BREAKOUT", m)==0

def test_deviation_le15_zero():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"d.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 12, 8)  # 60% win rate dev 10pp <=15 -> 0
        adj=get_adjustments(m, force=True)
        assert adj.get("BREAKOUT") is None

def test_capped():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"e.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 20, 0)  # 100% -> dev 50pp -> capped +5
        adj=get_adjustments(m, force=True)
        assert adj.get("BREAKOUT")==5

def test_unknown_missing_neutral():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"f.sqlite3")
        m=SignalMemory(p)
        assert adjustment_for("UNKNOWN", m)==0
        assert adjustment_for("", m)==0
        assert adjustment_for(None, m)==0

def test_existing_scoring_intact():
    eng=DiscoveryEngine.__new__(DiscoveryEngine)
    c=DiscoveryCandidate(symbol="TST", price=50, volume=5_000_000, change_percent=10, market_cap=500_000_000, sources=["a"])
    eng._score(c)
    assert "move" in c.score_breakdown and "adaptive" in c.score_breakdown

def test_deterministic():
    with tempfile.TemporaryDirectory() as d:
        clear_cache()
        p=os.path.join(d,"g.sqlite3")
        m=_seed_memory(p, "BREAKOUT", 18, 2)
        a=get_adjustments(m, force=True)
        b=get_adjustments(m)
        assert a==b
