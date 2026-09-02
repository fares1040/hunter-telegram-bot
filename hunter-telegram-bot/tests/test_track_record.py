import tempfile, os
from datetime import datetime, timezone
from models.signal import HunterSignal, HunterDecision
from core.memory import SignalMemory

def _sig(ticker="TESTX", decision=HunterDecision.HUNT_NOW, score=75):
    s=HunterSignal(ticker=ticker, decision=decision, hunter_score=score, data_confidence=80)
    s.entry_trigger=100; s.stop_price=95; s.target_1=110; s.catalyst_type="EARNINGS"; s.sentiment="POSITIVE"; s.technical_setup="BREAKOUT"
    return s

def test_create_and_persist():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        s=_sig()
        sid=m.create_signal(s)
        assert sid
        got=m.get_signal(sid)
        assert got["ticker"]=="TESTX" and got["status"]=="OPEN"

def test_duplicate_handling():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        s=_sig()
        a=m.create_signal(s); b=m.create_signal(s)
        assert a==b
        assert len(m.list_signals())==1

def test_outcome_win_loss():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        sid=m.create_signal(_sig())
        m.update_outcome(sid, outcome_price=110, status="WON", forward_return=0.1)
        assert m.get_signal(sid)["status"]=="WON"
        sid2=m.create_signal(_sig("TESTY"))
        m.update_outcome(sid2, outcome_price=90, status="LOST", forward_return=-0.1)
        metrics=m.track_record_metrics()
        assert metrics["wins"]==1 and metrics["losses"]==1 and metrics["sample_count"]==2

def test_missing_outcome():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        sid=m.create_signal(_sig())
        m.update_outcome(sid, outcome_price=None, status="EXPIRED", forward_return=None)
        assert m.get_signal(sid)["status"]=="EXPIRED"
        assert m.track_record_metrics()["sample_count"]==0  # missing not counted

def test_metrics():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        for ret in [0.05, 0.03, -0.02]:
            sid=m.create_signal(_sig(score=70))
            status="WON" if ret>0 else "LOST"
            m.update_outcome(sid, outcome_price=100*(1+ret), status=status, forward_return=ret)
        metrics=m.track_record_metrics()
        assert metrics["sample_count"]==3 and abs(metrics["avg_return"]-0.02)<0.001 and abs(metrics["win_rate"]-0.666)<0.01

def test_reload_persistence():
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"mem.sqlite3")
        m1=SignalMemory(p); sid=m1.create_signal(_sig())
        m2=SignalMemory(p)
        assert m2.get_signal(sid) is not None

def test_backward_compat():
    with tempfile.TemporaryDirectory() as d:
        m=SignalMemory(os.path.join(d,"mem.sqlite3"))
        m.remember("k1","AAPL","HUNT_NOW",80)
        assert m.seen("k1")
        m.record_outcome("AAPL","HUNT_NOW",80,0.05)
        assert m.alert_count()==1
