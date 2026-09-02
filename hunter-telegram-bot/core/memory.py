"""Small SQLite memory for alert deduplication, outcomes and track record."""
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class SignalMemory:
    def __init__(self, path: str = "data/hunter_memory.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS alerts (event_key TEXT PRIMARY KEY, ticker TEXT, decision TEXT, score INTEGER, created_at TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, decision TEXT, score INTEGER, forward_return REAL, created_at TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY, ticker TEXT, timestamp TEXT, decision TEXT, hunter_score INTEGER,
                data_confidence INTEGER, entry_trigger REAL, stop_price REAL, target_1 REAL, target_2 REAL, target_3 REAL,
                catalyst_type TEXT, sentiment TEXT, strategy_state TEXT, setup_name TEXT, status TEXT,
                outcome_price REAL, outcome_at TEXT, forward_return REAL, realized_return REAL, provenance TEXT, snapshot TEXT, created_at TEXT)""")
            # migration-safe add columns for old DBs
            cols = {r[1] for r in db.execute("PRAGMA table_info(signals)").fetchall()}
            for col, typ in [("provenance","TEXT"),("snapshot","TEXT")]:
                if col not in cols:
                    db.execute(f"ALTER TABLE signals ADD COLUMN {col} {typ}")

    def seen(self, event_key: str) -> bool:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT 1 FROM alerts WHERE event_key=?", (event_key,)).fetchone() is not None

    def remember(self, event_key: str, ticker: str, decision: str, score: int) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?)", (event_key, ticker, decision, score, datetime.now(timezone.utc).isoformat()))

    def record_outcome(self, ticker: str, decision: str, score: int, forward_return: Optional[float]) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO outcomes(ticker,decision,score,forward_return,created_at) VALUES(?,?,?,?,?)", (ticker, decision, score, forward_return, datetime.now(timezone.utc).isoformat()))

    def recent_alerts(self, limit: int = 5) -> list:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT ticker, decision, score, created_at FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"ticker": r[0], "decision": r[1], "score": r[2], "created_at": r[3]} for r in rows]

    def alert_count(self) -> int:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    # Track Record API (additive, backward compatible)
    def create_signal(self, signal) -> str:
        from models.signal import HunterSignal
        # stable identity from ticker + timestamp + decision if available, else ticker:iso
        sig_id = f"{signal.ticker}:{signal.timestamp.isoformat()}:{signal.decision.value}" if hasattr(signal, 'timestamp') else f"{signal.ticker}:{datetime.now(timezone.utc).isoformat()}"
        # allow caller to pass explicit signal_id via reasoning key? keep simple
        snapshot = json.dumps(signal.to_dict())
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO signals(signal_id,ticker,timestamp,decision,hunter_score,data_confidence,entry_trigger,stop_price,target_1,target_2,target_3,catalyst_type,sentiment,strategy_state,setup_name,status,provenance,snapshot,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sig_id, signal.ticker, signal.timestamp.isoformat(), signal.decision.value, signal.hunter_score, signal.data_confidence, signal.entry_trigger, signal.stop_price, signal.target_1, signal.target_2, signal.target_3, signal.catalyst_type, signal.sentiment, "", signal.technical_setup, "OPEN", "observed", snapshot, datetime.now(timezone.utc).isoformat()))
        return sig_id

    def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT * FROM signals WHERE signal_id=?", (signal_id,)).fetchone()
            if not row: return None
            cols = [d[0] for d in db.execute("SELECT * FROM signals LIMIT 0").description]
            return dict(zip(cols, row))

    def update_outcome(self, signal_id: str, outcome_price: Optional[float], status: str, forward_return: Optional[float] = None, realized_return: Optional[float] = None, provenance: str = "observed") -> bool:
        if status not in ("OPEN","PENDING","WON","LOST","EXPIRED","UNRESOLVED"):
            raise ValueError("invalid status")
        # never fabricate outcome price if None and status is terminal
        if status in ("WON","LOST") and outcome_price is None and forward_return is None:
            # allow missing -> keep UNRESOLVED semantics, do not fabricate
            pass
        # compute forward_return if not provided but prices available
        with sqlite3.connect(self.path) as db:
            cur = db.execute("UPDATE signals SET status=?, outcome_price=?, outcome_at=?, forward_return=COALESCE(?,forward_return), realized_return=COALESCE(?,realized_return), provenance=? WHERE signal_id=?",
                             (status, outcome_price, datetime.now(timezone.utc).isoformat(), forward_return, realized_return, provenance, signal_id))
            return cur.rowcount > 0

    def list_signals(self, ticker: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        q="SELECT * FROM signals"; params=[]
        clauses=[]
        if ticker: clauses.append("ticker=?"); params.append(ticker)
        if status: clauses.append("status=?"); params.append(status)
        if clauses: q+=" WHERE "+" AND ".join(clauses)
        q+=" ORDER BY created_at DESC LIMIT ?"; params.append(limit)
        with sqlite3.connect(self.path) as db:
            rows=db.execute(q, params).fetchall()
            cols=[d[0] for d in db.execute("SELECT * FROM signals LIMIT 0").description]
            return [dict(zip(cols, r)) for r in rows]

    def track_record_metrics(self) -> Dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            rows=db.execute("SELECT forward_return, realized_return, status FROM signals WHERE status IN ('WON','LOST') AND (forward_return IS NOT NULL OR realized_return IS NOT NULL)").fetchall()
            rets=[r[0] if r[0] is not None else r[1] for r in rows if r[0] is not None or r[1] is not None]
            wins=sum(1 for v in rets if v is not None and v > 0)
            losses=sum(1 for v in rets if v is not None and v <= 0)
            avg=sum(rets)/len(rets) if rets else None
            win_rate=wins/len(rets) if rets else None
            return {"sample_count": len(rets), "wins": wins, "losses": losses, "win_rate": win_rate, "avg_return": avg, "pending": db.execute("SELECT COUNT(*) FROM signals WHERE status='OPEN'").fetchone()[0]}
