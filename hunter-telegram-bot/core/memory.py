"""Small SQLite memory for alert deduplication and outcomes."""
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class SignalMemory:
    def __init__(self, path: str = "data/hunter_memory.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS alerts (event_key TEXT PRIMARY KEY, ticker TEXT, decision TEXT, score INTEGER, created_at TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, decision TEXT, score INTEGER, forward_return REAL, created_at TEXT)")

    def seen(self, event_key: str) -> bool:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT 1 FROM alerts WHERE event_key=?", (event_key,)).fetchone() is not None

    def remember(self, event_key: str, ticker: str, decision: str, score: int) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?)", (event_key, ticker, decision, score, datetime.now(timezone.utc).isoformat()))

    def record_outcome(self, ticker: str, decision: str, score: int, forward_return: Optional[float]) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO outcomes(ticker,decision,score,forward_return,created_at) VALUES(?,?,?,?,?)", (ticker, decision, score, forward_return, datetime.now(timezone.utc).isoformat()))
