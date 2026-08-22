"""Persistent ticker watchlist backed by SQLite."""
import re
import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone


TICKER_PATTERN = re.compile(r"^[A-Z]{1,6}$")
DEFAULT_WATCHLIST = ("AAPL", "NVDA", "TSLA")


def normalize_ticker(raw: str) -> str:
    """Normalize a raw symbol: strip $/spaces, uppercase. Raises ValueError on invalid format."""
    t = (raw or "").strip().upper().lstrip("$").strip()
    if not TICKER_PATTERN.match(t):
        raise ValueError(f"Invalid ticker symbol: {raw!r}. Use 1-6 letters, e.g. AAPL.")
    return t


class WatchlistStore:
    def __init__(self, path: str = "data/hunter_memory.sqlite3", seed_defaults: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, added_at TEXT)")
            if seed_defaults:
                now = datetime.now(timezone.utc).isoformat()
                for ticker in DEFAULT_WATCHLIST:
                    db.execute("INSERT OR IGNORE INTO watchlist VALUES (?,?)", (ticker, now))

    def list(self) -> List[str]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
        return [r[0] for r in rows]

    def contains(self, ticker: str) -> bool:
        try:
            t = normalize_ticker(ticker)
        except ValueError:
            return False
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT 1 FROM watchlist WHERE ticker=?", (t,)).fetchone() is not None

    def add(self, ticker: str) -> Optional[str]:
        """Add ticker. Returns normalized symbol when newly added, None when already present."""
        t = normalize_ticker(ticker)
        with sqlite3.connect(self.path) as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO watchlist VALUES (?,?)",
                (t, datetime.now(timezone.utc).isoformat()),
            )
            return t if cur.rowcount > 0 else None

    def remove(self, ticker: str) -> Optional[str]:
        """Remove ticker. Returns removed symbol, or None when it was not present."""
        t = normalize_ticker(ticker)
        with sqlite3.connect(self.path) as db:
            cur = db.execute("DELETE FROM watchlist WHERE ticker=?", (t,))
            return t if cur.rowcount > 0 else None
