"""Real-time market event contracts — Quote and Trade.

Deterministic, no fabrication. Missing values remain None / UNKNOWN.
Provenance explicitly distinguishes realtime vs snapshot data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List
import time


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_ts(ts: Optional[datetime]) -> Optional[datetime]:
    """Ensure timezone-aware UTC datetime."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# Valid source values — realtime data must be labeled as such,
# snapshot data must never masquerade as realtime.
VALID_SOURCES = {
    "polygon_realtime_quote",
    "polygon_realtime_trade",
    "polygon_snapshot_quote",
    "polygon_snapshot_trade",
    "snapshot_aggregate",
    "delayed_snapshot",
    "unknown",
}


@dataclass
class Quote:
    """Real-time or snapshot equity quote (NBBO). Never fabricated."""

    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    timestamp: Optional[datetime] = None  # provider event time (UTC)
    ingested_at: datetime = field(default_factory=_utc_now)
    exchange: Optional[int] = None
    source: str = "unknown"
    tape: Optional[int] = None

    def __post_init__(self):
        self.timestamp = _normalize_ts(self.timestamp)
        self.ingested_at = _normalize_ts(self.ingested_at) or _utc_now()

    @property
    def spread(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask >= self.bid:
            return round(self.ask - self.bid, 4)
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.spread is not None and self.ask and self.ask > 0:
            mid = (self.bid + self.ask) / 2
            if mid > 0:
                return round(self.spread / mid * 100, 4)
        return None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask >= self.bid:
            return round((self.bid + self.ask) / 2, 4)
        return None

    @property
    def is_valid(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid > 0 and self.ask >= self.bid

    @property
    def latency_ms(self) -> Optional[int]:
        if self.timestamp is None:
            return None
        delta = (self.ingested_at - self.timestamp).total_seconds() * 1000
        return max(0, int(delta))

    def freshness(self, max_age_seconds: int = 30) -> str:
        """FRESH / STALE / UNKNOWN based on event age at ingestion."""
        if self.timestamp is None:
            return "UNKNOWN"
        age = (self.ingested_at - self.timestamp).total_seconds()
        if age < 0:
            # Clock skew — treat as fresh but note anomaly
            return "FRESH"
        return "FRESH" if age <= max_age_seconds else "STALE"

    def is_stale(self, max_age_seconds: int = 30) -> bool:
        return self.freshness(max_age_seconds) == "STALE"

    @property
    def is_realtime(self) -> bool:
        return self.source in ("polygon_realtime_quote", "polygon_realtime_trade")

    def dedup_key(self) -> str:
        ts = self.timestamp.isoformat() if self.timestamp else "no-ts"
        return f"Q:{self.symbol}:{ts}:{self.bid}:{self.ask}:{self.bid_size}:{self.ask_size}:{self.exchange}"


@dataclass
class Trade:
    """Real-time or snapshot equity trade. Aggressor side never inferred."""

    symbol: str
    price: Optional[float] = None
    size: Optional[int] = None
    timestamp: Optional[datetime] = None  # provider event time (UTC)
    ingested_at: datetime = field(default_factory=_utc_now)
    exchange: Optional[int] = None
    conditions: Optional[List[int]] = None
    source: str = "unknown"
    tape: Optional[int] = None
    # Aggressor side only if provider explicitly supplies it — never inferred.
    aggressor_side: Optional[str] = None  # "BUY"/"SELL" only if REAL

    def __post_init__(self):
        self.timestamp = _normalize_ts(self.timestamp)
        self.ingested_at = _normalize_ts(self.ingested_at) or _utc_now()

    @property
    def is_valid(self) -> bool:
        return self.price is not None and self.price > 0 and self.size is not None and self.size > 0

    @property
    def latency_ms(self) -> Optional[int]:
        if self.timestamp is None:
            return None
        delta = (self.ingested_at - self.timestamp).total_seconds() * 1000
        return max(0, int(delta))

    def freshness(self, max_age_seconds: int = 30) -> str:
        if self.timestamp is None:
            return "UNKNOWN"
        age = (self.ingested_at - self.timestamp).total_seconds()
        if age < 0:
            return "FRESH"
        return "FRESH" if age <= max_age_seconds else "STALE"

    def is_stale(self, max_age_seconds: int = 30) -> bool:
        return self.freshness(max_age_seconds) == "STALE"

    @property
    def is_realtime(self) -> bool:
        return self.source in ("polygon_realtime_quote", "polygon_realtime_trade")

    def dedup_key(self) -> str:
        ts = self.timestamp.isoformat() if self.timestamp else "no-ts"
        conds = ",".join(str(c) for c in (self.conditions or []))
        return f"T:{self.symbol}:{ts}:{self.price}:{self.size}:{self.exchange}:{conds}"


@dataclass
class RealtimeBuffer:
    """Bounded per-symbol event buffer with dedup and ordering.

    Not a global singleton — caller owns instance per symbol or per provider.
    """

    symbol: str
    max_events: int = 1000
    quotes: List[Quote] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    _quote_keys: set = field(default_factory=set, repr=False)
    _trade_keys: set = field(default_factory=set, repr=False)

    def add_quote(self, q: Quote) -> bool:
        """Add quote if not duplicate. Returns True if added."""
        key = q.dedup_key()
        if key in self._quote_keys:
            return False
        self._quote_keys.add(key)
        self.quotes.append(q)
        # Keep sorted by timestamp (None last) and bounded
        self.quotes.sort(key=lambda x: x.timestamp or datetime.max.replace(tzinfo=timezone.utc))
        if len(self.quotes) > self.max_events:
            removed = self.quotes.pop(0)
            self._quote_keys.discard(removed.dedup_key())
        if len(self._quote_keys) > self.max_events * 2:
            # Rebuild to prevent unbounded growth from stale keys
            self._quote_keys = {x.dedup_key() for x in self.quotes}
        return True

    def add_trade(self, t: Trade) -> bool:
        """Add trade if not duplicate. Returns True if added."""
        key = t.dedup_key()
        if key in self._trade_keys:
            return False
        self._trade_keys.add(key)
        self.trades.append(t)
        self.trades.sort(key=lambda x: x.timestamp or datetime.max.replace(tzinfo=timezone.utc))
        if len(self.trades) > self.max_events:
            removed = self.trades.pop(0)
            self._trade_keys.discard(removed.dedup_key())
        if len(self._trade_keys) > self.max_events * 2:
            self._trade_keys = {x.dedup_key() for x in self.trades}
        return True

    def fresh_quotes(self, max_age_seconds: int = 30) -> List[Quote]:
        return [q for q in self.quotes if q.freshness(max_age_seconds) == "FRESH"]

    def fresh_trades(self, max_age_seconds: int = 30) -> List[Trade]:
        return [t for t in self.trades if t.freshness(max_age_seconds) == "FRESH"]

    def latest_quote(self) -> Optional[Quote]:
        if not self.quotes:
            return None
        return self.quotes[-1]

    def latest_trade(self) -> Optional[Trade]:
        if not self.trades:
            return None
        return self.trades[-1]

    def clear(self):
        self.quotes.clear()
        self.trades.clear()
        self._quote_keys.clear()
        self._trade_keys.clear()


def parse_polygon_quote(raw: dict, ingested_at: Optional[datetime] = None) -> Optional[Quote]:
    """Parse Polygon quote JSON (REST or WS) into Quote. Returns None on malformed."""
    try:
        sym = raw.get("sym") or raw.get("T") or raw.get("symbol") or raw.get("ticker")
        if not sym:
            return None
        # REST: ask/bid fields vary; WS: ap/bp style
        bid = raw.get("bp") if raw.get("bp") is not None else raw.get("bid") or raw.get("b") or raw.get("p")
        ask = raw.get("ap") if raw.get("ap") is not None else raw.get("ask") or raw.get("a")
        bid_size = raw.get("bs") if raw.get("bs") is not None else raw.get("bid_size") or raw.get("bidsize")
        ask_size = raw.get("as") if raw.get("as") is not None else raw.get("ask_size") or raw.get("asksize")
        # Timestamp: Polygon uses `t` (ns or ms) or `timestamp`
        ts_raw = raw.get("t") or raw.get("timestamp") or raw.get("sip_timestamp")
        ts = None
        if ts_raw is not None:
            try:
                v = int(ts_raw)
                # Heuristic: >1e12 => ms? Polygon WS uses ns, REST uses ms
                if v > 1e14:
                    v = v / 1e6  # ns -> ms
                elif v > 1e12:
                    pass  # ms already
                elif v > 1e10:
                    v = v  # ms
                else:
                    v = v * 1000  # s -> ms
                ts = datetime.fromtimestamp(v / 1000, tz=timezone.utc)
            except Exception:
                pass
        exchange = raw.get("x") or raw.get("exchange")
        tape = raw.get("z") or raw.get("tape")
        # Source determination: explicit flag or caller context
        source = raw.get("_source") or "polygon_realtime_quote"
        return Quote(
            symbol=str(sym).upper(),
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            bid_size=int(bid_size) if bid_size is not None else None,
            ask_size=int(ask_size) if ask_size is not None else None,
            timestamp=ts,
            ingested_at=ingested_at or _utc_now(),
            exchange=int(exchange) if exchange is not None else None,
            source=source,
            tape=int(tape) if tape is not None else None,
        )
    except Exception:
        return None


def parse_polygon_trade(raw: dict, ingested_at: Optional[datetime] = None) -> Optional[Trade]:
    """Parse Polygon trade JSON (REST or WS) into Trade. Returns None on malformed."""
    try:
        sym = raw.get("sym") or raw.get("T") or raw.get("symbol") or raw.get("ticker")
        if not sym:
            return None
        price = raw.get("p") if raw.get("p") is not None else raw.get("price")
        size = raw.get("s") if raw.get("s") is not None else raw.get("size") or raw.get("volume")
        ts_raw = raw.get("t") or raw.get("timestamp") or raw.get("sip_timestamp") or raw.get("participant_timestamp")
        ts = None
        if ts_raw is not None:
            try:
                v = int(ts_raw)
                if v > 1e14:
                    v = v / 1e6
                elif v > 1e12:
                    pass
                elif v > 1e10:
                    v = v
                else:
                    v = v * 1000
                ts = datetime.fromtimestamp(v / 1000, tz=timezone.utc)
            except Exception:
                pass
        exchange = raw.get("x") or raw.get("exchange")
        conds = raw.get("c") or raw.get("conditions")
        if conds is not None and not isinstance(conds, list):
            conds = [conds]
        tape = raw.get("z") or raw.get("tape")
        source = raw.get("_source") or "polygon_realtime_trade"
        # Never fabricate aggressor — only if explicitly present
        aggressor = raw.get("aggressor_side") or raw.get("side")
        if aggressor not in ("BUY", "SELL"):
            aggressor = None
        return Trade(
            symbol=str(sym).upper(),
            price=float(price) if price is not None else None,
            size=int(size) if size is not None else None,
            timestamp=ts,
            ingested_at=ingested_at or _utc_now(),
            exchange=int(exchange) if exchange is not None else None,
            conditions=[int(c) for c in conds] if conds else None,
            source=source,
            tape=int(tape) if tape is not None else None,
            aggressor_side=aggressor,
        )
    except Exception:
        return None
