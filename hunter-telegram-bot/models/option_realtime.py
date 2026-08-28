"""True Flow options event contracts — OptionQuote / OptionTrade + aggregation.

No fabrication. Aggressor/sweep/opening never inferred.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, List, Dict, Any
import math


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _norm(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

@dataclass
class OptionQuote:
    underlying: str
    contract: str  # OCC symbol
    expiry: Optional[date] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # CALL/PUT
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    timestamp: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=_utc_now)
    exchange: Optional[int] = None
    source: str = "unknown"
    conditions: Optional[List[int]] = None

    def __post_init__(self):
        self.timestamp = _norm(self.timestamp)
        self.ingested_at = _norm(self.ingested_at) or _utc_now()

    @property
    def is_valid(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid >= 0 and self.ask >= self.bid

    @property
    def spread(self) -> Optional[float]:
        if self.is_valid:
            return round(self.ask - self.bid, 4)
        return None

    @property
    def mid(self) -> Optional[float]:
        if self.is_valid:
            return round((self.bid + self.ask) / 2, 4)
        return None

    @property
    def latency_ms(self) -> Optional[int]:
        if self.timestamp is None:
            return None
        return max(0, int((self.ingested_at - self.timestamp).total_seconds() * 1000))

    def freshness(self, max_age: int = 30) -> str:
        if self.timestamp is None:
            return "UNKNOWN"
        age = (self.ingested_at - self.timestamp).total_seconds()
        if age < 0:
            return "FRESH"
        return "FRESH" if age <= max_age else "STALE"

    @property
    def is_realtime(self) -> bool:
        return self.source in ("polygon_options_realtime_quote", "polygon_options_quote_ws")

    def dedup_key(self) -> str:
        ts = self.timestamp.isoformat() if self.timestamp else "no-ts"
        return f"OQ:{self.contract}:{ts}:{self.bid}:{self.ask}:{self.bid_size}:{self.ask_size}:{self.exchange}"

@dataclass
class OptionTrade:
    underlying: str
    contract: str
    expiry: Optional[date] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None
    price: Optional[float] = None
    size: Optional[int] = None
    timestamp: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=_utc_now)
    exchange: Optional[int] = None
    conditions: Optional[List[int]] = None
    source: str = "unknown"
    aggressor_side: Optional[str] = None  # BUY/SELL only if REAL

    def __post_init__(self):
        self.timestamp = _norm(self.timestamp)
        self.ingested_at = _norm(self.ingested_at) or _utc_now()

    @property
    def is_valid(self) -> bool:
        return self.price is not None and self.price > 0 and self.size is not None and self.size > 0

    @property
    def premium(self) -> Optional[float]:
        if self.price is not None and self.size is not None:
            try:
                return round(self.price * self.size * 100, 2)
            except Exception:
                return None
        return None

    @property
    def latency_ms(self) -> Optional[int]:
        if self.timestamp is None:
            return None
        return max(0, int((self.ingested_at - self.timestamp).total_seconds() * 1000))

    def freshness(self, max_age: int = 30) -> str:
        if self.timestamp is None:
            return "UNKNOWN"
        age = (self.ingested_at - self.timestamp).total_seconds()
        if age < 0:
            return "FRESH"
        return "FRESH" if age <= max_age else "STALE"

    @property
    def is_realtime(self) -> bool:
        return self.source in ("polygon_options_realtime_trade", "polygon_options_trade_ws")

    def dedup_key(self) -> str:
        ts = self.timestamp.isoformat() if self.timestamp else "no-ts"
        conds = ",".join(str(c) for c in (self.conditions or []))
        return f"OT:{self.contract}:{ts}:{self.price}:{self.size}:{self.exchange}:{conds}"

@dataclass
class OptionRealtimeBuffer:
    symbol: str  # underlying
    max_events: int = 1000
    quotes: List[OptionQuote] = field(default_factory=list)
    trades: List[OptionTrade] = field(default_factory=list)
    _q_keys: set = field(default_factory=set, repr=False)
    _t_keys: set = field(default_factory=set, repr=False)

    def add_quote(self, q: OptionQuote) -> bool:
        k = q.dedup_key()
        if k in self._q_keys:
            return False
        self._q_keys.add(k)
        self.quotes.append(q)
        self.quotes.sort(key=lambda x: x.timestamp or datetime.max.replace(tzinfo=timezone.utc))
        if len(self.quotes) > self.max_events:
            self._q_keys.discard(self.quotes.pop(0).dedup_key())
        if len(self._q_keys) > self.max_events * 2:
            self._q_keys = {x.dedup_key() for x in self.quotes}
        return True

    def add_trade(self, t: OptionTrade) -> bool:
        k = t.dedup_key()
        if k in self._t_keys:
            return False
        self._t_keys.add(k)
        self.trades.append(t)
        self.trades.sort(key=lambda x: x.timestamp or datetime.max.replace(tzinfo=timezone.utc))
        if len(self.trades) > self.max_events:
            self._t_keys.discard(self.trades.pop(0).dedup_key())
        if len(self._t_keys) > self.max_events * 2:
            self._t_keys = {x.dedup_key() for x in self.trades}
        return True

    def fresh_trades(self, max_age: int = 30) -> List[OptionTrade]:
        return [t for t in self.trades if t.freshness(max_age) == "FRESH" and t.is_valid]

    def fresh_quotes(self, max_age: int = 30) -> List[OptionQuote]:
        return [q for q in self.quotes if q.freshness(max_age) == "FRESH" and q.is_valid]

@dataclass
class TrueFlowMetrics:
    """Pure aggregation of REAL option trades — never mixed with snapshot volume."""
    underlying: str
    total_trades: int = 0
    call_trades: int = 0
    put_trades: int = 0
    call_volume: int = 0
    put_volume: int = 0
    call_premium: float = 0.0
    put_premium: float = 0.0
    largest_premium: float = 0.0
    largest_contract: Optional[str] = None
    repeated_contracts: List[str] = field(default_factory=list)
    by_contract: Dict[str, int] = field(default_factory=dict)
    by_expiry: Dict[str, int] = field(default_factory=dict)
    by_strike: Dict[str, int] = field(default_factory=dict)
    large_prints: List[OptionTrade] = field(default_factory=list)  # size/premium threshold
    stale_excluded: int = 0
    source: str = "polygon_options_realtime_trade"

def aggregate_true_flow(trades: List[OptionTrade], max_age: int = 30, large_size: int = 100, large_premium: float = 50000) -> TrueFlowMetrics:
    if not trades:
        return TrueFlowMetrics(underlying="UNKNOWN")
    underlying = trades[0].underlying if trades else "UNKNOWN"
    m = TrueFlowMetrics(underlying=underlying)
    # dedup already done by buffer; freshness filter
    fresh: List[OptionTrade] = []
    for t in trades:
        if t.freshness(max_age) != "FRESH" or not t.is_valid:
            m.stale_excluded += 1
            continue
        fresh.append(t)
    m.total_trades = len(fresh)
    counts: Dict[str, int] = {}
    for t in fresh:
        is_call = t.option_type == "CALL"
        is_put = t.option_type == "PUT"
        if is_call:
            m.call_trades += 1
            m.call_volume += t.size or 0
            m.call_premium += t.premium or 0
        elif is_put:
            m.put_trades += 1
            m.put_volume += t.size or 0
            m.put_premium += t.premium or 0
        # largest
        prem = t.premium or 0
        if prem > m.largest_premium:
            m.largest_premium = prem
            m.largest_contract = t.contract
        # concentration
        m.by_contract[t.contract] = m.by_contract.get(t.contract, 0) + 1
        if t.expiry:
            m.by_expiry[str(t.expiry)] = m.by_expiry.get(str(t.expiry), 0) + 1
        if t.strike is not None:
            m.by_strike[str(t.strike)] = m.by_strike.get(str(t.strike), 0) + 1
        # large prints (evidence-based, not "block" claim)
        if (t.size and t.size >= large_size) or (prem >= large_premium):
            m.large_prints.append(t)
    # repeated prints: contract appears >=3 times
    m.repeated_contracts = [c for c, n in m.by_contract.items() if n >= 3]
    return m

def parse_polygon_option_quote(raw: dict, ingested_at: Optional[datetime] = None) -> Optional[OptionQuote]:
    try:
        sym = raw.get("sym") or raw.get("T") or raw.get("ticker") or raw.get("contract")
        if not sym:
            return None
        # Polygon option quote fields: bp/ap/bs/as or bid/ask
        bid = raw.get("bp") if raw.get("bp") is not None else raw.get("bid")
        ask = raw.get("ap") if raw.get("ap") is not None else raw.get("ask")
        bid_size = raw.get("bs") if raw.get("bs") is not None else raw.get("bid_size")
        ask_size = raw.get("as") if raw.get("as") is not None else raw.get("ask_size")
        ts_raw = raw.get("t") or raw.get("timestamp") or raw.get("sip_timestamp")
        ts = None
        if ts_raw is not None:
            try:
                v = int(ts_raw)
                if v > 1e14: v = v / 1e6
                ts = datetime.fromtimestamp(v/1000, tz=timezone.utc)
            except Exception:
                pass
        # Derive underlying/type/strike/expiry from contract if possible
        underlying = raw.get("underlying") or raw.get("underlying_ticker") or ""
        if not underlying:
            # Try to parse OCC: e.g. AAPL250117C00150000
            s = str(sym)
            # Heuristic: letters prefix is underlying
            import re
            mm = re.match(r"^([A-Z]+)", s)
            if mm:
                underlying = mm.group(1)
        # Expiry/strike/type optionally in raw
        expiry = raw.get("expiry") or raw.get("expiration_date")
        exp_date = None
        if expiry:
            try:
                exp_date = date.fromisoformat(str(expiry)[:10])
            except Exception:
                pass
        strike = _safe_float(raw.get("strike") or raw.get("strike_price"))
        otype = raw.get("option_type") or raw.get("contract_type") or raw.get("type")
        if otype:
            otype = str(otype).upper()
            if otype not in ("CALL", "PUT"):
                otype = None
        return OptionQuote(
            underlying=str(underlying).upper() or str(sym).upper(),
            contract=str(sym),
            expiry=exp_date,
            strike=strike,
            option_type=otype,
            bid=_safe_float(bid),
            ask=_safe_float(ask),
            bid_size=int(bid_size) if bid_size is not None else None,
            ask_size=int(ask_size) if ask_size is not None else None,
            timestamp=ts,
            ingested_at=ingested_at or _utc_now(),
            exchange=int(raw.get("x") or raw.get("exchange")) if raw.get("x") is not None or raw.get("exchange") is not None else None,
            source=raw.get("_source") or "polygon_options_realtime_quote",
            conditions=raw.get("c"),
        )
    except Exception:
        return None

def parse_polygon_option_trade(raw: dict, ingested_at: Optional[datetime] = None) -> Optional[OptionTrade]:
    try:
        sym = raw.get("sym") or raw.get("T") or raw.get("ticker") or raw.get("contract")
        if not sym:
            return None
        price = raw.get("p") if raw.get("p") is not None else raw.get("price")
        size = raw.get("s") if raw.get("s") is not None else raw.get("size")
        ts_raw = raw.get("t") or raw.get("timestamp") or raw.get("sip_timestamp")
        ts = None
        if ts_raw is not None:
            try:
                v = int(ts_raw)
                if v > 1e14: v = v / 1e6
                ts = datetime.fromtimestamp(v/1000, tz=timezone.utc)
            except Exception:
                pass
        underlying = raw.get("underlying") or raw.get("underlying_ticker") or ""
        if not underlying:
            import re
            mm = re.match(r"^([A-Z]+)", str(sym))
            if mm:
                underlying = mm.group(1)
        expiry = raw.get("expiry") or raw.get("expiration_date")
        exp_date = None
        if expiry:
            try:
                exp_date = date.fromisoformat(str(expiry)[:10])
            except Exception:
                pass
        strike = _safe_float(raw.get("strike") or raw.get("strike_price"))
        otype = raw.get("option_type") or raw.get("contract_type")
        if otype:
            otype = str(otype).upper()
            if otype not in ("CALL", "PUT"):
                otype = None
        conds = raw.get("c") or raw.get("conditions")
        if conds is not None and not isinstance(conds, list):
            conds = [conds]
        aggressor = raw.get("aggressor_side") or raw.get("side")
        if aggressor not in ("BUY", "SELL"):
            aggressor = None
        return OptionTrade(
            underlying=str(underlying).upper() or str(sym).upper(),
            contract=str(sym),
            expiry=exp_date,
            strike=strike,
            option_type=otype,
            price=_safe_float(price),
            size=int(size) if size is not None else None,
            timestamp=ts,
            ingested_at=ingested_at or _utc_now(),
            exchange=int(raw.get("x") or raw.get("exchange")) if raw.get("x") is not None or raw.get("exchange") is not None else None,
            conditions=[int(c) for c in conds] if conds else None,
            source=raw.get("_source") or "polygon_options_realtime_trade",
            aggressor_side=aggressor,
        )
    except Exception:
        return None
