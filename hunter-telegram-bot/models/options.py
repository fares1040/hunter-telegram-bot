"""Options models. The system distinguishes observable chain data from inferred flow."""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List


@dataclass
class OptionContract:
    ticker: str
    contract_symbol: str
    contract_type: str  # CALL / PUT
    strike: float
    expiration: date
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    source: str = "unknown"

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.bid >= 0 and self.ask >= self.bid:
            return round((self.bid + self.ask) / 2, 4)
        return self.last

    @property
    def premium_volume(self) -> Optional[float]:
        if self.volume is None or self.mid is None:
            return None
        return float(self.volume * self.mid * 100)

    def moneyness(self, underlying_price: Optional[float]) -> Optional[float]:
        if not underlying_price or underlying_price <= 0:
            return None
        return round((self.strike / underlying_price - 1.0) * 100.0, 2)

    @property
    def spread_pct(self) -> Optional[float]:
        if self.bid is None or self.ask is None or self.mid in (None, 0):
            return None
        return round((self.ask - self.bid) / self.mid * 100.0, 2)


@dataclass
class OptionsSnapshot:
    ticker: str
    underlying_price: Optional[float] = None
    contracts: List[OptionContract] = field(default_factory=list)
    source: str = "none"
    timestamp: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.contracts)


@dataclass
class OptionsFlowProfile:
    call_volume: int = 0
    put_volume: int = 0
    call_open_interest: int = 0
    put_open_interest: int = 0
    call_premium: float = 0.0
    put_premium: float = 0.0
    put_call_volume_ratio: Optional[float] = None
    put_call_premium_ratio: Optional[float] = None
    flow_score: int = 50
    bias: str = "NEUTRAL"
    confidence: int = 0
    contract_candidate: Optional[OptionContract] = None
    notes: List[str] = field(default_factory=list)
    source: str = "none"
    inferred: bool = False
