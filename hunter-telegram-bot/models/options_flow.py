"""Options Flow Intelligence Models (RR21).

Deterministic options flow analysis from real market data.
Separates observable chain data from inferred flow intelligence.
All missing data remains UNKNOWN/UNAVAILABLE - never fabricated.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import field as dc_field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OptionsDataQuality(Enum):
    REAL = "REAL"
    PROXY = "PROXY"
    MISSING = "MISSING"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class OptionsFlowBias(Enum):
    STRONG_CALL = "STRONG_CALL"
    CALL_LEAN = "CALL_LEAN"
    NEUTRAL = "NEUTRAL"
    PUT_LEAN = "PUT_LEAN"
    STRONG_PUT = "STRONG_PUT"
    UNKNOWN = "UNKNOWN"


class OptionsDataFreshness(Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


@dataclass
class OptionChainMetrics:
    """Aggregated metrics from a real options chain."""
    call_volume: int = 0
    put_volume: int = 0
    call_open_interest: int = 0
    put_open_interest: int = 0
    call_premium_volume: float = 0.0
    put_premium_volume: float = 0.0
    put_call_volume_ratio: Optional[float] = None
    put_call_oi_ratio: Optional[float] = None
    put_call_premium_ratio: Optional[float] = None
    unusual_volume_strikes: List[Dict[str, Any]] = field(default_factory=list)
    high_oi_strikes: List[Dict[str, Any]] = field(default_factory=list)
    expiration_concentration: Dict[str, int] = field(default_factory=dict)
    strike_concentration: Dict[str, int] = field(default_factory=dict)
    iv_skew: Optional[float] = None
    atm_iv: Optional[float] = None
    iv_skew_slope: Optional[float] = None
    bid_ask_spread_quality: Optional[float] = None
    unusual_volume_detected: bool = False
    unusual_oi_detected: bool = False
    data_quality: str = "REAL"
    missing_fields: List[str] = field(default_factory=list)


@dataclass
class OptionsFlowIntelligence:
    """Structured options flow intelligence from real chain data."""
    ticker: str = "UNKNOWN"
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Chain metadata
    chain_source: str = "unknown"
    chain_timestamp: Optional[datetime] = None
    underlying_price: Optional[float] = None

    # Chain metrics
    metrics: Optional['OptionChainMetrics'] = None

    # Flow analysis
    bias: str = "NEUTRAL"
    flow_score: int = 0
    bias_confidence: int = 0

    # Best contract candidate
    contract_candidate: Optional['OptionContract'] = None

    # Freshness and quality
    freshness: str = "UNKNOWN"  # FRESH / STALE / UNKNOWN
    data_quality: str = "UNKNOWN"  # REAL / PROXY / MISSING / STALE / UNAVAILABLE
    chain_age_minutes: Optional[int] = None

    # Best contract candidate
    contract_candidate: Optional['OptionContract'] = None

    # Notes and warnings
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_fresh(self) -> bool:
        if self.chain_age_minutes is None:
            return False
        return self.chain_age_minutes <= 60

    @property
    def has_reliable_chain(self) -> bool:
        if self.data_quality in ("REAL", "PROXY"):
            return True
        return False


@dataclass
class OptionsFlowComponent:
    name: str
    weight: float
    value: Optional[int] = None
    reason: Optional[str] = None
    available: bool = False


@dataclass
class OptionsFlowScore:
    total: int = 0
    components: List['OptionsFlowComponent'] = field(default_factory=list)

    @property
    def available_weight(self) -> float:
        return sum(c.weight for c in self.components if c.value is not None)

    @property
    def is_renormalized(self) -> bool:
        return self.available_weight > 0 and abs(self.available_weight - 1.0) > 1e-9


class OptionsDataFreshness(Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


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
    contract_candidate: Optional['OptionContract'] = None
    notes: List[str] = field(default_factory=list)
    source: str = "none"
    inferred: bool = False


from models.options import OptionContract, OptionsSnapshot
from models.technical import TechnicalIntelligence
from models.intraday import IntradayIntelligence
from models.swing import SwingIntelligence
from models.target import TargetResult
from models.risk import RiskPlan
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from models.risk import RiskPlan
from models.ticker import TickerData
from models.news import CatalystEvent
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from engines.technical_engine import TechnicalEngine
from models.technical import TechnicalIntelligence
from models.intraday import IntradayIntelligence
from models.swing import SwingIntelligence
from models.target import TargetResult
from models.risk import RiskPlan