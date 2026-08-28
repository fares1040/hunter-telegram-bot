"""Hunter Bot — Structured Technical Intelligence Model (Phase 2.7).

Separation of concerns (mirrors catalyst/news model):
- REAL DATA: prices/volumes/VWAP values copied from provider data or history bars
- INFERENCE: trend/momentum/volatility classifications from deterministic rules
- SCORE: explainable 0-100 components with visible weights
- AI EXPLANATION: never stored here

Multi-timeframe readiness: every object carries a ``timeframe`` label
("1d" today; intraday frames may be added later without structural changes).
Insufficient history yields UNKNOWN / UNAVAILABLE values — never fabricated.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class MomentumLabel(Enum):
    STRONG = "STRONG"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    WEAK = "WEAK"


class VolatilityRegime(Enum):
    SQUEEZE = "SQUEEZE"
    EXPANSION = "EXPANSION"
    NORMAL = "NORMAL"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


class VolumeRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class TrendIntelligence:
    timeframe: str = "1d"
    direction: TrendDirection = TrendDirection.UNKNOWN
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    price_vs_ma20_pct: Optional[float] = None
    price_vs_ma50_pct: Optional[float] = None
    price_vs_ma200_pct: Optional[float] = None
    ma_alignment: Optional[str] = None          # e.g. PRICE>MA20>MA50>MA200
    ma20_slope_pct: Optional[float] = None      # 5-day slope of MA20 in %
    ma50_slope_pct: Optional[float] = None
    structure: Optional[str] = None             # HH_HL / LH_LL / MIXED
    structure_evidence: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class MomentumIntelligence:
    rsi: Optional[float] = None
    roc_5: Optional[float] = None               # % change over 5 bars
    roc_10: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    direction: MomentumLabel = MomentumLabel.NEUTRAL
    acceleration: Optional[str] = None          # BUILDING / FADING
    divergence: Optional[str] = None            # only when clearly detectable
    missing: List[str] = field(default_factory=list)


@dataclass
class VolatilityIntelligence:
    atr: Optional[float] = None
    atr_pct: Optional[float] = None             # ATR / price * 100
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct: Optional[float] = None        # (upper-lower)/mid * 100
    width_percentile: Optional[float] = None    # rank of width vs trailing year of widths
    regime: VolatilityRegime = VolatilityRegime.UNKNOWN
    expanding: bool = False
    contracting: bool = False
    missing: List[str] = field(default_factory=list)


@dataclass
class VwapIntelligence:
    vwap: Optional[float] = None                # REAL value only (session snapshot or bar-computed)
    source: Optional[str] = None                # "regular_session" / "premarket" / "intraday_bars"
    price_vs_vwap_pct: Optional[float] = None
    status: str = "UNAVAILABLE"                 # ABOVE / BELOW / UNAVAILABLE
    reclaim: bool = False                       # crossed above within recent window
    rejection: bool = False                     # failed at VWAP then fell
    note: str = ""


@dataclass
class VolumeIntelligence:
    volume: Optional[int] = None                # current session volume
    average_volume: Optional[float] = None      # 20-bar mean from history when available
    rvol: Optional[float] = None
    spike_ratio: Optional[float] = None         # last bar vs trailing avg
    acceleration: Optional[float] = None        # recent-5-bar mean vs prior-5-bar mean
    dollar_volume: Optional[float] = None
    regime: VolumeRegime = VolumeRegime.NORMAL
    missing: List[str] = field(default_factory=list)


@dataclass
class PriceLevel:
    price: float
    level_type: str                             # SUPPORT / RESISTANCE
    strength: int                               # 0-100
    distance_pct: float                         # from current price, negative below
    evidence: str
    source: str = "unknown"                     # provenance: swing_pivot / pd_highlow / vwap / etc.


@dataclass
class SupportResistanceIntelligence:
    levels: List[PriceLevel] = field(default_factory=list)
    nearest_support: Optional[PriceLevel] = None
    nearest_resistance: Optional[PriceLevel] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SetupDetection:
    name: str                                   # BREAKOUT / PULLBACK / VWAP_RECLAIM ...
    direction: str                              # BULLISH / BEARISH / NEUTRAL
    detected: bool = False
    evidence: List[str] = field(default_factory=list)


@dataclass
class ScoreComponent:
    name: str
    weight: float                               # nominal weight (renormalized when unavailable)
    available: bool
    value: Optional[int] = None                 # 0-100 sub-score
    reason: str = ""                            # why unavailable / key drivers


@dataclass
class TechnicalScore:
    total: int = 0                              # 0-100
    components: List[ScoreComponent] = field(default_factory=list)

    def breakdown_lines(self) -> List[str]:
        lines = []
        for c in self.components:
            state = f"{c.value}" if c.available else "N/A (" + c.reason + ")"
            lines.append(f"{c.name}: {state} [weight {c.weight:.0%}]")
        return lines


@dataclass
class TechnicalIntelligence:
    ticker: str
    timeframe: str = "1d"
    generated_at: datetime = field(default_factory=_utc_now)
    current_price: Optional[float] = None       # REAL
    trend: TrendIntelligence = field(default_factory=TrendIntelligence)
    momentum: MomentumIntelligence = field(default_factory=MomentumIntelligence)
    volatility: VolatilityIntelligence = field(default_factory=VolatilityIntelligence)
    vwap: VwapIntelligence = field(default_factory=VwapIntelligence)
    volume: VolumeIntelligence = field(default_factory=VolumeIntelligence)
    support_resistance: SupportResistanceIntelligence = field(default_factory=SupportResistanceIntelligence)
    setups: List[SetupDetection] = field(default_factory=list)
    primary_setup: Optional[SetupDetection] = None
    score: TechnicalScore = field(default_factory=TechnicalScore)
    missing_data: List[str] = field(default_factory=list)   # global honesty ledger

    def summary(self) -> str:
        parts = [f"{self.ticker} [{self.timeframe}] price={self.current_price}"]
        parts.append(f"trend={self.trend.direction.value} momentum={self.momentum.direction.value}"
                     f" vol_regime={self.volatility.regime.value} volume={self.volume.regime.value}")
        if self.primary_setup:
            parts.append(f"setup={self.primary_setup.name}")
        parts.append(f"score={self.score.total}")
        return " | ".join(parts)
