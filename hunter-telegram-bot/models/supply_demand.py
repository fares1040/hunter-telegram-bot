"""Hunter Bot — Supply & Demand Intelligence Models (RR19).

Deterministic, explainable supply/demand zone detection from real market data.
Zones are structural areas where price is expected to react based on real
volume/order-flow evidence. Zones are zones, never single magical prices.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import field as dc_field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ZoneType(Enum):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"


class ZoneFreshness(Enum):
    FRESH = "FRESH"
    TESTED = "TESTED"
    WEAKENED = "WEAKENED"
    INVALIDATED = "INVALIDATED"
    UNKNOWN = "UNKNOWN"


class ZoneStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    UNKNOWN = "UNKNOWN"


class Timeframe(Enum):
    MONTHLY = "MONTHLY"
    WEEKLY = "WEEKLY"
    DAILY = "DAILY"
    INTRADAY = "INTRADAY"


@dataclass
class ZoneEvidence:
    """Structured evidence for a supply/demand zone."""
    base_formation: Optional[str] = None          # "base_before_bullish_departure" etc.
    departure_strength: Optional[float] = None    # pct move from base
    departure_bars: Optional[int] = None          # bars to leave base
    volume_confirmation: Optional[float] = None   # volume ratio on departure
    retest_count: int = 0
    reaction_quality: Optional[str] = None        # "strong_bounce" / "weak_reaction" / None
    structural_confirmation: List[str] = field(default_factory=list)  # e.g. ["swing_low", "vwap"]
    missing: List[str] = field(default_factory=list)


@dataclass
class SupplyDemandZone:
    """A real supply or demand zone where price is expected to react."""
    zone_low: float
    zone_high: float
    zone_type: ZoneType                   # DEMAND / SUPPLY
    timeframe: Timeframe
    strength: ZoneStrength = ZoneStrength.UNKNOWN
    freshness: ZoneFreshness = ZoneFreshness.UNKNOWN
    source: str = "unknown"               # provenance: yfinance_history / polygon / etc.

    # Core metrics
    base_low: Optional[float] = None
    base_high: Optional[float] = None
    departure_price: Optional[float] = None
    departure_pct: Optional[float] = None
    departure_bars: Optional[int] = None

    # Volume metrics
    volume_on_departure: Optional[float] = None
    avg_volume_in_base: Optional[float] = None
    volume_ratio_on_departure: Optional[float] = None

    # Retest tracking
    retest_count: int = 0
    last_retest_price: Optional[float] = None
    last_retest_time: Optional[datetime] = None
    retest_reaction_quality: Optional[str] = None  # "strong_bounce" / "weak_reaction" / None

    # Freshness state
    freshness: ZoneFreshness = ZoneFreshness.UNKNOWN
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None

    # Evidence and quality
    evidence: ZoneEvidence = field(default_factory=ZoneEvidence)
    strength: ZoneStrength = ZoneStrength.UNKNOWN
    confidence: int = 0  # 0-100

    # Metadata
    detected_at: datetime = _utc_now()
    last_updated: datetime = _utc_now()
    data_quality: str = "REAL"  # REAL / PARTIAL / MISSING
    evidence_summary: List[str] = field(default_factory=list)

    @property
    def zone_mid(self) -> float:
        return (self.zone_low + self.zone_high) / 2

    @property
    def zone_height(self) -> float:
        return self.zone_high - self.zone_low

    @property
    def zone_height_pct(self) -> float:
        if self.zone_mid == 0:
            return 0.0
        return (self.zone_height / self.zone_mid) * 100

    @property
    def is_demand(self) -> bool:
        return self.zone_type == ZoneType.DEMAND

    @property
    def is_supply(self) -> bool:
        return self.zone_type == ZoneType.SUPPLY

    def distance_from_price(self, current_price: float) -> Optional[float]:
        """Distance from current price to nearest zone edge (pct)."""
        if current_price is None:
            return None
        if self.is_demand:
            # For demand, distance to zone_low (below price = negative)
            return ((self.zone_low - self.zone_mid) / self.zone_mid) * 100 if self.zone_mid else None
        else:
            # For supply, distance to zone_high (above price = positive)
            return ((self.zone_high - self.zone_mid) / self.zone_mid) * 100 if self.zone_mid else None


@dataclass
class ZoneCluster:
    """A cluster of overlapping/adjacent zones from multiple timeframes."""
    zones: List['SupplyDemandZone'] = field(default_factory=list)
    cluster_type: ZoneType = ZoneType.DEMAND
    zone_low: float = 0.0
    zone_high: float = 0.0
    timeframes: List[str] = field(default_factory=list)
    combined_strength: ZoneStrength = ZoneStrength.UNKNOWN
    combined_confidence: int = 0
    alignment: str = "ALIGNED"  # ALIGNED / CONFLICTING / NEUTRAL
    evidence: List[str] = field(default_factory=list)

    @property
    def zone_mid(self) -> float:
        if not self.zones:
            return 0.0
        return sum(z.zone_mid for z in self.zones) / len(self.zones)


@dataclass
class ZoneScoreComponent:
    """One weighted component of the zone quality score."""
    name: str
    weight: float
    value: Optional[int] = None
    reason: Optional[str] = None
    available: bool = False


@dataclass
class SupplyDemandScore:
    """Overall supply/demand zone quality score (0-100)."""
    total: int = 0
    components: List['ZoneScoreComponent'] = field(default_factory=list)

    @property
    def available_weight(self) -> float:
        return sum(c.weight for c in self.components if c.value is not None)


@dataclass
class SupplyDemandResult:
    """Complete output of SupplyDemandEngine.build()."""
    demand_zones: List['SupplyDemandZone'] = field(default_factory=list)
    supply_zones: List['SupplyDemandZone'] = field(default_factory=list)
    demand_clusters: List['ZoneCluster'] = field(default_factory=list)
    supply_clusters: List['ZoneCluster'] = field(default_factory=list)

    # Nearest zones to current price
    nearest_demand: Optional['SupplyDemandZone'] = None
    nearest_supply: Optional['SupplyDemandZone'] = None

    # Multi-timeframe alignment
    mtf_demand_clusters: List['ZoneCluster'] = field(default_factory=list)
    mtf_supply_clusters: List['ZoneCluster'] = field(default_factory=list)
    conflicting_zones: List[str] = field(default_factory=list)  # descriptions of conflicts

    # Overall assessment
    dominant_zone_type: Optional[str] = None  # "DEMAND" / "SUPPLY" / None
    nearest_zone_distance_pct: Optional[float] = None
    dominant_zone_strength: Optional[str] = None

    # Evidence and quality
    missing_data: List[str] = field(default_factory=list)
    data_quality: str = "REAL"
    evidence: List[str] = field(default_factory=list)
