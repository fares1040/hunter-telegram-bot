"""Hunter Bot — Target Intelligence data models (Phase 2.10).

Targets are always zones (zone_low / zone_high), never single magical prices.
A TargetResult is returned by TargetEngine.build() and consumed downstream.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TargetZone:
    """A real price zone where price is expected to react (take-profit area)."""
    zone_low: float
    zone_high: float
    source: str                 # human-readable source label
    source_type: str            # structural source category
    distance: float             # price distance from entry (positive away from entry)
    distance_pct: float         # % distance from entry
    confidence: str             # VALID / WEAK / UNAVAILABLE
    quality: str                # VALID / WEAK / UNAVAILABLE
    evidence: str               # explainable reasoning
    timeframe: str = "1d"
    direction: str = "LONG"     # LONG (target above) / SHORT (target below)


@dataclass
class Target:
    """A single take-profit target (TP1/TP2/TP3) as a zone."""
    tp_id: str                  # TP1 / TP2 / TP3
    zone: TargetZone
    status: str = "UNAVAILABLE"
    reason: Optional[str] = None


@dataclass
class TargetScoreComponent:
    """One weighted component of the target quality score/confidence."""
    name: str
    weight: float
    value: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class TargetScore:
    """Overall target quality score (0-100), renormalized over available parts."""
    total: int = 0
    components: List[TargetScoreComponent] = field(default_factory=list)


@dataclass
class TargetConfidence:
    """Separate confidence metric (0-100) reflecting trust in the targets."""
    value: int = 0
    components: List[TargetScoreComponent] = field(default_factory=list)


@dataclass
class TargetResult:
    """Full output of TargetEngine.build()."""
    tp1: Optional[Target] = None
    tp2: Optional[Target] = None
    tp3: Optional[Target] = None
    entry: Optional[float] = None
    invalidation: Optional[float] = None
    direction: str = "LONG"
    status: str = "UNAVAILABLE"          # READY / WATCH / UNAVAILABLE
    risk_reward: Optional[float] = None
    score: Optional[TargetScore] = None
    confidence: Optional[TargetConfidence] = None
    evidence: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)
