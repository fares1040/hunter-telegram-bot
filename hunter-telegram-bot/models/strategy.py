"""Abu Rakan Strategy Intelligence Models (RR22).

Deterministic strategy evidence layer based on documented Abu Rakan / PODC methodology.
Translates strategy concepts into measurable, deterministic market-data rules.
All missing data remains UNKNOWN/UNAVAILABLE - never fabricated.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrategyState(Enum):
    CONFIRMED = "CONFIRMED"
    DEVELOPING = "DEVELOPING"
    WATCH = "WATCH"
    INVALIDATED = "INVALIDATED"
    UNAVAILABLE = "UNAVAILABLE"


class StrategyDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ConfirmationState(Enum):
    CONFIRMED = "CONFIRMED"
    DEVELOPING = "DEVELOPING"
    WATCH = "WATCH"
    UNCONFIRMED = "UNCONFIRMED"
    UNAVAILABLE = "UNAVAILABLE"


class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNACCEPTABLE = "UNACCEPTABLE"


@dataclass
class StrategyEvidence:
    """Complete evidence package for strategy decision."""
    monthly_structure: Optional[str] = None
    weekly_structure: Optional[str] = None
    daily_structure: Optional[str] = None
    higher_timeframe_demand: Optional[bool] = None
    higher_timeframe_supply: Optional[bool] = None
    breakout_expansion: Optional[bool] = None
    pullback_toward_demand: Optional[bool] = None
    volume_confirmation: Optional[bool] = None
    supply_overhead: Optional[bool] = None
    retest_behavior: Optional[str] = None
    structure_preservation: Optional[bool] = None
    risk_invalidation_clear: Optional[bool] = None
    reward_potential: Optional[str] = None
    confirmation_quality: str = "UNAVAILABLE"
    missing: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    source: str = "unknown"               # provenance: strategy_engine / rr22


@dataclass
class StrategyEntry:
    """Structured entry intelligence."""
    status: str = "UNAVAILABLE"
    reason: Optional[str] = None
    direction: Optional[str] = None
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    invalidation_price: Optional[float] = None
    invalidation_basis: Optional[str] = None
    risk_distance_abs: Optional[float] = None
    risk_distance_pct: Optional[float] = None
    confirmation_quality: str = "UNAVAILABLE"
    confirmations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class StrategyTarget:
    """Strategy target zones."""
    status: str = "UNAVAILABLE"
    tp1_low: Optional[float] = None
    tp1_high: Optional[float] = None
    tp2_low: Optional[float] = None
    tp2_high: Optional[float] = None
    tp3_low: Optional[float] = None
    tp3_high: Optional[float] = None
    risk_reward: Optional[float] = None
    confidence: str = "UNAVAILABLE"
    evidence: List[str] = field(default_factory=list)


@dataclass
class StrategyRisk:
    """Risk assessment."""
    status: str = "UNAVAILABLE"
    flags: List[str] = field(default_factory=list)
    risk_level: str = "UNKNOWN"
    invalidation_clear: bool = False
    invalidation_price: Optional[float] = None
    invalidation_basis: Optional[str] = None
    risk_reward_ratio: Optional[float] = None
    risk_acceptable: bool = False


@dataclass
class StrategyResult:
    """Complete Abu Rakan strategy analysis result."""
    ticker: str
    as_of: datetime

    # Strategy state
    state: str = "UNAVAILABLE"
    direction: Optional[str] = None
    confirmation: str = "UNAVAILABLE"

    # Evidence (authoritative field - used by both StrategyEngine and DecisionEngine)
    evidence: StrategyEvidence = field(default_factory=StrategyEvidence)

    # Entry / Target / Risk
    entry: StrategyEntry = field(default_factory=StrategyEntry)
    target: StrategyTarget = field(default_factory=StrategyTarget)
    risk: StrategyRisk = field(default_factory=StrategyRisk)

    # Quality and confidence
    confidence: int = 0
    data_quality: str = "UNKNOWN"
    data_quality_note: str = ""

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return (
            self.state == "CONFIRMED" and
            self.confirmation == "CONFIRMED" and
            self.risk.invalidation_clear
        )

    def summary(self) -> str:
        parts = [
            f"{self.ticker} [{self.as_of.strftime('%Y-%m-%d')}]",
            f"State: {self.state}",
            f"Direction: {self.direction or 'NONE'}",
            f"Confirmation: {self.confirmation}",
            f"Confidence: {self.confidence}/100",
            f"Data Quality: {self.data_quality}",
        ]
        if self.evidence.missing:
            parts.append(f"Missing: {', '.join(self.evidence.missing)}")
        if self.warnings:
            parts.append(f"Warnings: {'; '.join(self.warnings)}")
        return " | ".join(parts)
