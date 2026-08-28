"""Hunter Decision 2.0 models — additive, no fabrication."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

def _utc_now(): return datetime.now(timezone.utc)

@dataclass
class DecisionEvidence:
    name: str
    value: Optional[str] = None
    direction: str = "UNKNOWN"  # BULLISH/BEARISH/NEUTRAL/UNKNOWN
    quality: str = "UNKNOWN"  # REAL/INFERENCE/UNKNOWN/MISSING/STALE/UNAVAILABLE
    source: str = "unknown"
    freshness: str = "UNKNOWN"
    description: str = ""

@dataclass
class WhyNow:
    status: str = "UNKNOWN"  # CONFIRMED/PARTIAL/UNKNOWN
    explanation: str = "INSUFFICIENT_EVIDENCE"
    catalyst_timestamp: Optional[datetime] = None
    reaction_timestamp: Optional[datetime] = None
    elapsed_seconds: Optional[int] = None
    temporal_overlap: bool = False
    supporting: List[str] = field(default_factory=list)
    freshness: str = "UNKNOWN"
    provenance: str = "unknown"

@dataclass
class Conflict:
    type: str
    severity: str = "MEDIUM"  # LOW/MEDIUM/HIGH
    description: str = ""
    bullish_evidence: Optional[str] = None
    bearish_evidence: Optional[str] = None
    provenance: str = "unknown"

@dataclass
class Conviction:
    score: int = 0  # 0-100
    level: str = "INSUFFICIENT"  # HIGH/MEDIUM/LOW/INSUFFICIENT
    alignment_score: int = 0
    quality_score: int = 0
    freshness_score: int = 0
    conflict_penalty: int = 0
    completeness_score: int = 0
    rationale: str = ""

@dataclass
class OpportunityQuality:
    score: int = 0
    tier: str = "UNAVAILABLE"  # HIGH_QUALITY/ACTIONABLE/INTERESTING/LOW_QUALITY/UNAVAILABLE
    risk_adjusted_score: int = 0
    risk_valid: bool = False
    trap_risk: int = 0
    rationale: str = ""

@dataclass
class DecisionRationale:
    supporting: List[DecisionEvidence] = field(default_factory=list)
    conflicting: List[Conflict] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    why_now: WhyNow = field(default_factory=WhyNow)
    conviction: Conviction = field(default_factory=Conviction)
    opportunity_quality: OpportunityQuality = field(default_factory=OpportunityQuality)
    summary: str = ""
    timestamp: datetime = field(default_factory=_utc_now)
