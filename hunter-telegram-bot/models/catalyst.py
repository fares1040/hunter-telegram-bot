"""Normalized catalyst representation.

Separation of concerns (never mixed):
- REAL DATA: headline/source/published_at/url copied from the provider item
- INFERENCE: category/sentiment/materiality derived by deterministic rules
- SCORE: materiality + its transparent breakdown
- AI EXPLANATION: never stored here; the AI layer only annotates afterwards
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FreshnessBucket(Enum):
    BREAKING = "BREAKING"      # <= 30 min
    RECENT = "RECENT"          # <= 120 min
    AGING = "AGING"            # <= 360 min
    STALE = "STALE"            # > 360 min


class SentimentLabel(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class Recommendation(Enum):
    OPPORTUNITY = "OPPORTUNITY"
    WATCH = "WATCH"
    TRAP_RISK = "TRAP_RISK"
    NEUTRAL = "NEUTRAL"


TRAP_CATEGORIES = {"FINANCING_OFFERING", "DILUTION", "BANKRUPTCY_RESTRUCTURING"}


@dataclass
class CatalystProfile:
    """Deterministic intelligence about one catalyst. AI may explain it later;
    the AI never overwrites these computed values."""
    symbol: str
    headline: str
    source: str
    source_tier_score: int
    published_at: Optional[datetime] = None
    discovered_at: datetime = field(default_factory=_utc_now)
    url: Optional[str] = None

    age_minutes: Optional[float] = None
    freshness: FreshnessBucket = FreshnessBucket.STALE

    category: str = "UNKNOWN"
    sentiment: SentimentLabel = SentimentLabel.UNKNOWN
    classification_confidence: int = 0

    materiality: int = 0
    materiality_breakdown: Dict[str, int] = field(default_factory=dict)

    matched_rules: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    cluster_size: int = 1
    is_trap_risk: bool = False
    trap_reasons: List[str] = field(default_factory=list)
    recommendation: Recommendation = Recommendation.NEUTRAL

    @property
    def has_financial_figures(self) -> bool:
        return "figures" not in self.missing_fields
