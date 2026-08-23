"""Hunter Bot — News Data Model"""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone
from enum import Enum


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize any datetime to timezone-aware UTC.

    Naive datetimes are assumed to be UTC (same convention as
    NewsItem.age_minutes). None passes through unchanged. This keeps mixed
    naive/aware collections safe to compare, subtract and sort.
    """
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


class SourceTier(Enum):
    TIER_1_OFFICIAL = 1
    TIER_2_MAJOR = 2
    TIER_3_FINANCIAL = 3
    TIER_4_UNVERIFIED = 4


class CatalystType(Enum):
    EARNINGS = "EARNINGS"
    FDA = "FDA"
    CONTRACT = "CONTRACT"
    PARTNERSHIP = "PARTNERSHIP"
    MERGER = "MERGER"
    ACQUISITION = "ACQUISITION"
    GOVERNMENT = "GOVERNMENT"
    PRODUCT = "PRODUCT"
    AI = "AI"
    UPGRADE = "UPGRADE"
    DOWNGRADE = "DOWNGRADE"
    SEC_FILING = "SEC_FILING"
    OFFERING = "OFFERING"
    GUIDANCE = "GUIDANCE"
    DILUTION = "DILUTION"
    BANKRUPTCY = "BANKRUPTCY"
    REGULATORY = "REGULATORY"
    CLINICAL_TRIAL = "CLINICAL_TRIAL"
    OTHER = "OTHER"


@dataclass
class NewsItem:
    id: str
    ticker: str
    headline: str
    source: str
    source_tier: SourceTier
    url: Optional[str] = None
    summary: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=_utc_now)

    @property
    def age_minutes(self) -> Optional[float]:
        if self.published_at:
            if self.published_at.tzinfo is None:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                return (now - self.published_at).total_seconds() / 60.0
            else:
                now = datetime.now(timezone.utc)
                return (now - self.published_at).total_seconds() / 60.0
        return None


@dataclass
class CatalystEvent:
    event_id: str
    ticker: str
    catalyst_type: CatalystType
    headline_summary: str
    primary_source: NewsItem
    additional_sources: List[NewsItem] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=_utc_now)

    source_tier_score: int = 0
    freshness_score: int = 0
    materiality_score: int = 0
    sentiment: str = "NEUTRAL"
    impact_score: int = 0
    priced_in_probability: float = 0.0

    price_before: Optional[float] = None
    price_after: Optional[float] = None
    volume_before: Optional[int] = None
    volume_after: Optional[int] = None

    @property
    def all_sources(self) -> List[NewsItem]:
        return [self.primary_source] + self.additional_sources

    @property
    def best_tier(self) -> SourceTier:
        tiers = [s.source_tier for s in self.all_sources]
        return min(tiers, key=lambda t: t.value)

    def is_fresh(self, max_age_minutes: float = 120.0) -> bool:
        age = self.primary_source.age_minutes
        if age is None:
            return False
        return age <= max_age_minutes
