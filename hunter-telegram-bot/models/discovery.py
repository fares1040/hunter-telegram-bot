"""Discovery models — structured candidates produced by the Market Discovery Engine."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.session_clock import MarketSession


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DiscoveryCandidate:
    """A single discovered symbol. Only fields supplied by a real source are set;
    anything absent stays None and is listed in missing_fields (never invented)."""
    symbol: str
    sources: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    price: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    discovery_score: int = 0
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=_utc_now)

    def merge_from(self, other: "DiscoveryCandidate") -> None:
        for src in other.sources:
            if src not in self.sources:
                self.sources.append(src)
        for reason in other.reasons:
            if reason not in self.reasons:
                self.reasons.append(reason)

    @property
    def is_enriched(self) -> bool:
        return bool(self.change_percent is not None or self.volume is not None or self.market_cap is not None)


@dataclass
class CandidatePool:
    """Ranked output of one discovery pass. This pool feeds the existing Hunter engines."""
    session: MarketSession
    candidates: List[DiscoveryCandidate] = field(default_factory=list)
    generated_at: datetime = field(default_factory=_utc_now)
    raw_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def symbols(self) -> List[str]:
        return [c.symbol for c in self.candidates]


def merge_entries_to_candidates(entries) -> List[DiscoveryCandidate]:
    """Normalize + deduplicate universe entries into candidates.

    Shared by the discovery engine so provider payloads and engine-level merging
    follow identical rules.
    """
    by_symbol: Dict[str, DiscoveryCandidate] = {}
    for entry in entries:
        cand = by_symbol.get(entry.symbol)
        if cand is None:
            cand = DiscoveryCandidate(
                symbol=entry.symbol,
                sources=[entry.source],
                reasons=[entry.reason],
                price=entry.price,
                change_percent=entry.change_percent,
                volume=entry.volume,
                market_cap=entry.market_cap,
            )
            by_symbol[entry.symbol] = cand
            continue
        cand.merge_from(DiscoveryCandidate(symbol=entry.symbol, sources=[entry.source], reasons=[entry.reason]))
        if cand.price is None:
            cand.price = entry.price
        if cand.change_percent is None:
            cand.change_percent = entry.change_percent
        if cand.volume is None:
            cand.volume = entry.volume
        if cand.market_cap is None:
            cand.market_cap = entry.market_cap
    return list(by_symbol.values())
