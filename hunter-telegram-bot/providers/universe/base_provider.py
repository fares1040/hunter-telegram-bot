"""Hunter Bot — Abstract Market Universe Provider.

A universe provider answers: "which symbols are worth looking at right now?"
It returns raw entries with whatever real metrics its source supplies; it never
fabricates values. The DiscoveryEngine owns normalization, deduplication,
filtering and scoring.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from core.session_clock import MarketSession


@dataclass
class UniverseEntry:
    symbol: str
    source: str
    reason: str = "UNIVERSE_MEMBER"
    price: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None


@dataclass
class UniverseResult:
    source: str
    success: bool = True
    entries: List[UniverseEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MarketUniverseProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def fetch_universe(self, session: MarketSession, limit: int = 25) -> UniverseResult:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass
