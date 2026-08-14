"""Hunter Bot — Session-aware data models"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from core.session_clock import MarketSession


@dataclass
class SessionSnapshot:
    session_type: MarketSession
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return all([
            self.high is not None,
            self.low is not None,
            self.volume is not None,
        ])

    @property
    def range_pct(self) -> Optional[float]:
        if self.low and self.high and self.low > 0:
            return round(((self.high - self.low) / self.low) * 100, 2)
        return None
