"""Hunter Bot — Ticker Data Model"""
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from models.session import SessionSnapshot
from core.session_clock import MarketSession


@dataclass
class TickerData:
    ticker: str
    timestamp: datetime

    previous_close: Optional[float] = None

    premarket: SessionSnapshot = field(default_factory=lambda: SessionSnapshot(session_type=MarketSession.PREMARKET))
    regular: SessionSnapshot = field(default_factory=lambda: SessionSnapshot(session_type=MarketSession.REGULAR))
    after_hours: SessionSnapshot = field(default_factory=lambda: SessionSnapshot(session_type=MarketSession.AFTER_HOURS))

    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    gap_percent: Optional[float] = None

    market_cap: Optional[int] = None
    float_shares: Optional[int] = None
    short_interest_pct: Optional[float] = None
    avg_volume_20d: Optional[int] = None

    provider_name: str = "unknown"
    data_latency_ms: Optional[int] = None

    # Raw intraday bars used by MarketReactionEngine.
    # Expected columns: timestamp/index, Open, High, Low, Close, Volume.
    intraday_bars: Optional[Any] = None

    @property
    def dollar_volume(self) -> Optional[float]:
        vol = self.regular.volume or self.premarket.volume
        price = self.current_price or self.regular.close or self.premarket.close
        if vol and price:
            return vol * price
        return None

    @property
    def relative_volume(self) -> Optional[float]:
        today_vol = self.regular.volume or self.premarket.volume
        if today_vol and self.avg_volume_20d and self.avg_volume_20d > 0:
            return round(today_vol / self.avg_volume_20d, 2)
        return None

    @property
    def is_data_sufficient(self) -> bool:
        return (
            self.current_price is not None
            and self.previous_close is not None
            and (self.premarket.is_complete or self.regular.is_complete)
        )
