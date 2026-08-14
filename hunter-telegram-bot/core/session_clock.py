"""
Hunter Bot — Market Session Clock
Handles NY timezone, DST, Premarket / Regular / After-Hours boundaries.
NEVER uses naive datetime.now().
"""
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional
import pytz


class MarketSession(Enum):
    PREMARKET = "PREMARKET"
    REGULAR = "REGULAR"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED = "CLOSED"


class SessionClock:
    """Single source of truth for market time. All internal timestamps are UTC-aware."""
    _tz = pytz.timezone("America/New_York")

    @classmethod
    def now(cls) -> datetime:
        """Current time in market timezone (ET)."""
        return datetime.now(cls._tz)

    @classmethod
    def now_utc(cls) -> datetime:
        """Current time in UTC."""
        return datetime.now(pytz.UTC)

    @classmethod
    def localize(cls, dt: datetime) -> datetime:
        """Naive datetime → market timezone aware."""
        if dt.tzinfo is None:
            return cls._tz.localize(dt)
        return dt.astimezone(cls._tz)

    @classmethod
    def to_utc(cls, dt: datetime) -> datetime:
        """Any aware datetime → UTC."""
        return dt.astimezone(pytz.UTC)

    @classmethod
    def get_session(cls, dt: Optional[datetime] = None) -> MarketSession:
        """Determine market session for a given timestamp. Defaults to now()."""
        if dt is None:
            dt = cls.now()
        else:
            dt = cls.localize(dt)

        t = dt.time()
        weekday = dt.weekday()

        if weekday >= 5:
            return MarketSession.CLOSED

        pre_start = time(4, 0)
        mkt_open = time(9, 30)
        mkt_close = time(16, 0)
        ah_end = time(20, 0)

        if pre_start <= t < mkt_open:
            return MarketSession.PREMARKET
        elif mkt_open <= t < mkt_close:
            return MarketSession.REGULAR
        elif mkt_close <= t < ah_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    @classmethod
    def is_premarket(cls, dt: Optional[datetime] = None) -> bool:
        return cls.get_session(dt) == MarketSession.PREMARKET

    @classmethod
    def is_regular(cls, dt: Optional[datetime] = None) -> bool:
        return cls.get_session(dt) == MarketSession.REGULAR

    @classmethod
    def is_after_hours(cls, dt: Optional[datetime] = None) -> bool:
        return cls.get_session(dt) == MarketSession.AFTER_HOURS
