"""Hunter Bot — Intraday Intelligence data models (Phase 2.8).

All values are derived from real market data only. Any field that cannot be
computed from available data stays None/UNAVAILABLE with an explicit reason.
Missing data never produces bullish points.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class IntradayLevels:
    """Real intraday reference levels. Every level records its source."""
    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None
    previous_day_high: Optional[float] = None
    previous_day_low: Optional[float] = None
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    vwap: Optional[float] = None
    vwap_source: Optional[str] = None
    intraday_support: Optional[float] = None
    intraday_resistance: Optional[float] = None
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None

    def sources(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if self.premarket_high is not None:
            out["premarket_high"] = "premarket_session_bars"
        if self.premarket_low is not None:
            out["premarket_low"] = "premarket_session_bars"
        if self.previous_day_high is not None:
            out["previous_day_high"] = "daily_history_prev_bar"
        if self.previous_day_low is not None:
            out["previous_day_low"] = "daily_history_prev_bar"
        if self.opening_range_high is not None:
            out["opening_range_high"] = f"first_{self._or_minutes()}m_regular_bars"
        if self.opening_range_low is not None:
            out["opening_range_low"] = f"first_{self._or_minutes()}m_regular_bars"
        if self.vwap is not None and self.vwap_source:
            out["vwap"] = self.vwap_source
        if self.intraday_support is not None:
            out["intraday_support"] = "intraday_swing_pivot_low"
        if self.intraday_resistance is not None:
            out["intraday_resistance"] = "intraday_swing_pivot_high"
        if self.recent_high is not None:
            out["recent_high"] = "rolling_intraday_window"
        if self.recent_low is not None:
            out["recent_low"] = "rolling_intraday_window"
        return out

    _or_window_minutes: int = 15

    def _or_minutes(self) -> int:
        return int(self._or_window_minutes)


@dataclass
class IntradayMomentumVolume:
    """Deterministic momentum + volume measurements on the analysis timeframe."""
    momentum_direction: str = "UNKNOWN"          # UP / DOWN / FLAT / UNKNOWN
    price_acceleration: Optional[float] = None   # pct change of last k vs prior k closes
    volume_acceleration: Optional[float] = None  # last-k avg vol / prior-k avg vol
    volume_spike: bool = False                   # last-bar vol >= spike multiple of rolling mean
    volume_spike_ratio: Optional[float] = None
    rvol: Optional[float] = None                 # session relative volume (from TickerData)
    dollar_volume_minute: Optional[float] = None # typical session $ per minute
    price_vs_vwap_pct: Optional[float] = None
    above_vwap: Optional[bool] = None


@dataclass
class IntradaySetup:
    name: str
    direction: str                                # BULLISH / BEARISH / NEUTRAL
    detected: bool
    evidence: List[str] = field(default_factory=list)
    quality: str = "UNCONFIRMED"                  # CONFIRMED / UNCONFIRMED
    anchor_price: Optional[float] = None          # real level this setup is built on
    anchor_basis: Optional[str] = None            # what that level represents


@dataclass
class ConfirmationCheck:
    name: str
    met: Optional[bool]                           # None = cannot evaluate (missing data)
    detail: str = ""


@dataclass
class EntryPlan:
    """Structured entry intelligence. status=UNKNOWN when a reliable plan
    cannot be calculated from real data. Targets are intentionally excluded
    (Phase 2.10 scope)."""
    status: str = "UNKNOWN"                       # READY / UNKNOWN
    reason: Optional[str] = None
    setup: Optional[str] = None
    side: Optional[str] = None                    # LONG / SHORT
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    invalidation_price: Optional[float] = None
    invalidation_basis: Optional[str] = None
    risk_distance_abs: Optional[float] = None
    risk_distance_pct: Optional[float] = None
    confidence: int = 0
    confirmations: List[ConfirmationCheck] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class IntradayScoreComponent:
    name: str
    weight: float
    value: Optional[int]                          # 0-100 or None when unavailable
    reason: Optional[str] = None                  # why unavailable
    contribution: Optional[float] = None


@dataclass
class IntradayScore:
    total: int = 0
    components: List[IntradayScoreComponent] = field(default_factory=list)

    @property
    def available_weight(self) -> float:
        return sum(c.weight for c in self.components if c.value is not None)

    @property
    def renormalized(self) -> bool:
        return self.available_weight > 0 and abs(self.available_weight - 100.0) > 1e-9


@dataclass
class TimeframeAnalysis:
    timeframe: str = "1m"                         # "1m" / "5m" / "15m"
    bars_available: int = 0
    analyzed: bool = False
    unavailable_reason: Optional[str] = None


@dataclass
class IntradayIntelligence:
    ticker: str
    as_of: Optional[object] = None                # data.timestamp (datetime)
    timeframe: str = "1m"
    timeframes: List[TimeframeAnalysis] = field(default_factory=list)
    data_status: str = "OK"                       # OK / INSUFFICIENT_INTRADAY / NO_INTRADAY
    data_reasons: List[str] = field(default_factory=list)
    levels: IntradayLevels = field(default_factory=IntradayLevels)
    momentum_volume: IntradayMomentumVolume = field(default_factory=IntradayMomentumVolume)
    setups: List[IntradaySetup] = field(default_factory=list)
    entry: EntryPlan = field(default_factory=EntryPlan)
    score: IntradayScore = field(default_factory=IntradayScore)
    trap_flags: List[str] = field(default_factory=list)   # intraday-specific risk flags
    warnings: List[str] = field(default_factory=list)

    def primary_setup(self) -> Optional[IntradaySetup]:
        detected = [s for s in self.setups if s.detected]
        if not detected:
            return None
        priority = getattr(IntradayIntelligence, "SETUP_PRIORITY", None) or []
        return sorted(
            detected,
            key=lambda s: priority.index(s.name) if s.name in priority else len(priority),
        )[0]

    SETUP_PRIORITY = [
        "FAILED_BREAKOUT", "BREAKDOWN", "OPENING_RANGE_BREAKOUT",
        "CONSOLIDATION_BREAK", "BREAKOUT_PULLBACK", "BREAKOUT",
        "VWAP_RECLAIM", "VWAP_REJECTION", "VWAP_BOUNCE",
        "MOMENTUM_CONTINUATION", "VOLUME_EXPANSION", "NO_SETUP",
    ]

    def summary(self) -> str:
        parts = [f"{self.timeframe}", f"score={self.score.total}"]
        setup = self.primary_setup()
        if setup:
            parts.append(f"setup={setup.name}:{setup.direction}:{setup.quality}")
        else:
            parts.append("setup=NONE")
        parts.append(f"entry={self.entry.status}")
        if self.score.renormalized:
            parts.append(f"weights={int(self.score.available_weight)}")
        return " | ".join(parts)
