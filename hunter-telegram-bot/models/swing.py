"""Hunter Bot — Swing Intelligence data models (Phase 2.9).

Separation of concerns (mirrors intraday / technical models):
- REAL DATA: prices/volumes/MA values copied from provider history or computed
  deterministically from daily bars
- INFERENCE: trend/structure/momentum/setup classifications from rules
- SCORE: explainable 0-100 components with visible weights
- AI EXPLANATION: never stored here

All values derive from real daily history. Insufficient history yields
UNKNOWN / None. Missing data never becomes bullish points. Targets are
explicitly excluded (Phase 2.10 scope).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class SwingLevel:
    """A real support or resistance level with explainable strength."""
    price: float
    level_type: str                                 # SUPPORT / RESISTANCE
    strength: int                                   # 0-100
    distance_pct: float                             # from current price; negative below
    evidence: str
    role: str = ""                                  # e.g. ROLE_REVERSAL / MAJOR / PIVOT


@dataclass
class SwingTrend:
    """Multi-day trend context. Prioritizes MA50 / MA200."""
    direction: str = "UNKNOWN"                      # BULLISH / BEARISH / NEUTRAL / TRANSITION
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    price_vs_ma50_pct: Optional[float] = None
    price_vs_ma200_pct: Optional[float] = None
    ma_alignment: Optional[str] = None             # PRICE>MA50>MA200 etc.
    ma50_slope_pct: Optional[float] = None
    ma200_slope_pct: Optional[float] = None
    structure: Optional[str] = None                 # HH_HL / LH_LL / MIXED
    structure_evidence: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class SwingSetup:
    """Deterministic swing setup detection."""
    name: str
    direction: str                                 # BULLISH / BEARISH / NEUTRAL
    detected: bool = False
    evidence: List[str] = field(default_factory=list)
    quality: str = "UNCONFIRMED"                   # CONFIRMED / UNCONFIRMED / WATCH
    anchor_price: Optional[float] = None           # real level this setup is built on
    anchor_basis: Optional[str] = None             # what that level represents


@dataclass
class SwingMomentum:
    """Multi-day momentum context (reused TechnicalIntelligence where present)."""
    rsi: Optional[float] = None
    roc_10: Optional[float] = None
    roc_20: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    direction: str = "NEUTRAL"                     # STRONG/POSITIVE/NEUTRAL/NEGATIVE/WEAK
    acceleration: Optional[str] = None             # BUILDING / FADING
    divergence: Optional[str] = None               # only when clearly detectable
    missing: List[str] = field(default_factory=list)


@dataclass
class SwingVolume:
    """Real daily volume context."""
    rvol: Optional[float] = None                   # session vs 20d avg (relative volume)
    last_bar_rvol: Optional[float] = None          # last daily bar vs trailing mean
    volume_expansion: bool = False                 # breakout bar expanding vs baseline
    pullback_contraction: bool = False             # pullback bar contracting vs baseline
    dollar_volume: Optional[float] = None
    regime: str = "NORMAL"                         # LOW / NORMAL / ELEVATED / HIGH / EXTREME
    missing: List[str] = field(default_factory=list)


@dataclass
class SwingConfirmationCheck:
    name: str
    met: Optional[bool]                            # None = cannot evaluate (missing data)
    detail: str = ""


@dataclass
class SwingEntry:
    """Structured swing entry intelligence. status=UNKNOWN when a reliable plan
    cannot be calculated from real data. Targets are intentionally excluded
    (Phase 2.10 scope)."""
    status: str = "UNKNOWN"                        # READY / UNKNOWN
    reason: Optional[str] = None
    setup: Optional[str] = None
    side: Optional[str] = None                     # LONG / SHORT
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    confirmation: Optional[str] = None
    invalidation_price: Optional[float] = None
    invalidation_basis: Optional[str] = None
    risk_distance_abs: Optional[float] = None
    risk_distance_pct: Optional[float] = None
    confidence: int = 0
    confirmations: List[SwingConfirmationCheck] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class SwingCatalystContext:
    """Lightweight catalyst integration — reuses Phase 2.6, never duplicates it."""
    present: bool = False
    sentiment: Optional[str] = None                # POSITIVE / NEGATIVE / MIXED / NEUTRAL
    category: Optional[str] = None
    freshness: Optional[str] = None
    materiality: Optional[int] = None
    is_trap_risk: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class SwingScoreComponent:
    name: str
    weight: float
    value: Optional[int]                           # 0-100 or None when unavailable
    reason: Optional[str] = None                   # why unavailable
    contribution: Optional[float] = None


@dataclass
class SwingScore:
    total: int = 0                                 # 0-100
    components: List[SwingScoreComponent] = field(default_factory=list)

    @property
    def available_weight(self) -> float:
        return sum(c.weight for c in self.components if c.value is not None)

    @property
    def renormalized(self) -> bool:
        return self.available_weight > 0 and abs(self.available_weight - 100.0) > 1e-9


@dataclass
class SwingIntelligence:
    """Top-level swing intelligence output."""
    ticker: str
    as_of: Optional[Any] = None                    # data.timestamp
    timeframe: str = "1d"
    data_status: str = "OK"                        # OK / INSUFFICIENT_HISTORY / NO_DATA
    data_reasons: List[str] = field(default_factory=list)
    levels: List[SwingLevel] = field(default_factory=list)
    trend: SwingTrend = field(default_factory=SwingTrend)
    setups: List[SwingSetup] = field(default_factory=list)
    momentum: SwingMomentum = field(default_factory=SwingMomentum)
    volume: SwingVolume = field(default_factory=SwingVolume)
    entry: SwingEntry = field(default_factory=SwingEntry)
    score: SwingScore = field(default_factory=SwingScore)
    catalyst: SwingCatalystContext = field(default_factory=SwingCatalystContext)
    intraday_confirmation: Optional[str] = None    # optional confirmation summary
    trap_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)

    def primary_setup(self) -> Optional[SwingSetup]:
        detected = [s for s in self.setups if s.detected]
        if not detected:
            return None
        priority = getattr(SwingIntelligence, "SETUP_PRIORITY", None) or []
        return sorted(
            detected,
            key=lambda s: priority.index(s.name) if s.name in priority else len(priority),
        )[0]

    SETUP_PRIORITY = [
        "FAILED_BREAKOUT", "BREAKDOWN", "BREAKOUT_RETEST", "BREAKOUT",
        "BASE_BREAKOUT", "RANGE_BREAKOUT", "HIGHER_LOW_CONTINUATION",
        "TREND_CONTINUATION", "PULLBACK_UPTREND", "PULLBACK_DOWNTREND", "NO_SETUP",
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
