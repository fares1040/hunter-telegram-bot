"""Hunter Signal — full decision payload."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HunterDecision(Enum):
    HUNT_NOW = "HUNT_NOW"
    WATCH = "WATCH"
    IGNORE = "IGNORE"


@dataclass
class HunterSignal:
    ticker: str
    decision: HunterDecision
    hunter_score: int = 0
    data_confidence: int = 0
    timestamp: datetime = field(default_factory=_utc_now)
    news_quality: int = 0
    news_impact: int = 0
    market_reaction: int = 0
    liquidity_proxy: int = 0
    technical_structure: int = 0
    momentum: int = 0
    options_flow: int = 50
    risk_score: int = 0
    catalyst_type: str = ""
    sentiment: str = "NEUTRAL"
    session: str = ""
    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    rvol: Optional[float] = None
    technical_setup: str = "NO_SETUP"
    liquidity_status: str = "UNKNOWN"
    reaction_status: str = "UNKNOWN"
    options_bias: str = "UNAVAILABLE"
    market_regime: str = "UNKNOWN"
    market_regime_score: int = 50
    sector: str = "UNKNOWN"
    sector_strength: int = 50
    contract_score: Optional[int] = None
    invalidation: Optional[str] = None
    contract_symbol: Optional[str] = None
    contract_strike: Optional[float] = None
    contract_expiration: Optional[str] = None
    contract_mid: Optional[float] = None
    contract_iv: Optional[float] = None
    entry_trigger: Optional[float] = None
    stop_price: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    reward_to_risk: Optional[float] = None
    position_size: Optional[int] = None
    trap_risk: int = 0
    warnings: List[str] = field(default_factory=list)
    reasoning: str = ""
    data_insufficient_note: Optional[str] = None
    educational_only: bool = True

    @property
    def is_actionable(self) -> bool:
        return self.decision == HunterDecision.HUNT_NOW and self.data_confidence >= 60 and self.risk_score >= 60

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["decision"] = self.decision.value
        d["timestamp"] = self.timestamp.isoformat()
        return d
