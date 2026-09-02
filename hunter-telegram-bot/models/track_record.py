"""Track Record domain models."""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone
from enum import Enum

class SignalStatus(str, Enum):
    OPEN = "OPEN"
    PENDING = "PENDING"
    WON = "WON"
    LOST = "LOST"
    EXPIRED = "EXPIRED"
    UNRESOLVED = "UNRESOLVED"

@dataclass
class SignalRecord:
    signal_id: str
    ticker: str
    timestamp: str
    decision: str
    hunter_score: int = 0
    data_confidence: int = 0
    entry_trigger: Optional[float] = None
    stop_price: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    catalyst_type: str = ""
    sentiment: str = ""
    strategy_state: str = ""
    setup_name: str = ""
    status: str = SignalStatus.OPEN.value
    outcome_price: Optional[float] = None
    outcome_at: Optional[str] = None
    forward_return: Optional[float] = None
    realized_return: Optional[float] = None
    provenance: str = "observed"
