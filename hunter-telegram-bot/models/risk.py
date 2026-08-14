"""Risk model. Values are calculated scenarios, not execution instructions."""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RiskPlan:
    reference_price: Optional[float] = None
    entry_trigger: Optional[float] = None
    stop_price: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    risk_per_share: Optional[float] = None
    reward_to_risk_1: Optional[float] = None
    reward_to_risk_2: Optional[float] = None
    reward_to_risk_3: Optional[float] = None
    suggested_risk_pct: float = 0.5
    max_loss_amount: Optional[float] = None
    position_size: Optional[int] = None
    valid: bool = False
    confidence: int = 0
    warnings: List[str] = field(default_factory=list)
    method: str = "ATR_STRUCTURE"
