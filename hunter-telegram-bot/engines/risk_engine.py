"""Risk gate and scenario levels."""
from typing import Optional
from models.risk import RiskPlan
from engines.technical_engine import TechnicalProfile


class RiskEngine:
    def build_plan(self, price: Optional[float], technical: TechnicalProfile, account_size: Optional[float] = None, risk_pct: float = 0.5) -> RiskPlan:
        plan = RiskPlan(reference_price=price, suggested_risk_pct=risk_pct)
        if not price or price <= 0:
            plan.warnings.append("No valid reference price")
            return plan
        atr = technical.atr or price * 0.02
        structure_stop = technical.recent_swing_low or technical.premarket_low
        if structure_stop is None or structure_stop >= price:
            structure_stop = price - max(atr * 1.2, price * 0.02)
        risk = price - structure_stop
        if risk <= 0:
            plan.warnings.append("Invalid stop distance")
            return plan
        entry = max(price, technical.premarket_high or price)
        # Avoid chasing a very distant breakout trigger.
        if entry > price * 1.08:
            plan.warnings.append("Breakout trigger is extended >8% above reference")
        plan.entry_trigger = round(entry, 2)
        plan.stop_price = round(structure_stop, 2)
        plan.risk_per_share = round(risk, 2)
        plan.target_1 = round(entry + risk * 1.5, 2)
        plan.target_2 = round(entry + risk * 2.5, 2)
        plan.target_3 = round(entry + risk * 4.0, 2)
        plan.reward_to_risk_1 = 1.5
        plan.reward_to_risk_2 = 2.5
        plan.reward_to_risk_3 = 4.0
        if account_size and account_size > 0:
            max_loss = account_size * (risk_pct / 100.0)
            plan.max_loss_amount = round(max_loss, 2)
            plan.position_size = max(0, int(max_loss / risk))
        plan.confidence = 75 if technical.recent_swing_low or technical.premarket_low else 55
        plan.valid = plan.confidence >= 55
        # Compute actual R:R from plan values and validate against minimum threshold.
        MIN_RR = 1.5
        if plan.entry_trigger is not None and plan.stop_price is not None and plan.target_1 is not None:
            actual_rr = (plan.target_1 - plan.entry_trigger) / max(1e-9, plan.entry_trigger - plan.stop_price)
            if actual_rr < MIN_RR:
                plan.warnings.append("Low reward-to-risk")
        return plan
