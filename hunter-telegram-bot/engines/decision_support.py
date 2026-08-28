"""Decision 2.0 support — pure deterministic, no I/O, no authority."""
from datetime import datetime, timezone
from typing import List, Optional
from models.decision import DecisionEvidence, WhyNow, Conflict, Conviction, OpportunityQuality, DecisionRationale

class WhyNowBuilder:
    @staticmethod
    def build(catalyst_ts: Optional[datetime], reaction_ts: Optional[datetime], reaction_label: str, fresh_realtime: bool = False) -> WhyNow:
        if not catalyst_ts or not reaction_ts:
            return WhyNow(status="UNKNOWN", explanation="INSUFFICIENT_EVIDENCE")
        # Must be after catalyst and within window
        delta = (reaction_ts - catalyst_ts).total_seconds()
        if delta < 0 or delta > 7200:  # 2h window max
            return WhyNow(status="UNKNOWN", explanation="INSUFFICIENT_EVIDENCE", catalyst_timestamp=catalyst_ts, reaction_timestamp=reaction_ts, elapsed_seconds=int(delta) if delta>=0 else None)
        if reaction_label in ("DATA_INSUFFICIENT", "UNKNOWN"):
            return WhyNow(status="UNKNOWN", explanation="INSUFFICIENT_EVIDENCE")
        overlap = delta >= 0 and delta <= 3600
        supporting = [f"catalyst {catalyst_ts.isoformat()}", f"reaction {reaction_label} after {int(delta)}s"]
        if fresh_realtime:
            supporting.append("fresh realtime evidence inside window")
        status = "CONFIRMED" if overlap and reaction_label in ("STRONG_POSITIVE_REACTION","POSITIVE_REACTION") else "PARTIAL" if overlap else "UNKNOWN"
        explanation = "catalyst-reaction temporal overlap confirmed" if status=="CONFIRMED" else "partial temporal evidence" if status=="PARTIAL" else "INSUFFICIENT_EVIDENCE"
        return WhyNow(status=status, explanation=explanation, catalyst_timestamp=catalyst_ts, reaction_timestamp=reaction_ts, elapsed_seconds=int(delta), temporal_overlap=overlap, supporting=supporting, provenance="reaction_engine+catalyst")

class ConflictDetector:
    @staticmethod
    def detect(event_sentiment: str, reaction_label: str, liquidity_status: str, options_bias: str, market_regime: str, trap_risk: int, realtime_bullish: bool, options_bullish: bool, stale_critical: bool = False) -> List[Conflict]:
        out = []
        # bullish catalyst + weak reaction
        if event_sentiment in ("POSITIVE","VERY_POSITIVE") and reaction_label in ("WEAK_REACTION","NEUTRAL","NEGATIVE_REACTION"):
            out.append(Conflict(type="catalyst_vs_reaction", severity="MEDIUM", description="bullish catalyst but weak reaction", bullish_evidence=event_sentiment, bearish_evidence=reaction_label, provenance="catalyst+reaction"))
        # bullish price + bearish flow (infer from options bias)
        if realtime_bullish and options_bias in ("PUT_BIASED","STRONG_PUT","PUT_LEAN"):
            out.append(Conflict(type="price_vs_options_flow", severity="HIGH", description="bullish price but bearish options flow", provenance="realtime+options"))
        # strong flow vs weak liquidity
        if options_bias in ("CALL_BIASED","STRONG_CALL") and liquidity_status=="WEAK":
            out.append(Conflict(type="flow_vs_liquidity", severity="HIGH", description="strong flow but weak liquidity", provenance="options+liquidity"))
        # momentum trap
        if trap_risk >= 40:
            out.append(Conflict(type="momentum_vs_trap", severity="HIGH" if trap_risk>=60 else "MEDIUM", description=f"trap risk {trap_risk}", provenance="trap_engine"))
        # regime
        if market_regime=="RISK_OFF":
            out.append(Conflict(type="setup_vs_regime", severity="MEDIUM", description="risk-off regime", provenance="market_context"))
        # stale
        if stale_critical:
            out.append(Conflict(type="signal_vs_staleness", severity="HIGH", description="critical evidence stale", provenance="freshness"))
        # realtime vs options
        if realtime_bullish and not options_bullish and options_bias not in ("UNAVAILABLE","UNKNOWN"):
            # already covered price_vs_flow
            pass
        return out

class ConvictionEngine:
    @staticmethod
    def build(alignment: int, quality: int, freshness: int, completeness: int, conflicts: List[Conflict]) -> Conviction:
        penalty = min(40, len(conflicts)*10 + sum(15 if c.severity=="HIGH" else 8 for c in conflicts))
        # Never increase from UNKNOWN/MISSING — those already lower quality/completeness/freshness inputs
        raw = int(alignment*0.35 + quality*0.25 + freshness*0.2 + completeness*0.2) - penalty
        score = max(0, min(100, raw))
        if score >= 75: level="HIGH"
        elif score >= 55: level="MEDIUM"
        elif score >= 30: level="LOW"
        else: level="INSUFFICIENT"
        return Conviction(score=score, level=level, alignment_score=alignment, quality_score=quality, freshness_score=freshness, conflict_penalty=penalty, completeness_score=completeness, rationale=f"alignment {alignment} quality {quality} freshness {freshness} completeness {completeness} penalty {penalty}")

class OpportunityQualityEngine:
    @staticmethod
    def build(conviction: Conviction, reaction_score: int, liquidity_score: int, risk_valid: bool, trap_risk: int, market_regime: str, freshness_ok: bool) -> OpportunityQuality:
        if not risk_valid:
            return OpportunityQuality(score=0, tier="UNAVAILABLE", risk_valid=False, trap_risk=trap_risk, rationale="risk invalid")
        if trap_risk >= 60:
            return OpportunityQuality(score=10, tier="LOW_QUALITY", risk_valid=True, trap_risk=trap_risk, rationale="high trap risk")
        base = int(conviction.score*0.5 + reaction_score*0.25 + liquidity_score*0.25)
        if not freshness_ok:
            base = max(0, base - 20)
        if market_regime=="RISK_OFF":
            base = max(0, base - 15)
        risk_adj = max(0, min(100, base))
        if risk_adj >= 75 and conviction.level=="HIGH": tier="HIGH_QUALITY"
        elif risk_adj >= 55: tier="ACTIONABLE"
        elif risk_adj >= 35: tier="INTERESTING"
        else: tier="LOW_QUALITY"
        return OpportunityQuality(score=risk_adj, tier=tier, risk_adjusted_score=risk_adj, risk_valid=True, trap_risk=trap_risk, rationale=f"base {base}")

def build_rationale(supporting: List[DecisionEvidence], conflicts: List[Conflict], risks: List[str], why_now: WhyNow, conviction: Conviction, quality: OpportunityQuality) -> DecisionRationale:
    summary = f"{why_now.status} why_now; conviction {conviction.level} ({conviction.score}); quality {quality.tier}; {len(conflicts)} conflicts"
    return DecisionRationale(supporting=supporting, conflicting=conflicts, risks=risks, why_now=why_now, conviction=conviction, opportunity_quality=quality, summary=summary)
