"""Cheap pre-filter that protects downstream API/AI capacity."""
from dataclasses import dataclass
from models.ticker import TickerData
from config.settings import SETTINGS

@dataclass
class CandidateGateResult:
    passed: bool
    score: int
    reasons: list[str]

class CandidateGate:
    def evaluate(self, data: TickerData) -> CandidateGateResult:
        reasons=[]; score=100
        if data.current_price is None or data.current_price <= 0:
            return CandidateGateResult(False, 0, ["NO_PRICE"])
        if data.current_price > SETTINGS.max_watchlist_price:
            score -= 35; reasons.append("PRICE_ABOVE_LIMIT")
        if data.dollar_volume is not None and data.dollar_volume < 1_000_000:
            score -= 30; reasons.append("LOW_DOLLAR_VOLUME")
        if data.gap_percent is not None and abs(data.gap_percent) > 50:
            score -= 20; reasons.append("EXTREME_GAP")
        return CandidateGateResult(score >= 50, max(0, score), reasons)
