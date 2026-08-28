"""Hunter Bot — Liquidity Proxy Engine"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

from models.ticker import TickerData
from core.data_confidence import DataQuality
from utils.logger import LOGGER


@dataclass
class LiquidityProxyResult:
    status: str = "UNKNOWN"
    score: int = 0
    rvol: Optional[float] = None
    dollar_volume: Optional[float] = None
    volume_spike: bool = False
    notes: str = ""
    data_quality_notes: List[Tuple[str, DataQuality, float]] = None

    def __post_init__(self):
        if self.data_quality_notes is None:
            self.data_quality_notes = []


class LiquidityProxyEngine:
    def analyze(self, data: TickerData, quotes: Optional[list] = None, trades: Optional[list] = None, realtime_max_age_seconds: int = 30) -> LiquidityProxyResult:
        result = LiquidityProxyResult()

        rvol = data.relative_volume
        result.rvol = rvol
        if rvol is not None:
            result.data_quality_notes.append(("rvol", DataQuality.PROXY, 1.0))
        else:
            result.data_quality_notes.append(("rvol", DataQuality.MISSING, 1.0))

        dv = data.dollar_volume
        result.dollar_volume = dv
        if dv is not None:
            result.data_quality_notes.append(("dollar_volume", DataQuality.PROXY, 1.0))
        else:
            result.data_quality_notes.append(("dollar_volume", DataQuality.MISSING, 1.0))

        if rvol and rvol >= 3.0:
            result.volume_spike = True

        score = 50

        if rvol is not None:
            if rvol >= 5.0:
                score += 25
                result.notes += f"RVOL {rvol}x (very strong). "
            elif rvol >= 3.0:
                score += 15
                result.notes += f"RVOL {rvol}x (strong). "
            elif rvol >= 2.0:
                score += 5
                result.notes += f"RVOL {rvol}x (elevated). "
            else:
                score -= 20
                result.notes += f"RVOL {rvol}x (weak). "
        else:
            score -= 15
            result.notes += "RVOL unavailable. "

        if dv is not None:
            if dv >= 50_000_000:
                score += 15
                result.notes += "High dollar volume. "
            elif dv >= 10_000_000:
                score += 8
                result.notes += "Moderate dollar volume. "
            elif dv < 1_000_000:
                score -= 15
                result.notes += "Low dollar volume. "
        else:
            score -= 10
            result.notes += "Dollar volume unavailable. "

        # Additive realtime evidence (stale/missing never increases conviction)
        fresh_trades = [t for t in (trades or []) if t.freshness(realtime_max_age_seconds) == "FRESH"] if trades else []
        fresh_quotes = [q for q in (quotes or []) if q.freshness(realtime_max_age_seconds) == "FRESH"] if quotes else []
        if fresh_trades:
            # Recent trade activity adds a small boost (capped)
            score = min(100, score + min(5, len(fresh_trades)))
            result.notes += f"Realtime trades: {len(fresh_trades)} fresh. "
        if fresh_quotes:
            valid = [q for q in fresh_quotes if q.is_valid]
            if valid:
                spreads = [q.spread_pct for q in valid if q.spread_pct is not None]
                if spreads:
                    avg_spread = sum(spreads) / len(spreads)
                    if avg_spread is not None and avg_spread <= 0.5:
                        result.notes += f"Tight spread {avg_spread:.2f}% (realtime). "

        result.score = max(0, min(100, score))

        if result.score >= 80:
            result.status = "STRONG_LIQUIDITY_PROXY"
        elif result.score >= 60:
            result.status = "LIQUIDITY_PROXY"
        elif result.score >= 40:
            result.status = "NEUTRAL"
        else:
            result.status = "WEAK"

        LOGGER.info(f"[Liquidity] {data.ticker} | {result.status} | Score: {result.score}")
        return result
