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
    def analyze(self, data: TickerData) -> LiquidityProxyResult:
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
