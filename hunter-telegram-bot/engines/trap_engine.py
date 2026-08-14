"""Trap and quality-control gate."""
from typing import List
from models.ticker import TickerData
from models.news import CatalystEvent
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from engines.technical_engine import TechnicalProfile


class TrapEngine:
    def analyze(self, data: TickerData, event: CatalystEvent, reaction: ReactionMetrics, liquidity: LiquidityProxyResult, technical: TechnicalProfile) -> tuple[int, List[str]]:
        score = 0
        warnings: List[str] = []
        if reaction.reaction_label == "NEGATIVE_REACTION":
            score += 45; warnings.append("NEGATIVE_NEWS_REACTION")
        if liquidity.status == "WEAK":
            score += 20; warnings.append("WEAK_LIQUIDITY")
        if data.gap_percent is not None and abs(data.gap_percent) >= 30:
            score += 15; warnings.append("EXTREME_GAP")
        if event.priced_in_probability >= 0.7:
            score += 20; warnings.append("PRICED_IN_RISK")
        if technical.rsi is not None and technical.rsi >= 85:
            score += 15; warnings.append("EXTREME_RSI")
        return min(100, score), warnings
