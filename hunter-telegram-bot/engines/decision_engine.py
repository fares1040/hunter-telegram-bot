"""Hunter decision engine — catalyst + market + options + risk + trap gates."""
from models.ticker import TickerData
from models.news import CatalystEvent
from models.signal import HunterSignal, HunterDecision
from engines.market_reaction_engine import ReactionMetrics
from engines.liquidity_proxy import LiquidityProxyResult
from engines.technical_engine import TechnicalProfile
from engines.options_engine import OptionsFlowProfile
from models.risk import RiskPlan
from core.data_confidence import DataConfidenceReport
from core.session_clock import SessionClock
from config.settings import SETTINGS
from engines.scoring_engine import CompositeScoringEngine
from utils.logger import LOGGER


class DecisionEngine:
    def __init__(self):
        self.scorer = CompositeScoringEngine()

    def decide(self, ticker_data, event, reaction, liquidity, technical, confidence_report, options=None, risk_plan=None, trap_risk=0, trap_warnings=None, market_context=None, technical_intelligence=None, intraday_intelligence=None):
        options = options or OptionsFlowProfile()
        risk_plan = risk_plan or RiskPlan(valid=True, confidence=70)
        trap_warnings = trap_warnings or []
        signal = HunterSignal(ticker=ticker_data.ticker, decision=HunterDecision.IGNORE)
        signal.data_confidence = confidence_report.score
        signal.session = SessionClock.get_session(ticker_data.timestamp).value
        signal.current_price = ticker_data.current_price
        signal.change_percent = ticker_data.change_percent
        signal.rvol = ticker_data.relative_volume
        signal.catalyst_type = event.catalyst_type.value
        signal.sentiment = event.sentiment
        signal.news_quality = event.source_tier_score
        signal.news_impact = event.impact_score
        signal.market_reaction = reaction.reaction_score
        signal.liquidity_proxy = liquidity.score
        signal.reaction_status = reaction.reaction_label
        signal.liquidity_status = liquidity.status
        signal.technical_structure = technical.setup_score
        signal.options_flow = options.flow_score
        signal.options_bias = options.bias
        if market_context:
            signal.market_regime = market_context.regime
            signal.market_regime_score = market_context.regime_score
            signal.sector = market_context.sector
            signal.sector_strength = market_context.sector_strength
        signal.trap_risk = trap_risk
        signal.warnings = list(technical.warnings) + list(trap_warnings) + list(options.notes)
        if technical_intelligence is not None:
            signal.warnings.append(
                f"TECH[{technical_intelligence.timeframe}] {technical_intelligence.summary()}"
            )
        if intraday_intelligence is not None:
            note = f"INTRADAY[{intraday_intelligence.timeframe}] {intraday_intelligence.summary()}"
            if intraday_intelligence.trap_flags:
                note += " | flags:" + ",".join(intraday_intelligence.trap_flags)
            signal.warnings.append(note)
        if options.contract_candidate:
            c = options.contract_candidate
            signal.contract_symbol = c.contract_symbol
            signal.contract_strike = c.strike
            signal.contract_expiration = c.expiration.isoformat()
            signal.contract_mid = c.mid
            signal.contract_iv = c.implied_volatility
        signal.entry_trigger = risk_plan.entry_trigger
        signal.stop_price = risk_plan.stop_price
        signal.target_1 = risk_plan.target_1
        signal.target_2 = risk_plan.target_2
        signal.target_3 = risk_plan.target_3
        signal.reward_to_risk = risk_plan.reward_to_risk_1
        signal.position_size = risk_plan.position_size
        signal.risk_score = 100 if risk_plan.valid and not risk_plan.warnings else max(0, risk_plan.confidence)
        signal.hunter_score = self.scorer.score(news_quality=signal.news_quality, news_impact=signal.news_impact, reaction=signal.market_reaction, liquidity=signal.liquidity_proxy, technical=signal.technical_structure, options=signal.options_flow, risk=signal.risk_score, market_regime=getattr(signal,"market_regime_score",50), sector_strength=getattr(signal,"sector_strength",50), trap_risk=signal.trap_risk)
        if signal.data_confidence < SETTINGS.hunter_min_data_confidence:
            signal.reasoning = f"Data confidence too low ({signal.data_confidence}%)"
            signal.data_insufficient_note = f"Confidence {signal.data_confidence}% < {SETTINGS.hunter_min_data_confidence}%"
            return signal
        if event.sentiment in {"NEGATIVE", "VERY_NEGATIVE"}:
            signal.reasoning = "Negative catalyst sentiment"
            return signal
        if not event.is_fresh(max_age_minutes=120):
            signal.reasoning = "News is stale (>2 hours old)"
            return signal
        if event.priced_in_probability > 0.7:
            signal.reasoning = f"Likely priced in ({event.priced_in_probability:.0%})"
            return signal
        if reaction.reaction_label == "NEGATIVE_REACTION":
            signal.reasoning = "Market reacted negatively to catalyst"
            return signal
        if trap_risk >= 60:
            signal.reasoning = f"Trap risk too high ({trap_risk}/100)"
            return signal
        if liquidity.status == "WEAK":
            signal.reasoning = "Liquidity proxy too weak"
            return signal
        if getattr(signal, "market_regime", "UNKNOWN") == "RISK_OFF" and signal.hunter_score < 85:
            signal.reasoning = "Risk-off market regime requires stronger confirmation"
            return signal
        hunt = all([
            signal.hunter_score >= SETTINGS.hunter_min_score,
            event.impact_score >= 70,
            reaction.reaction_score >= 60,
            liquidity.score >= 60,
            technical.setup_score >= 60,
            risk_plan.valid,
        ])
        if hunt:
            signal.decision = HunterDecision.HUNT_NOW
            signal.reasoning = "Catalyst + reaction + liquidity + structure + risk gate aligned"
        elif event.impact_score >= 60 and event.sentiment in {"POSITIVE", "VERY_POSITIVE"}:
            signal.decision = HunterDecision.WATCH
            signal.reasoning = "Material positive catalyst; confirmation still required"
        else:
            signal.reasoning = "Did not meet HUNT_NOW or WATCH gates"
        LOGGER.info(f"[Decision] {ticker_data.ticker} {signal.decision.value} score={signal.hunter_score}")
        return signal
