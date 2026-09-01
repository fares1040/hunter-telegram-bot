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
from models.supply_demand import SupplyDemandResult
from models.options_flow import OptionsFlowIntelligence
from models.strategy import StrategyResult


class DecisionEngine:
    def __init__(self):
        self.scorer = CompositeScoringEngine()

    def decide(self, ticker_data, event, reaction, liquidity, technical, confidence_report, options=None, risk_plan=None, trap_risk=0, trap_warnings=None, market_context=None, technical_intelligence=None, intraday_intelligence=None, swing_intelligence=None, target_result=None, supply_demand_result=None, options_flow_intelligence=None, strategy_result=None, fresh_realtime: bool = False, has_true_flow: bool = False):
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
        # Use RR21 OptionsFlowIntelligence flow_score when available and reliable
        if options_flow_intelligence is not None and options_flow_intelligence.data_quality in ("REAL", "PROXY", "FRESH"):
            signal.options_flow = options_flow_intelligence.flow_score
            signal.options_bias = options_flow_intelligence.bias
        else:
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
        if swing_intelligence is not None:
            note = f"SWING[{swing_intelligence.timeframe}] {swing_intelligence.summary()}"
            if swing_intelligence.trap_flags:
                note += " | flags:" + ",".join(swing_intelligence.trap_flags)
            signal.warnings.append(note)
        if supply_demand_result is not None:
            sd = supply_demand_result
            if sd.demand_zones or sd.supply_zones:
                parts = [f"SUPDEM: {len(sd.demand_zones)}D/{len(sd.supply_zones)}S zones"]
                if sd.nearest_demand:
                    parts.append(f"nearest_demand={sd.nearest_demand.zone_low:.2f}-{sd.nearest_demand.zone_high:.2f}")
                if sd.nearest_supply:
                    parts.append(f"nearest_supply={sd.nearest_supply.zone_low:.2f}-{sd.nearest_supply.zone_high:.2f}")
                if sd.dominant_zone_type:
                    parts.append(f"dominant={sd.dominant_zone_type}")
                signal.warnings.append(" | ".join(parts))
            if sd.warnings:
                signal.warnings.extend(sd.warnings)
            if sd.missing_data:
                signal.warnings.append(f"SUPDEM missing: {', '.join(sd.missing_data)}")
        if options_flow_intelligence is not None:
            of = options_flow_intelligence
            parts = [f"OPTFLOW: quality={of.data_quality} freshness={of.freshness} bias={of.bias} flow_score={of.flow_score}"]
            if of.chain_age_minutes is not None:
                parts.append(f"age={of.chain_age_minutes}min")
            if of.contract_candidate:
                c = of.contract_candidate
                parts.append(f"candidate={c.contract_symbol} strike={c.strike} mid={c.mid:.2f} iv={c.implied_volatility:.2f}")
            signal.warnings.append(" | ".join(parts))
            if of.warnings:
                signal.warnings.extend(of.warnings)
            if of.notes:
                signal.warnings.extend(of.notes)
        if strategy_result is not None:
            st = strategy_result
            parts = [f"STRATEGY: state={st.state} direction={st.direction or 'NONE'} confirm={st.confirmation} confidence={st.confidence}"]
            if st.entry and st.entry.status != "UNAVAILABLE":
                parts.append(f"entry={st.entry.entry_zone_low:.2f}-{st.entry.entry_zone_high:.2f} inv={st.entry.invalidation_price:.2f}")
            if st.target and st.target.status != "UNAVAILABLE":
                if st.target.tp1_low:
                    parts.append(f"tp1={st.target.tp1_low:.2f}-{st.target.tp1_high:.2f}")
                if st.target.tp2_low:
                    parts.append(f"tp2={st.target.tp2_low:.2f}-{st.target.tp2_high:.2f}")
            if st.risk and st.risk.invalidation_clear is not None:
                parts.append(f"risk_clear={st.risk.invalidation_clear}")
            signal.warnings.append(" | ".join(parts))
            if st.warnings:
                signal.warnings.extend(st.warnings)
            if st.evidence and hasattr(st.evidence, 'missing') and st.evidence.missing:
                signal.warnings.append(f"STRATEGY missing: {', '.join(st.evidence.missing)}")
        signal.target_result = target_result
        if target_result is not None:
            t = target_result
            parts = [f"TARGETS[{t.direction}] status={t.status}"]
            if t.tp1:
                parts.append(f"TP1={t.tp1.zone.zone_low}-{t.tp1.zone.zone_high}({t.tp1.zone.source_type})")
            if t.tp2:
                parts.append(f"TP2={t.tp2.zone.zone_low}-{t.tp2.zone.zone_high}({t.tp2.zone.source_type})")
            if t.tp3:
                parts.append(f"TP3={t.tp3.zone.zone_low}-{t.tp3.zone.zone_high}({t.tp3.zone.source_type})")
            if t.risk_reward is not None:
                parts.append(f"RR={t.risk_reward:.2f}")
            if t.score is not None:
                parts.append(f"tgt_score={t.score.total}")
            if t.confidence is not None:
                parts.append(f"tgt_conf={t.confidence.value}")
            signal.warnings.append(" | ".join(parts))
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
        # Decision 2.0 — additive support (no gate change)
        try:
            from engines.decision_support import WhyNowBuilder, ConflictDetector, ConvictionEngine, OpportunityQualityEngine, build_rationale
            from models.decision import DecisionEvidence
            # derive realtime freshness
            catalyst_ts = getattr(event.primary_source, "published_at", None) if hasattr(event, "primary_source") else None
            reaction_ts = getattr(reaction, "reaction_timestamp", None)
            why = WhyNowBuilder.build(catalyst_ts, reaction_ts, reaction.reaction_label, fresh_realtime)
            realtime_bullish = reaction.reaction_label in ("STRONG_POSITIVE_REACTION","POSITIVE_REACTION")
            options_bullish = "CALL" in signal.options_bias
            # Freshness from actual evidence: catalyst age, reaction data_sufficient, options freshness
            catalyst_fresh = "FRESH" if catalyst_ts and event.is_fresh(max_age_minutes=120) else ("UNKNOWN" if not catalyst_ts else "STALE")
            reaction_fresh = "FRESH" if reaction.data_sufficient else "UNKNOWN" if reaction.reaction_label=="UNKNOWN" else "STALE" if reaction.reaction_label=="DATA_INSUFFICIENT" else "FRESH"
            options_fresh = getattr(options_flow_intelligence, "freshness", "UNKNOWN") if options_flow_intelligence else "UNKNOWN"
            stale_critical = catalyst_fresh=="STALE" or reaction_fresh=="STALE" or options_fresh=="STALE" or reaction.reaction_label=="DATA_INSUFFICIENT"
            unknown_critical = catalyst_fresh=="UNKNOWN" or reaction_fresh=="UNKNOWN"
            conflicts = ConflictDetector.detect(event.sentiment, reaction.reaction_label, liquidity.status, signal.options_bias, signal.market_regime, trap_risk, realtime_bullish, options_bullish, stale_critical)
            # Conviction: alignment from hunter_score, quality from data_confidence, freshness/completeness from actual freshness
            alignment = 80 if signal.hunter_score >= 75 else 60 if signal.hunter_score >= 60 else 30
            quality = signal.data_confidence  # quality remains confidence as proxy for completeness of snapshot data (already REAL/UNKNOWN tagged in supporting)
            if stale_critical:
                freshness = 20
            elif unknown_critical:
                freshness = 40
            else:
                freshness = 80 if catalyst_fresh=="FRESH" and reaction_fresh=="FRESH" else 50
            completeness = 70 if signal.data_confidence >= 70 and not unknown_critical else 40
            conviction = ConvictionEngine.build(alignment, quality, freshness, completeness, conflicts)
            quality_obj = OpportunityQualityEngine.build(conviction, reaction.reaction_score, liquidity.score, risk_plan.valid, trap_risk, signal.market_regime, not stale_critical)
            supporting = [DecisionEvidence(name="catalyst", direction="BULLISH" if event.sentiment in ("POSITIVE","VERY_POSITIVE") else "UNKNOWN", quality="REAL" if catalyst_ts else "UNKNOWN", source="catalyst", description=str(event.catalyst_type.value if hasattr(event.catalyst_type,'value') else event.catalyst_type))]
            if fresh_realtime:
                supporting.append(DecisionEvidence(name="realtime", direction="BULLISH" if realtime_bullish else "UNKNOWN", quality="REAL", source="polygon_realtime_quote", freshness="FRESH", description="fresh realtime quote/trade"))
            if has_true_flow:
                supporting.append(DecisionEvidence(name="true_flow", direction="BULLISH" if options_bullish else "UNKNOWN", quality="REAL", source="polygon_options_realtime_trade", freshness="FRESH", description="fresh true option flow"))
            risks = list(trap_warnings)[:5]
            signal.decision2 = build_rationale(supporting, conflicts, risks, why, conviction, quality_obj)
        except Exception:
            pass
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
        elif event.impact_score >= 55 and event.sentiment in {"POSITIVE", "VERY_POSITIVE"} and reaction.reaction_score >= 55 and liquidity.score >= 55 and technical.setup_score >= 55 and trap_risk < 60 and not stale_critical and not unknown_critical and getattr(signal, "market_regime", "UNKNOWN") != "RISK_OFF":
            signal.decision = HunterDecision.WATCH
            signal.reasoning = "Borderline confluence: supportive reaction/liquidity/structure with material catalyst"
        else:
            signal.reasoning = "Did not meet HUNT_NOW or WATCH gates"
        LOGGER.info(f"[Decision] {ticker_data.ticker} {signal.decision.value} score={signal.hunter_score}")
        return signal
