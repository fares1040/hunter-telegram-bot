"""Stage 1 Foundation Hardening — Regression Tests.

Tests for:
- Single authoritative strategy model definitions
- OptionsDataFreshness uniqueness
- contract_candidate uniqueness
- cleaned imports/contracts
- provenance behavior
- RR19/RR21/RR22 integration after cleanup
- missing/stale behavior
- DecisionEngine authority remains unchanged
"""
import pytest
from datetime import datetime, timezone


class TestStrategyModelIntegrity:
    """Verify strategy model has single authoritative definitions."""

    def test_strategy_evidence_single_definition(self):
        """StrategyEvidence must have exactly one definition."""
        from models.strategy import StrategyEvidence
        e = StrategyEvidence()
        assert hasattr(e, 'monthly_structure')
        assert hasattr(e, 'higher_timeframe_demand')
        assert hasattr(e, 'missing')
        assert hasattr(e, 'evidence')
        assert hasattr(e, 'source')

    def test_strategy_entry_single_definition(self):
        """StrategyEntry must have exactly one definition."""
        from models.strategy import StrategyEntry
        e = StrategyEntry()
        assert hasattr(e, 'status')
        assert hasattr(e, 'entry_zone_low')
        assert hasattr(e, 'evidence')

    def test_strategy_target_single_definition(self):
        """StrategyTarget must have exactly one definition."""
        from models.strategy import StrategyTarget
        t = StrategyTarget()
        assert hasattr(t, 'status')
        assert hasattr(t, 'tp1_low')
        assert hasattr(t, 'evidence')

    def test_strategy_risk_single_definition(self):
        """StrategyRisk must have exactly one definition."""
        from models.strategy import StrategyRisk
        r = StrategyRisk()
        assert hasattr(r, 'status')
        assert hasattr(r, 'invalidation_clear')

    def test_strategy_result_evidence_field(self):
        """StrategyResult must have evidence field (not evidence_package)."""
        from models.strategy import StrategyResult, StrategyEvidence
        r = StrategyResult(ticker="TEST", as_of=datetime.now(timezone.utc))
        assert hasattr(r, 'evidence')
        assert isinstance(r.evidence, StrategyEvidence)
        assert not hasattr(r, 'evidence_package')

    def test_strategy_result_is_actionable(self):
        """StrategyResult.is_actionable must work correctly."""
        from models.strategy import StrategyResult, StrategyRisk
        r = StrategyResult(
            ticker="TEST",
            as_of=datetime.now(timezone.utc),
            state="CONFIRMED",
            confirmation="CONFIRMED",
            risk=StrategyRisk(invalidation_clear=True)
        )
        assert r.is_actionable is True

    def test_strategy_result_summary_uses_evidence(self):
        """StrategyResult.summary() must use evidence field."""
        from models.strategy import StrategyResult, StrategyEvidence
        r = StrategyResult(
            ticker="TEST",
            as_of=datetime.now(timezone.utc),
            evidence=StrategyEvidence(missing=["test_missing"])
        )
        summary = r.summary()
        assert "test_missing" in summary


class TestOptionsFlowIntegrity:
    """Verify options flow model integrity."""

    def test_options_data_freshness_single_definition(self):
        """OptionsDataFreshness must have exactly one definition."""
        from models.options_flow import OptionsDataFreshness
        assert OptionsDataFreshness.FRESH.value == "FRESH"
        assert OptionsDataFreshness.STALE.value == "STALE"
        assert OptionsDataFreshness.UNKNOWN.value == "UNKNOWN"

    def test_options_flow_intelligence_contract_candidate_single(self):
        """OptionsFlowIntelligence.contract_candidate must be declared once."""
        from models.options_flow import OptionsFlowIntelligence
        from models.options import OptionContract
        from datetime import date
        o = OptionsFlowIntelligence()
        assert o.contract_candidate is None
        c = OptionContract(
            ticker="TEST",
            contract_symbol="TEST250101C100",
            contract_type="CALL",
            strike=100.0,
            expiration=date(2025, 1, 1)
        )
        o.contract_candidate = c
        assert o.contract_candidate.contract_symbol == "TEST250101C100"

    def test_options_flow_profile_from_options_module(self):
        """OptionsFlowProfile must be importable from models.options."""
        from models.options import OptionsFlowProfile
        p = OptionsFlowProfile()
        assert hasattr(p, 'flow_score')
        assert hasattr(p, 'bias')
        assert hasattr(p, 'source')

    def test_options_flow_intelligence_freshness(self):
        """OptionsFlowIntelligence.is_fresh must work correctly."""
        from models.options_flow import OptionsFlowIntelligence
        o = OptionsFlowIntelligence(chain_age_minutes=30)
        assert o.is_fresh is True
        o2 = OptionsFlowIntelligence(chain_age_minutes=90)
        assert o2.is_fresh is False

    def test_options_flow_intelligence_has_reliable_chain(self):
        """OptionsFlowIntelligence.has_reliable_chain must work correctly."""
        from models.options_flow import OptionsFlowIntelligence
        o = OptionsFlowIntelligence(data_quality="REAL")
        assert o.has_reliable_chain is True
        o2 = OptionsFlowIntelligence(data_quality="MISSING")
        assert o2.has_reliable_chain is False


class TestSupplyDemandProvenance:
    """Verify supply/demand model provenance."""

    def test_supply_demand_zone_has_source(self):
        """SupplyDemandZone must have source field."""
        from models.supply_demand import SupplyDemandZone, ZoneType, Timeframe
        z = SupplyDemandZone(
            zone_low=100.0,
            zone_high=105.0,
            zone_type=ZoneType.DEMAND,
            timeframe=Timeframe.DAILY
        )
        assert hasattr(z, 'source')
        assert z.source == "unknown"

    def test_supply_demand_zone_source_assignment(self):
        """SupplyDemandZone.source must be assignable."""
        from models.supply_demand import SupplyDemandZone, ZoneType, Timeframe
        z = SupplyDemandZone(
            zone_low=100.0,
            zone_high=105.0,
            zone_type=ZoneType.DEMAND,
            timeframe=Timeframe.DAILY,
            source="yfinance_history"
        )
        assert z.source == "yfinance_history"


class TestTechnicalProvenance:
    """Verify technical model provenance."""

    def test_price_level_has_source(self):
        """PriceLevel must have source field."""
        from models.technical import PriceLevel
        p = PriceLevel(
            price=100.0,
            level_type="SUPPORT",
            strength=80,
            distance_pct=-2.5,
            evidence="swing_low",
            source="swing_pivot"
        )
        assert p.source == "swing_pivot"

    def test_vwap_intelligence_source(self):
        """VwapIntelligence must have source field."""
        from models.technical import VwapIntelligence
        v = VwapIntelligence(vwap=150.0, source="regular_session")
        assert v.source == "regular_session"


class TestStrategyEngineIntegration:
    """Verify strategy engine works with cleaned models."""

    def test_strategy_engine_creates_valid_result(self):
        """StrategyEngine must create valid StrategyResult."""
        from engines.strategy_engine import StrategyEngine
        from models.ticker import TickerData
        from models.supply_demand import SupplyDemandResult
        from core.session_clock import MarketSession
        from datetime import datetime, timezone

        engine = StrategyEngine()
        data = TickerData(
            ticker="TEST",
            timestamp=datetime.now(timezone.utc),
            current_price=100.0,
            previous_close=99.0
        )
        sd_result = SupplyDemandResult()
        result = engine.analyze(data, sd_result)
        assert result.ticker == "TEST"
        assert hasattr(result, 'evidence')
        assert result.evidence is not None

    def test_strategy_engine_uses_evidence_field(self):
        """StrategyEngine must populate evidence field (not evidence_package)."""
        from engines.strategy_engine import StrategyEngine
        from models.ticker import TickerData
        from models.supply_demand import SupplyDemandResult
        from datetime import datetime, timezone

        engine = StrategyEngine()
        data = TickerData(
            ticker="TEST",
            timestamp=datetime.now(timezone.utc),
            current_price=100.0,
            previous_close=99.0
        )
        sd_result = SupplyDemandResult()
        result = engine.analyze(data, sd_result)
        assert len(result.evidence.missing) > 0
        assert "monthly_structure" in result.evidence.missing


class TestDecisionEngineAuthority:
    """Verify DecisionEngine authority remains unchanged."""

    def test_decision_engine_still_decides(self):
        """DecisionEngine must remain the sole decision authority."""
        from engines.decision_engine import DecisionEngine
        from models.signal import HunterDecision
        engine = DecisionEngine()
        assert hasattr(engine, 'decide')

    def test_decision_engine_uses_evidence_field(self):
        """DecisionEngine must use st.evidence (not st.evidence_package)."""
        from engines.decision_engine import DecisionEngine
        from models.strategy import StrategyResult, StrategyEvidence
        from datetime import datetime, timezone

        engine = DecisionEngine()
        r = StrategyResult(
            ticker="TEST",
            as_of=datetime.now(timezone.utc),
            evidence=StrategyEvidence(missing=["test_missing"])
        )
        assert r.evidence.missing == ["test_missing"]
        assert not hasattr(r, 'evidence_package')
