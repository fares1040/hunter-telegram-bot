"""Regression tests for RR21 Options Flow Intelligence."""
import pytest
from datetime import date, datetime, timezone
from models.options import OptionContract, OptionsSnapshot
from models.options_flow import (
    OptionsFlowIntelligence, OptionsFlowProfile, OptionsFlowScore,
    OptionsFlowComponent, OptionChainMetrics, OptionsDataQuality,
    OptionsFlowBias, OptionsDataFreshness
)
from engines.options_flow_engine import OptionsFlowEngine


class TestOptionsFlowEngine:
    """Tests for Options Flow Engine (RR21)."""

    def test_empty_chain(self):
        """Empty chain returns MISSING quality and no metrics."""
        engine = OptionsFlowEngine()
        result = engine.build(None, 100.0)

        assert result.data_quality == "MISSING"
        assert result.metrics is None
        assert "No options snapshot provided" in result.notes

    def test_empty_chain_snapshot(self):
        """Empty chain snapshot returns MISSING quality."""
        engine = OptionsFlowEngine()
        snapshot = OptionsSnapshot(ticker="TEST", underlying_price=100.0, contracts=[], source="test")
        result = OptionsFlowEngine().build(snapshot, 100.0)

        assert result.data_quality == "MISSING"
        assert result.metrics is not None
        assert result.metrics.data_quality == "MISSING"
        assert "no_contracts" in result.metrics.missing_fields

    def test_chain_with_data(self):
        """Chain with valid data produces REAL quality."""
        from models.options import OptionContract
        engine = OptionsFlowEngine()

        contracts = [
            OptionContract(
                ticker="TEST", contract_symbol="TEST240119C00100000",
                contract_type="CALL", strike=100.0, expiration=date(2024, 1, 19),
                bid=1.0, ask=1.1, last=1.05, volume=1000, open_interest=500,
                implied_volatility=0.25, source="test"
            ),
        ]

        snapshot = OptionsSnapshot(
            ticker="TEST", underlying_price=100.0, contracts=contracts, source="test"
        )
        result = OptionsFlowEngine().build(snapshot, 100.0)

        assert result.data_quality == "REAL"
        assert result.metrics is not None
        assert result.metrics.call_volume > 0
        assert result.metrics.put_volume == 0  # No PUT contracts in test data

    def test_options_flow_profile_creation(self):
        """Test OptionsFlowProfile creation with various states."""
        from models.options_flow import OptionsFlowProfile, OptionsFlowIntelligence
        from models.options import OptionsSnapshot

        # Test empty profile
        profile = OptionsFlowProfile()
        assert profile.bias == "NEUTRAL"
        assert profile.confidence == 0
        assert profile.source == "none"
        assert profile.notes == []

        # Test with unavailable chain
        profile = OptionsFlowProfile(notes=["Options chain unavailable"], confidence=0, source="none")
        assert "Options chain unavailable" in profile.notes

    def test_options_flow_intelligence_creation(self):
        """Test OptionsFlowIntelligence creation."""
        from models.options_flow import OptionsFlowIntelligence
        from datetime import datetime, timezone

        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0
        )
        assert intel.ticker == "AAPL"
        assert intel.underlying_price == 150.0
        assert intel.bias == "NEUTRAL"
        assert intel.flow_score == 0
        assert intel.data_quality == "UNKNOWN"
        assert intel.freshness == "UNKNOWN"

    def test_options_flow_intelligence_creation(self):
        """Test OptionsFlowIntelligence with various parameters."""
        from models.options_flow import OptionsFlowIntelligence
        from datetime import datetime, timezone

        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0
        )
        assert intel.ticker == "AAPL"
        assert intel.underlying_price == 150.0
        assert intel.bias == "NEUTRAL"
        assert intel.flow_score == 0
        assert intel.data_quality == "UNKNOWN"
        assert intel.freshness == "UNKNOWN"

    def test_options_flow_intelligence_freshness(self):
        """Test freshness determination."""
        from models.options_flow import OptionsFlowIntelligence
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=now,
            underlying_price=150.0,
            chain_age_minutes=30
        )
        assert intel.chain_age_minutes == 30
        assert intel.is_fresh == True

        # Test stale
        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0,
            chain_age_minutes=90
        )
        assert intel.chain_age_minutes == 90
        assert intel.is_fresh == False

    def test_options_flow_quality(self):
        """Test OptionsFlowIntelligence quality assessment."""
        from models.options_flow import OptionsFlowIntelligence
        from datetime import datetime, timezone

        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0,
            data_quality="REAL"
        )
        assert intel.data_quality == "REAL"

        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0,
            data_quality="MISSING"
        )
        assert intel.data_quality == "MISSING"

    def test_no_fabrication(self):
        """Ensure no fabricated data is created."""
        from engines.options_flow_engine import OptionsFlowEngine
        engine = OptionsFlowEngine()

        # Empty snapshot should return MISSING
        result = OptionsFlowEngine().build(None, 100.0)
        assert result.data_quality == "MISSING"
        assert result.metrics is None
        assert "No options snapshot provided" in result.notes

    def test_chain_age_stale_detection(self):
        """Test stale chain detection."""
        from engines.options_flow_engine import OptionsFlowEngine
        from datetime import datetime, timezone, timedelta
        from models.options import OptionsSnapshot

        engine = OptionsFlowEngine()
        # Create a mock snapshot with old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(minutes=90)
        snapshot = OptionsSnapshot(
            ticker='TEST',
            source='test',
            timestamp=old_time.isoformat(),
            underlying_price=100.0,
            contracts=[]
        )

        result = OptionsFlowEngine().build(snapshot, 100.0)
        # Should detect stale chain
        assert result.chain_age_minutes == 90
        assert result.freshness == "STALE"
        assert any("stale" in w.lower() for w in result.warnings)

    def test_no_fabricated_data(self):
        """Ensure no fabricated data is created."""
        from engines.options_flow_engine import OptionsFlowEngine

        engine = OptionsFlowEngine()
        result = engine.build(None, 100.0)

        # Should not have any fabricated data
        assert result.data_quality == "MISSING"
        assert result.metrics is None
        assert result.bias == "NEUTRAL"
        assert result.flow_score == 0


class TestOptionsFlowModels:
    """Test Options Flow models."""

    def test_option_chain_metrics_creation(self):
        from models.options_flow import OptionChainMetrics

        metrics = OptionChainMetrics(
            call_volume=1000,
            put_volume=500,
            call_open_interest=5000,
            put_open_interest=2500,
            call_premium_volume=100000.0,
            put_premium_volume=50000.0,
            put_call_volume_ratio=0.5,
            put_call_oi_ratio=0.5,
            put_call_premium_ratio=0.5
        )

        assert metrics.call_volume == 1000
        assert metrics.put_volume == 500
        assert metrics.put_call_volume_ratio == 0.5
        assert metrics.data_quality == "REAL"

    def test_options_flow_profile(self):
        from models.options_flow import OptionsFlowProfile

        profile = OptionsFlowProfile(
            call_volume=1000,
            put_volume=500,
            call_open_interest=5000,
            put_open_interest=2500,
            flow_score=75,
            bias="CALL_LEAN",
            confidence=80
        )

        assert profile.call_volume == 1000
        assert profile.bias == "CALL_LEAN"
        assert profile.flow_score == 75
        assert profile.confidence == 80

    def test_options_flow_intelligence(self):
        from models.options_flow import OptionsFlowIntelligence
        from datetime import datetime, timezone

        intel = OptionsFlowIntelligence(
            ticker="AAPL",
            as_of=datetime.now(timezone.utc),
            underlying_price=150.0,
            bias="CALL_LEAN",
            flow_score=75,
            bias_confidence=80
        )

        assert intel.ticker == "AAPL"
        assert intel.bias == "CALL_LEAN"
        assert intel.flow_score == 75
        assert intel.bias_confidence == 80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])