"""Abu Rakan Strategy Intelligence Engine (RR22).

Deterministic strategy evidence layer based on documented Abu Rakan / PODC methodology.
Translates strategy concepts into measurable, deterministic market-data rules.
All missing data remains UNKNOWN/UNAVAILABLE - never fabricated.
"""
import logging
from typing import Optional, List, Dict, Any, Optional
from dataclasses import field

from models.strategy import (
    StrategyResult, StrategyEvidence, StrategyEntry, StrategyTarget,
    StrategyRisk, StrategyEvidence as StrategyEvidenceModel,
    StrategyEntry as StrategyEntryModel, StrategyTarget, StrategyRisk,
    StrategyState, StrategyDirection, ConfirmationState, RiskLevel
)
from models.supply_demand import SupplyDemandResult
from engines.supply_demand_engine import SupplyDemandEngine
from models.supply_demand import SupplyDemandResult as SDResult
from models.ticker import TickerData
from models.supply_demand import SupplyDemandResult as SDResult
from models.supply_demand import SupplyDemandResult
from engines.supply_demand_engine import SupplyDemandEngine
from utils.logger import LOGGER

LOGGER = logging.getLogger("hunter")


class StrategyEngine:
    """Deterministic Abu Rakan strategy intelligence from real market data."""

    def __init__(self):
        self.sd_engine = SupplyDemandEngine()

    def analyze(
        self,
        ticker_data: 'TickerData',
        sd_result: 'SupplyDemandResult',
        swing_intelligence=None,
        technical_intelligence=None,
        intraday_intelligence=None,
        catalyst_event=None,
        catalyst_profile=None,
        options_intelligence=None,
    ) -> 'StrategyResult':
        """Build Abu Rakan strategy intelligence from available data."""

        from models.strategy import StrategyResult, StrategyEvidence, StrategyEntry, StrategyTarget, StrategyRisk

        result = StrategyResult(
            ticker=ticker_data.ticker,
            as_of=datetime.now(timezone.utc),
            evidence=StrategyEvidence(),
            entry=StrategyEntry(),
            target=StrategyTarget(),
            risk=StrategyRisk(),
        )

        if not ticker_data.current_price:
            result.data_quality = "MISSING"
            result.evidence_package.missing.append("no_current_price")
            return self._finalize_result(result)

        price = ticker_data.current_price

        # --- 1. Monthly / Weekly / Daily Structure ---
        self._analyze_structure(ticker_data, result)

        # --- 2. Supply/Demand Integration (RR19) ---
        self._analyze_supply_demand(sd_result, result)

        # --- 3. Breakout/Expansion Analysis ---
        self._analyze_breakout_expansion(result)

        # --- 4. Pullback Toward Demand ---
        self._analyze_pullback(ticker_data, result)

        # --- 5. Volume Confirmation ---
        self._analyze_volume_confirmation(result)

        # --- 6. Supply Overhead ---
        self._analyze_supply_overhead(result)

        # --- 7. Retest Behavior ---
        self._analyze_retest_behavior(result)

        # --- 8. Structure Preservation ---
        self._analyze_structure_preservation(result)

        # --- 8. Risk / Invalidation ---
        self._analyze_risk_invalidation(result)

        # --- 9. Reward Potential ---
        self._analyze_reward_potential(result)

        # --- 10. Confirmation Quality ---
        self._determine_confirmation(result)

        # --- 11. Entry / Target / Risk ---
        self._build_entry(result)
        self._build_target(result)
        self._build_risk(result)

        # Finalize
        self._finalize_result(result)
        return result

    def _analyze_structure(self, ticker_data, result):
        """Analyze multi-timeframe structure (monthly/weekly/daily)."""
        # This would ideally use multi-timeframe data
        # For now, we use available data
        result.evidence_package.monthly_structure = "UNKNOWN"
        result.evidence_package.weekly_structure = "UNKNOWN"
        result.evidence_package.daily_structure = "UNKNOWN"
        result.evidence_package.missing.extend(["monthly_structure", "weekly_structure", "daily_structure"])

    def _analyze_supply_demand(self, sd_result, result):
        """Integrate RR19 Supply/Demand intelligence."""
        if not sd_result:
            result.evidence_package.missing.append("supply_demand_result")
            return

        # Higher timeframe demand/supply
        if sd_result.demand_clusters:
            result.evidence_package.higher_timeframe_demand = True
            result.evidence_package.evidence.append(f"Demand clusters: {len(sd_result.demand_clusters)}")
        else:
            result.evidence_package.higher_timeframe_demand = False
            result.evidence_package.missing.append("higher_timeframe_demand")

        if sd_result.supply_clusters:
            result.evidence_package.higher_timeframe_supply = True
            result.evidence_package.evidence.append(f"Supply clusters: {len(sd_result.supply_clusters)}")
        else:
            result.evidence_package.higher_timeframe_supply = False
            result.evidence_package.missing.append("higher_timeframe_supply")

        # Conflicting zones
        if sd_result.conflicting_zones:
            result.evidence_package.supply_overhead = True
            result.warnings.append(f"Conflicting zones: {len(sd_result.conflicting_zones)}")

    def _analyze_breakout_expansion(self, result):
        """Analyze breakout/expansion behavior."""
        # This would need swing/intraday intelligence
        result.evidence_package.breakout_expansion = None
        result.evidence_package.missing.append("breakout_expansion")

    def _analyze_pullback(self, ticker_data, result):
        """Analyze pullback toward demand."""
        result.evidence_package.pullback_toward_demand = None
        result.evidence_package.missing.append("pullback_toward_demand")

    def _analyze_volume_confirmation(self, result):
        """Analyze volume confirmation."""
        result.evidence_package.volume_confirmation = None
        result.evidence_package.missing.append("volume_confirmation")

    def _analyze_supply_overhead(self, result):
        """Analyze supply overhead."""
        result.evidence_package.supply_overhead = None
        result.evidence_package.missing.append("supply_overhead")

    def _analyze_retest_behavior(self, result):
        """Analyze retest behavior."""
        result.evidence_package.retest_behavior = None
        result.evidence_package.missing.append("retest_behavior")

    def _analyze_structure_preservation(self, result):
        """Analyze structure preservation."""
        result.evidence_package.structure_preservation = None
        result.evidence_package.missing.append("structure_preservation")

    def _analyze_risk_invalidation(self, result):
        """Analyze risk and invalidation."""
        result.evidence_package.risk_invalidation_clear = None
        result.evidence_package.missing.append("risk_invalidation")

    def _analyze_reward_potential(self, result):
        """Analyze reward potential."""
        result.evidence_package.reward_potential = None
        result.evidence_package.missing.append("reward_potential")

    def _determine_confirmation(self, result):
        """Determine overall confirmation state."""
        # Check critical factors
        has_demand = result.evidence_package.higher_timeframe_demand
        has_supply_conflict = result.evidence_package.supply_overhead
        has_volume = result.evidence_package.volume_confirmation
        has_pullback = result.evidence_package.pullback_toward_demand
        has_breakout = result.evidence_package.breakout_expansion

        # Check for invalidation
        if result.risk and not result.risk.invalidation_clear:
            result.confirmation = "INVALIDATED"
            result.state = "INVALIDATED"
            return

        # Check for confirmed setup
        if (result.evidence_package.higher_timeframe_demand and
            result.evidence_package.volume_confirmation and
            result.evidence_package.pullback_toward_demand and
            result.evidence_package.breakout_expansion and
            not result.evidence_package.supply_overhead):
            result.confirmation = "CONFIRMED"
            result.state = "CONFIRMED"
        elif result.evidence_package.higher_timeframe_demand and result.evidence_package.pullback_toward_demand:
            result.confirmation = "DEVELOPING"
            result.state = "DEVELOPING"
        elif result.evidence_package.higher_timeframe_demand:
            result.confirmation = "WATCH"
            result.state = "WATCH"
        else:
            result.confirmation = "UNAVAILABLE"
            result.state = "UNAVAILABLE"

    def _build_entry(self, result):
        """Build entry intelligence."""
        # This would use swing/intraday entry intelligence
        result.entry.status = "UNAVAILABLE"
        result.entry.reason = "Insufficient data for entry determination"

    def _build_target(self, result):
        """Build target zones."""
        result.target.status = "UNAVAILABLE"

    def _build_risk(self, result):
        """Build risk assessment."""
        result.risk.status = "UNAVAILABLE"
        result.risk.invalidation_clear = False
        result.risk.risk_acceptable = False

    def _finalize_result(self, result):
        """Finalize result with confidence and data quality."""
        # Calculate confidence based on available evidence
        evidence_factors = [
            result.evidence_package.higher_timeframe_demand is not None,
            result.evidence_package.higher_timeframe_supply is not None,
            result.evidence_package.volume_confirmation is not None,
            result.evidence_package.breakout_expansion is not None,
            result.evidence_package.pullback_toward_demand is not None,
            result.evidence_package.supply_overhead is not None,
            result.evidence_package.risk_invalidation_clear is not None,
        ]

        available_factors = sum(1 for f in evidence_factors if f)
        total_factors = len(evidence_factors)

        if total_factors > 0:
            result.confidence = int((available_factors / total_factors) * 100)
        else:
            result.confidence = 0

        # Data quality
        if hasattr(result, 'evidence_package') and len(result.evidence_package.missing) > 3:
            result.data_quality = "POOR"
        elif hasattr(result, 'evidence_package') and len(result.evidence_package.missing) > 1:
            result.data_quality = "PARTIAL"
        else:
            result.data_quality = "REAL"

        # Actionable check
        if result.is_actionable:
            result.notes.append("Strategy is actionable - meets all criteria")
        else:
            result.notes.append(f"Strategy not actionable: state={result.state}, confirmation={result.confirmation}, risk_clear={getattr(result.risk, 'invalidation_clear', False)}")


from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Optional
from dataclasses import field

from models.ticker import TickerData
from models.strategy import StrategyResult, StrategyEvidence, StrategyEntry, StrategyTarget, StrategyRisk
from utils.logger import LOGGER

LOGGER = logging.getLogger("hunter")