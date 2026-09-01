"""Options Flow Intelligence Engine (RR21).

Deterministic options flow analysis from real market data.
Separates observable chain data from inferred flow intelligence.
All missing data remains UNKNOWN/UNAVAILABLE - never fabricated.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from models.news import ensure_utc

from models.options import OptionContract, OptionsSnapshot, OptionsFlowProfile
from models.options_flow import (
    OptionsFlowIntelligence, OptionsFlowScore,
    OptionsFlowComponent, OptionChainMetrics
)
from utils.logger import LOGGER

LOGGER = logging.getLogger("hunter")


# Configuration constants
MIN_CHAIN_AGE_MINUTES = 5
MAX_CHAIN_AGE_MINUTES = 120
MIN_VOLUME_FOR_ANALYSIS = 100
MIN_OI_FOR_ANALYSIS = 100
UNUSUAL_VOLUME_MULTIPLE = 3.0
UNUSUAL_OI_MULTIPLE = 2.0
STALE_CHAIN_MINUTES = 60


class OptionsFlowEngine:
    """Deterministic options flow analysis from real market data."""

    def __init__(self):
        pass

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse ISO format timestamp string to datetime."""
        try:
            return ensure_utc(datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')))
        except Exception:
            return None

    def build(
        self,
        options_snapshot: Optional[OptionsSnapshot],
        underlying_price: Optional[float],
        technical_intelligence=None,
        true_flow_trades: Optional[List] = None,
        true_flow_max_age: int = 30,
    ) -> OptionsFlowIntelligence:
        """Build options flow intelligence from real chain data."""
        result = OptionsFlowIntelligence()
        result.underlying_price = underlying_price

        if options_snapshot is None:
            return OptionsFlowIntelligence(
                ticker="UNKNOWN",
                underlying_price=underlying_price,
                data_quality="MISSING",
                notes=["No options snapshot provided"]
            )

        result.ticker = options_snapshot.ticker if hasattr(options_snapshot, 'ticker') else "UNKNOWN"
        result.chain_source = options_snapshot.source if hasattr(options_snapshot, 'source') else "unknown"
        result.chain_timestamp = self._parse_timestamp(options_snapshot.timestamp) if hasattr(options_snapshot, 'timestamp') and options_snapshot.timestamp else None
        result.underlying_price = underlying_price or (options_snapshot.underlying_price if hasattr(options_snapshot, 'underlying_price') else None)

        # Calculate chain age
        if result.chain_timestamp:
            chain_age = (datetime.now(timezone.utc) - ensure_utc(result.chain_timestamp)).total_seconds() / 60
            result.chain_age_minutes = int(chain_age)
            if chain_age > MAX_CHAIN_AGE_MINUTES:
                result.data_quality = "STALE"
                result.warnings.append(f"Options chain is {int(chain_age)} minutes old (exceeds max age)")
            elif chain_age > STALE_CHAIN_MINUTES:
                result.freshness = "STALE"
                result.warnings.append(f"Options chain is {int(chain_age)} minutes old (stale)")
            else:
                result.freshness = "FRESH"
        else:
            result.chain_age_minutes = None

        # Check chain availability
        if not hasattr(options_snapshot, 'contracts') or not options_snapshot.contracts:
            result.data_quality = "MISSING"
            result.notes.append("No options chain available")
            contracts = []
        else:
            contracts = options_snapshot.contracts if hasattr(options_snapshot, 'contracts') else []

        # Build metrics from real chain data
        metrics = self._analyze_chain(contracts, underlying_price)
        result.metrics = metrics

        # Score flow
        flow_profile = self._analyze_flow(contracts, underlying_price)
        result.bias = flow_profile.bias
        result.flow_score = flow_profile.flow_score
        result.bias_confidence = flow_profile.confidence

        # Find contract candidate
        result.contract_candidate = self._find_best_contract(contracts, underlying_price)

        # Additive True Flow aggregation (never double-count with snapshot volume)
        if true_flow_trades:
            try:
                from models.option_realtime import aggregate_true_flow
                tf = aggregate_true_flow(true_flow_trades, max_age=true_flow_max_age)
                # Attach as additive evidence — do not overwrite snapshot metrics
                if tf.total_trades > 0:
                    result.notes.append(f"TRUE_FLOW: {tf.call_trades}C/{tf.put_trades}P trades call_prem={tf.call_premium:.0f} put_prem={tf.put_premium:.0f} largest={tf.largest_contract or 'none'}:{tf.largest_premium:.0f}")
                    if tf.large_prints:
                        result.notes.append(f"TRUE_FLOW large_prints={len(tf.large_prints)}")
                    if tf.repeated_contracts:
                        result.notes.append(f"TRUE_FLOW repeated={len(tf.repeated_contracts)} contracts")
                    # Store on result for downstream consumers (additive field)
                    result.warnings.append(f"TRUE_FLOW provenance={tf.source} FRESH only")
            except Exception as e:
                LOGGER.warning(f"[OptionsFlow] true flow aggregation failed: {e}")

        return self._finalize_intelligence(result)

    def _analyze_chain(self, contracts: List, underlying_price: Optional[float]) -> OptionChainMetrics:
        """Analyze real options chain for metrics."""
        metrics = OptionChainMetrics()

        if not contracts:
            metrics.data_quality = "MISSING"
            metrics.missing_fields.append("no_contracts")
            return metrics

        # Volume metrics
        metrics.call_volume = sum(c.volume or 0 for c in contracts if c.contract_type == "CALL")
        metrics.put_volume = sum(c.volume or 0 for c in contracts if c.contract_type == "PUT")
        metrics.call_open_interest = sum(c.open_interest or 0 for c in contracts if c.contract_type == "CALL")
        metrics.put_open_interest = sum(c.open_interest or 0 for c in contracts if c.contract_type == "PUT")
        metrics.call_premium_volume = sum((c.mid or 0) * (c.volume or 0) for c in contracts if c.contract_type == "CALL")
        metrics.put_premium_volume = sum((c.mid or 0) * (c.volume or 0) for c in contracts if c.contract_type == "PUT")

        if metrics.call_volume > 0:
            metrics.put_call_volume_ratio = round(metrics.put_volume / metrics.call_volume, 3)
        if metrics.call_open_interest > 0:
            metrics.put_call_oi_ratio = round(metrics.put_open_interest / metrics.call_open_interest, 3)
        if metrics.call_premium_volume > 0:
            metrics.put_call_premium_ratio = round(metrics.put_premium_volume / metrics.call_premium_volume, 3)

        # Unusual volume detection
        for c in contracts:
            if c.volume and c.volume > MIN_VOLUME_FOR_ANALYSIS * UNUSUAL_VOLUME_MULTIPLE:
                metrics.unusual_volume_detected = True
                metrics.unusual_volume_strikes.append({
                    "strike": c.strike,
                    "type": c.contract_type,
                    "volume": c.volume,
                    "multiple": round(c.volume / MIN_VOLUME_FOR_ANALYSIS, 1)
                })

            if c.open_interest and c.open_interest > MIN_OI_FOR_ANALYSIS * UNUSUAL_OI_MULTIPLE:
                metrics.unusual_oi_detected = True
                metrics.high_oi_strikes.append({
                    "strike": c.strike,
                    "type": c.contract_type,
                    "oi": c.open_interest,
                    "multiple": round(c.open_interest / MIN_OI_FOR_ANALYSIS, 1)
                })

        # Expiration concentration
        exp_counts: Dict[str, int] = {}
        for c in contracts:
            if c.expiration:
                exp_key = str(c.expiration)
                exp_counts[exp_key] = exp_counts.get(exp_key, 0) + 1
        metrics.expiration_concentration = exp_counts

        # Strike concentration
        strike_counts: Dict[float, int] = {}
        for c in contracts:
            strike_key = round(c.strike, 2)
            strike_counts[strike_key] = strike_counts.get(strike_key, 0) + 1
        metrics.strike_concentration = {str(k): v for k, v in strike_counts.items()}

        # IV analysis
        ivs = [c.implied_volatility for c in contracts if c.implied_volatility and c.implied_volatility > 0]
        if ivs:
            metrics.atm_iv = self._calculate_atm_iv(contracts)
            metrics.iv_skew = self._calculate_iv_skew(contracts)
            metrics.iv_skew_slope = self._calculate_iv_skew_slope(contracts)

        # Bid/ask spread quality
        spreads = []
        for c in contracts:
            if c.bid and c.ask and c.mid and c.mid > 0:
                spread_pct = c.spread_pct
                if spread_pct is not None:
                    spreads.append(spread_pct)
        if spreads:
            metrics.bid_ask_spread_quality = sum(spreads) / len(spreads)

        # Data quality assessment
        if contracts:
            metrics.data_quality = "REAL"
        else:
            metrics.data_quality = "MISSING"
            metrics.missing_fields.append("no_contracts")

        # Add missing fields
        if not any(c.volume for c in contracts):
            metrics.missing_fields.append("volume")
        if not any(c.open_interest for c in contracts):
            metrics.missing_fields.append("open_interest")
        if not any(c.implied_volatility for c in contracts):
            metrics.missing_fields.append("implied_volatility")
        if not any(c.bid and c.ask for c in contracts):
            metrics.missing_fields.append("bid_ask")

        return metrics

    def _calculate_atm_iv(self, contracts: List) -> Optional[float]:
        """Calculate ATM implied volatility from chain."""
        ivs = [c.implied_volatility for c in contracts if c.implied_volatility and c.implied_volatility > 0]
        if not ivs:
            return None
        return sum(ivs) / len(ivs)

    def _calculate_iv_skew(self, contracts: List) -> Optional[float]:
        """Calculate IV skew (25 delta put IV - 25 delta call IV approximation)."""
        puts = [c for c in contracts if c.contract_type == "PUT" and c.implied_volatility]
        calls = [c for c in contracts if c.contract_type == "CALL" and c.implied_volatility]
        if not puts or not calls:
            return None
        put_iv = sum(p.implied_volatility for p in puts) / len(puts)
        call_iv = sum(c.implied_volatility for c in calls) / len(calls)
        return round(put_iv - call_iv, 4)

    def _calculate_iv_skew_slope(self, contracts: List) -> Optional[float]:
        """Calculate IV skew slope across strikes."""
        return None

    def _analyze_flow(self, contracts: List, underlying_price: Optional[float]) -> OptionsFlowProfile:
        """Analyze options flow from real chain data."""
        if not contracts:
            return OptionsFlowProfile(notes=["Options chain unavailable"], confidence=0, source="none")

        from engines.options_engine import OptionsEngine

        engine = OptionsEngine()
        snapshot = OptionsSnapshot(
            ticker="TEMP",
            underlying_price=0,
            contracts=contracts,
            source="options_flow_engine"
        )
        return engine.analyze(snapshot, underlying_price or 0)

    def _find_best_contract(self, contracts: List, underlying_price: Optional[float]) -> Optional[OptionContract]:
        """Find the best contract candidate based on liquidity and moneyness."""
        if not contracts or not underlying_price:
            return None

        # Filter for liquid contracts
        candidates = []
        for c in contracts:
            if c.volume and c.volume >= MIN_VOLUME_FOR_ANALYSIS and c.open_interest and c.open_interest >= MIN_OI_FOR_ANALYSIS:
                moneyness = c.moneyness(underlying_price)
                if moneyness is not None and -20 <= moneyness <= 20:
                    candidates.append(c)

        if not candidates:
            return None

        # Score by liquidity and moneyness
        best = None
        best_score = -1
        for c in candidates:
            score = 0
            if c.volume:
                score += min(c.volume / 1000, 50)
            if c.open_interest:
                score += min(c.open_interest / 1000, 30)
            if c.mid and c.mid > 0:
                score += 10
            if c.implied_volatility and c.implied_volatility > 0:
                score += 10
            if score > best_score:
                best_score = score
                best = c

        return best

    def _finalize_intelligence(self, result: OptionsFlowIntelligence) -> OptionsFlowIntelligence:
        """Finalize intelligence with data quality, bias, and freshness (single execution path)."""
        # Determine data quality from metrics
        if result.metrics:
            result.data_quality = result.metrics.data_quality
        else:
            result.data_quality = "MISSING"

        # Determine bias from flow score
        if result.flow_score >= 70:
            result.bias = "STRONG_CALL" if result.flow_score > 85 else "CALL_LEAN"
        elif result.flow_score <= 30:
            result.bias = "STRONG_PUT" if result.flow_score < 15 else "PUT_LEAN"
        else:
            result.bias = "NEUTRAL"

        # Freshness from chain age
        if result.chain_age_minutes is not None:
            if result.chain_age_minutes <= MIN_CHAIN_AGE_MINUTES:
                result.freshness = "FRESH"
            elif result.chain_age_minutes <= STALE_CHAIN_MINUTES:
                result.freshness = "STALE"
            else:
                result.freshness = "STALE"

        return result
