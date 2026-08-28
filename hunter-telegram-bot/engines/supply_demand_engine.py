"""Hunter Bot — Supply & Demand Intelligence Engine (RR19).

Deterministic supply/demand zone detection from real market data.
Detects structural zones based on real price/volume behavior.
No fabricated zones, no fabricated data. Missing data -> UNAVAILABLE.
"""
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
import pandas as pd

from models.ticker import TickerData
from models.supply_demand import (
    SupplyDemandZone, ZoneType, ZoneFreshness, ZoneStrength, Timeframe,
    ZoneCluster, ZoneEvidence, SupplyDemandResult, ZoneScoreComponent
)
from utils.logger import LOGGER

LOGGER = logging.getLogger("hunter")


# Configuration constants
MIN_BASE_BARS = 5
MAX_BASE_BARS = 50
MIN_DEPARTURE_PCT = 1.5
STRONG_DEPARTURE_PCT = 3.0
VOLUME_SPIKE_MULTIPLE = 1.5
RETEST_TOLERANCE_PCT = 0.5
MAX_LOOKBACK = 250
MIN_ZONE_HEIGHT_PCT = 0.2
MAX_ZONE_HEIGHT_PCT = 5.0
FRESHNESS_TESTED_THRESHOLD = 2
FRESHNESS_WEAKENED_THRESHOLD = 4
STRONG_DEPARTURE_MULTIPLE = 2.0


class SupplyDemandEngine:
    """Deterministic supply/demand zone detection from real market data."""

    def __init__(self):
        pass

    def build(
        self,
        data: TickerData,
        daily_history: Optional[pd.DataFrame] = None,
        weekly_history: Optional[pd.DataFrame] = None,
        monthly_history: Optional[pd.DataFrame] = None,
        intraday_intelligence=None,
    ) -> 'SupplyDemandResult':
        """Build supply/demand intelligence from multi-timeframe data."""
        result = SupplyDemandResult()

        price = data.current_price
        if price is None:
            result.missing_data.append("no_current_price")
            result.data_quality = "MISSING"
            return result

        timeframes = [
            (daily_history, Timeframe.DAILY, "1d"),
            (weekly_history, Timeframe.WEEKLY, "1wk"),
            (monthly_history, Timeframe.MONTHLY, "1mo"),
        ]

        all_demand_zones = []
        all_supply_zones = []

        for history, tf, tf_label in timeframes:
            if history is None or len(history) < 20:
                continue

            zones = self._detect_zones(history, price, tf, tf_label)
            for zone in zones:
                if zone.zone_type == ZoneType.DEMAND:
                    all_demand_zones.append(zone)
                else:
                    all_supply_zones.append(zone)

        valid_demand = [z for z in all_demand_zones if self._validate_zone(z, price)]
        valid_supply = [z for z in all_supply_zones if self._validate_zone(z, price)]

        valid_demand.sort(key=lambda z: (z.strength.value, -z.confidence), reverse=True)
        valid_supply.sort(key=lambda z: (z.strength.value, -z.confidence), reverse=True)

        demand_clusters = self._cluster_zones(valid_demand)
        supply_clusters = self._cluster_zones(valid_supply)

        nearest_demand = self._find_nearest_zone(valid_demand, price, ZoneType.DEMAND)
        nearest_supply = self._find_nearest_zone(valid_supply, price, ZoneType.SUPPLY)

        conflicting = self._detect_conflicts(valid_demand, valid_supply, price)

        result = SupplyDemandResult(
            demand_zones=valid_demand,
            supply_zones=valid_supply,
            demand_clusters=[c for c in demand_clusters if c.zones],
            supply_clusters=[c for c in supply_clusters if c.zones],
            nearest_demand=nearest_demand,
            nearest_supply=nearest_supply,
            conflicting_zones=conflicting,
        )

        result.dominant_zone_type = self._determine_dominant_zone(
            nearest_demand, nearest_supply, price
        )

        if nearest_demand:
            result.nearest_zone_distance_pct = nearest_demand.distance_from_price(price)
            result.dominant_zone_strength = nearest_demand.strength.value
        elif nearest_supply:
            result.nearest_zone_distance_pct = nearest_supply.distance_from_price(price)
            result.dominant_zone_strength = nearest_supply.strength.value

        result.data_quality = "REAL" if result.demand_zones or result.supply_zones else "MISSING"

        return result

    def _detect_zones(
        self,
        history: pd.DataFrame,
        current_price: float,
        timeframe: Timeframe,
        tf_label: str
    ) -> List[SupplyDemandZone]:
        zones = []
        df = self._normalize_history(history)
        if df is None or len(df) < 20:
            return []

        price = history["Close"].iloc[-1]
        highs = history["High"]
        lows = history["Low"]
        closes = history["Close"]
        volumes = history["Volume"]

        pivot_highs = self._find_pivot_highs(highs)
        pivot_lows = self._find_pivot_lows(lows)

        demand_zones = self._find_demand_zones(
            df, price, pivot_lows, pivot_highs, volumes, closes, Timeframe.DAILY
        )

        supply_zones = self._find_supply_zones(
            df, price, pivot_lows, pivot_highs, volumes, closes, Timeframe.DAILY
        )

        zones.extend(demand_zones)
        zones.extend(supply_zones)

        return zones

    def _find_demand_zones(
        self,
        df: pd.DataFrame,
        current_price: float,
        pivot_lows: List[int],
        pivot_highs: List[int],
        volumes: pd.Series,
        closes: pd.Series,
        timeframe: Timeframe
    ) -> List[SupplyDemandZone]:
        zones = []
        closes = df["Close"]

        for i, pivot_idx in enumerate(pivot_lows):
            if pivot_idx >= len(df) - 1:
                continue

            next_high_idx = None
            for ph in pivot_highs:
                if ph > pivot_idx:
                    next_high_idx = ph
                    break

            if next_high_idx is None or next_high_idx <= pivot_idx + 2:
                continue

            base_start = max(0, pivot_idx - 5)
            base_end = pivot_idx
            base_data = df.iloc[base_start:base_end + 1]

            if len(base_data) < MIN_BASE_BARS or len(base_data) > MAX_BASE_BARS:
                continue

            base_high = base_data["High"].max()
            base_low = base_data["Low"].min()
            base_mid = (base_high + base_low) / 2
            base_height_pct = ((base_high - base_low) / base_mid) * 100

            if base_height_pct < MIN_ZONE_HEIGHT_PCT or base_height_pct > MAX_ZONE_HEIGHT_PCT:
                continue

            next_high_idx = None
            for ph in pivot_highs:
                if ph > pivot_idx:
                    next_high_idx = ph
                    break

            if next_high_idx is None or next_high_idx >= len(df):
                continue

            departure_price = df["Close"].iloc[next_high_idx]
            base_price = base_data["Close"].mean()
            departure_pct = ((departure_price - base_price) / base_price) * 100

            if departure_pct < MIN_DEPARTURE_PCT:
                continue

            base_volumes = df["Volume"].iloc[base_start:base_end + 1]
            avg_base_volume = base_volumes.mean()
            departure_volume = df["Volume"].iloc[next_high_idx] if next_high_idx < len(df) else 0
            volume_ratio = departure_volume / avg_base_volume if avg_base_volume > 0 else 1.0

            volume_confirmed = volume_ratio >= VOLUME_SPIKE_MULTIPLE

            zone_low = base_low
            zone_high = base_high

            departure_pct = ((df["Close"].iloc[min(next_high_idx + 5, len(df)-1)] - base_price) / base_price) * 100 if next_high_idx + 5 < len(df) else departure_pct
            departure_strength = "STRONG" if departure_pct >= STRONG_DEPARTURE_PCT else "MODERATE"

            zone = SupplyDemandZone(
                zone_low=round(base_low, 2),
                zone_high=round(base_high, 2),
                zone_type=ZoneType.DEMAND,
                timeframe=Timeframe.DAILY,
                base_low=round(base_low, 2),
                base_high=round(base_high, 2),
                departure_price=round(departure_price, 2),
                departure_pct=round(departure_pct, 2),
                departure_bars=len(base_data) if len(base_data) > 0 else None,
                volume_on_departure=departure_volume if departure_volume > 0 else None,
                avg_volume_in_base=round(avg_base_volume, 0) if avg_base_volume > 0 else None,
                volume_ratio_on_departure=round(volume_ratio, 2) if volume_ratio > 0 else None,
                evidence=ZoneEvidence(
                    base_formation="base_before_bullish_departure",
                    departure_strength=round(departure_pct, 2),
                    departure_bars=len(base_data),
                    volume_confirmation=volume_ratio if volume_confirmed else None,
                    structural_confirmation=["base_before_bullish_departure"]
                ),
            )

            self._score_zone(zone)
            zones.append(zone)

        return zones

    def _find_supply_zones(
        self,
        df: pd.DataFrame,
        current_price: float,
        pivot_lows: List[int],
        pivot_highs: List[int],
        volumes: pd.Series,
        closes: pd.Series,
        timeframe: Timeframe
    ) -> List[SupplyDemandZone]:
        zones = []
        closes = df["Close"]

        for i, pivot_idx in enumerate(pivot_highs):
            if pivot_idx >= len(df) - 1:
                continue

            next_low_idx = None
            for pl in pivot_lows:
                if pl > pivot_idx:
                    next_low_idx = pl
                    break

            if next_low_idx is None or next_low_idx <= pivot_idx + 2:
                continue

            base_start = max(0, pivot_idx - 5)
            base_end = pivot_idx
            base_data = df.iloc[base_start:base_end + 1]

            if len(base_data) < MIN_BASE_BARS or len(base_data) > MAX_BASE_BARS:
                continue

            base_high = base_data["High"].max()
            base_low = base_data["Low"].min()
            base_mid = (base_high + base_low) / 2
            base_height_pct = ((base_high - base_low) / base_mid) * 100

            if base_height_pct < MIN_ZONE_HEIGHT_PCT or base_height_pct > MAX_ZONE_HEIGHT_PCT:
                continue

            next_low_idx = None
            for pl in pivot_lows:
                if pl > pivot_idx:
                    next_low_idx = pl
                    break

            if next_low_idx is None or next_low_idx >= len(df):
                continue

            departure_price = df["Close"].iloc[next_low_idx]
            base_price = base_data["Close"].mean()
            departure_pct = ((base_price - departure_price) / base_price) * 100

            if departure_pct < MIN_DEPARTURE_PCT:
                continue

            base_volumes = df["Volume"].iloc[base_start:base_end + 1]
            avg_base_volume = base_volumes.mean()
            departure_volume = df["Volume"].iloc[next_low_idx] if next_low_idx < len(df) else 0
            volume_ratio = departure_volume / avg_base_volume if avg_base_volume > 0 else 1.0
            volume_confirmed = volume_ratio >= VOLUME_SPIKE_MULTIPLE

            zone = SupplyDemandZone(
                zone_low=round(base_low, 2),
                zone_high=round(base_high, 2),
                zone_type=ZoneType.SUPPLY,
                timeframe=Timeframe.DAILY,
                base_low=round(base_low, 2),
                base_high=round(base_high, 2),
                departure_price=round(departure_price, 2),
                departure_pct=round(departure_pct, 2),
                departure_bars=len(base_data) if len(base_data) > 0 else None,
                volume_on_departure=departure_volume if departure_volume > 0 else None,
                avg_volume_in_base=round(avg_base_volume, 0) if avg_base_volume > 0 else None,
                volume_ratio_on_departure=round(volume_ratio, 2) if volume_ratio > 0 else None,
                evidence=ZoneEvidence(
                    base_formation="base_before_bearish_departure",
                    departure_strength=round(departure_pct, 2),
                    departure_bars=len(base_data),
                    volume_confirmation=volume_ratio if volume_confirmed else None,
                    structural_confirmation=["base_before_bearish_departure"]
                ),
            )

            self._score_zone(zone)
            zones.append(zone)

        return zones

    def _validate_zone(self, zone: SupplyDemandZone, current_price: float) -> bool:
        if zone.zone_type == ZoneType.DEMAND:
            if zone.zone_high > current_price * 1.02:
                zone.freshness = ZoneFreshness.INVALIDATED
                zone.invalidation_reason = "zone_above_price"
                zone.invalidated_at = datetime.now(timezone.utc)
                return False
        else:
            if zone.zone_low < current_price * 0.98:
                zone.freshness = ZoneFreshness.INVALIDATED
                zone.invalidation_reason = "zone_below_price"
                zone.invalidated_at = datetime.now(timezone.utc)
                return False

        # Mark as fresh if valid
        if zone.freshness == ZoneFreshness.UNKNOWN:
            zone.freshness = ZoneFreshness.FRESH

        if zone.freshness == ZoneFreshness.INVALIDATED:
            return False

        return True

    def _score_zone(self, zone: SupplyDemandZone):
        comps = []

        struct_score = 50
        if zone.departure_pct:
            if zone.departure_pct >= STRONG_DEPARTURE_PCT * 2:
                struct_score += 25
            elif zone.departure_pct >= STRONG_DEPARTURE_PCT:
                struct_score += 15
        comps.append(ZoneScoreComponent("Structure", 0.20, struct_score, "departure_strength"))

        dep_score = 50
        if zone.departure_pct:
            if zone.departure_pct >= STRONG_DEPARTURE_PCT * 2:
                dep_score = 90
            elif zone.departure_pct >= STRONG_DEPARTURE_PCT:
                dep_score = 70
            else:
                dep_score = 40
        comps.append(ZoneScoreComponent("DepartureStrength", 0.15, dep_score, "departure_pct"))

        vol_score = 50
        if zone.evidence.volume_confirmation:
            vol_score = 85
        elif zone.volume_ratio_on_departure and zone.volume_ratio_on_departure >= VOLUME_SPIKE_MULTIPLE:
            vol_score = 75
        elif zone.volume_ratio_on_departure:
            vol_score = 40
        else:
            vol_score = 20
        comps.append(ZoneScoreComponent("VolumeConfirmation", 0.15, vol_score, "volume_on_departure"))

        fresh_score = 50
        if zone.freshness == ZoneFreshness.FRESH:
            fresh_score = 90
        elif zone.freshness == ZoneFreshness.TESTED:
            fresh_score = 70
        elif zone.freshness == ZoneFreshness.WEAKENED:
            fresh_score = 30
        elif zone.freshness == ZoneFreshness.INVALIDATED:
            fresh_score = 0
        comps.append(ZoneScoreComponent("Freshness", 0.15, fresh_score, "freshness"))

        retest_score = 50
        if zone.retest_count == 0:
            retest_score = 70
        elif zone.retest_count <= 2:
            retest_score = 60
        elif zone.retest_count <= 4:
            retest_score = 40
        else:
            retest_score = 20
        comps.append(ZoneScoreComponent("RetestHistory", 0.10, retest_score, "retest_count"))

        tf_score = {"MONTHLY": 100, "WEEKLY": 80, "DAILY": 60, "INTRADAY": 40}.get(
            zone.timeframe.value if hasattr(zone.timeframe, 'value') else str(zone.timeframe), 50
        )
        comps.append(ZoneScoreComponent("Timeframe", 0.10, tf_score, "timeframe"))

        struct_conf_score = 50
        if zone.evidence.structural_confirmation:
            struct_conf_score = 70 + min(len(zone.evidence.structural_confirmation) * 5, 25)
        comps.append(ZoneScoreComponent("StructuralConfirmation", 0.05, struct_conf_score, "structural_confirmation"))

        available = [c for c in comps if c.value is not None]
        total_weight = sum(c.weight for c in comps if c.value is not None)
        if total_weight > 0:
            raw_score = sum((c.value or 0) * c.weight for c in comps) / total_weight
            score = max(0, min(100, int(round(raw_score))))
        else:
            score = 0

        zone.confidence = score

        if score >= 80:
            zone.strength = ZoneStrength.STRONG
        elif score >= 60:
            zone.strength = ZoneStrength.MODERATE
        elif score >= 40:
            zone.strength = ZoneStrength.WEAK
        else:
            zone.strength = ZoneStrength.UNKNOWN

        zone.evidence = comps

    def _cluster_zones(self, zones: List[SupplyDemandZone]) -> List[ZoneCluster]:
        if not zones:
            return []

        zones.sort(key=lambda z: z.zone_mid)

        clusters = []
        used = set()

        for i, z1 in enumerate(zones):
            if i in used:
                continue

            cluster = [z1]
            used.add(zones.index(z1))
            z1_mid = z1.zone_mid

            for j, z2 in enumerate(zones):
                if j in used:
                    continue
                if z2.zone_type != z1.zone_type:
                    continue
                if self._zones_overlap(z1, z2):
                    cluster.append(z2)
                    used.add(zones.index(z2))

            if cluster:
                cluster_type = cluster[0].zone_type
                all_low = min(z.zone_low for z in cluster)
                all_high = max(z.zone_high for z in cluster)
                timeframes = [z.timeframe.value if hasattr(z.timeframe, 'value') else str(z.timeframe) for z in cluster]

                avg_conf = sum(z.confidence for z in cluster) / len(cluster)
                if avg_conf >= 80:
                    combined_str = ZoneStrength.STRONG
                elif avg_conf >= 60:
                    combined_str = ZoneStrength.MODERATE
                else:
                    combined_str = ZoneStrength.WEAK

                alignments = set()
                for z in cluster:
                    if z.is_demand:
                        alignments.add("DEMAND")
                    else:
                        alignments.add("SUPPLY")
                alignment = "ALIGNED" if len(alignments) == 1 else "CONFLICTING"

                evidence = []
                for z in cluster:
                    evidence.extend(z.evidence_summary)

                cluster_obj = ZoneCluster(
                    zones=cluster,
                    cluster_type=cluster_type,
                    zone_low=min(z.zone_low for z in cluster),
                    zone_high=max(z.zone_high for z in cluster),
                    timeframes=timeframes,
                    combined_strength=combined_str,
                    combined_confidence=int(sum(z.confidence for z in cluster) / len(cluster)),
                    alignment=alignment,
                    evidence=evidence,
                )
                clusters.append(cluster_obj)

        return clusters

    def _zones_overlap(self, z1: SupplyDemandZone, z2: SupplyDemandZone) -> bool:
        if z1.zone_type != z2.zone_type:
            return False

        overlap_low = max(z1.zone_low, z2.zone_low)
        overlap_high = min(z1.zone_high, z2.zone_high)
        overlap = max(0, overlap_high - overlap_low)

        if overlap <= 0:
            return False

        min_height = min(z1.zone_height, z2.zone_height)
        return overlap / min_height >= 0.3

    def _find_nearest_zone(self, zones: List[SupplyDemandZone], price: float, zone_type: ZoneType) -> Optional[SupplyDemandZone]:
        relevant = [z for z in zones if z.zone_type == zone_type and z.freshness != ZoneFreshness.INVALIDATED]
        if not relevant:
            return None

        if zone_type == ZoneType.DEMAND:
            below = [z for z in relevant if z.zone_high <= price * 1.001]
            if not below:
                return None
            return max(below, key=lambda z: z.zone_high)
        else:
            above = [z for z in relevant if z.zone_low >= price * 0.999]
            if not above:
                return None
            return min(above, key=lambda z: z.zone_low)

    def _detect_conflicts(self, demand_zones: List, supply_zones: List, price: float) -> List[str]:
        conflicts = []
        for dz in demand_zones:
            for sz in supply_zones:
                if dz.freshness == ZoneFreshness.INVALIDATED or sz.freshness == ZoneFreshness.INVALIDATED:
                    continue
                if dz.zone_high >= sz.zone_low * 0.995:
                    conflicts.append(
                        f"DEMAND {dz.zone_low}-{dz.zone_high} conflicts with "
                        f"SUPPLY {sz.zone_low}-{sz.zone_high} near price"
                    )
        return conflicts

    def _determine_dominant_zone(self, nearest_demand, nearest_supply, price) -> Optional[str]:
        if nearest_demand and nearest_supply:
            demand_dist = price - nearest_demand.zone_mid
            supply_dist = nearest_supply.zone_mid - price
            if nearest_demand.confidence > nearest_supply.confidence and demand_dist <= supply_dist:
                return "DEMAND"
            elif nearest_supply.confidence > nearest_demand.confidence and supply_dist <= demand_dist:
                return "SUPPLY"
            else:
                return "DEMAND" if demand_dist <= supply_dist else "SUPPLY"
        elif nearest_demand:
            return "DEMAND"
        elif nearest_supply:
            return "SUPPLY"
        return None

    def _normalize_history(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        cols = {c.lower(): c for c in df.columns}
        needed = ("open", "high", "low", "close", "volume")
        if not all(n in cols for n in needed):
            return None
        out = df[[cols[c] for c in needed]].copy()
        out.columns = ["Open", "High", "Low", "Close", "Volume"]
        if isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, utc=True).tz_convert("UTC")
        out = out.sort_index().dropna(subset=["Close"])
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)
        return out if len(out) else None

    def _find_pivot_highs(self, highs: pd.Series) -> List[int]:
        pivots = []
        window = 5
        values = highs.values
        for i in range(5, len(values) - 5):
            left = values[i-5:i]
            right = values[i+1:i+6]
            if all(values[i] > v for v in left) and all(values[i] >= v for v in right):
                pivots.append(i)
        return pivots

    def _find_pivot_lows(self, lows: pd.Series) -> List[int]:
        pivots = []
        window = 5
        values = lows.values
        for i in range(5, len(values) - 5):
            left = values[i-5:i]
            right = values[i+1:i+6]
            if all(values[i] < v for v in left) and all(values[i] <= v for v in right):
                pivots.append(i)
        return pivots

    def _pct_diff(self, a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b * 100, 2)

    def _safe_round(self, val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return None
