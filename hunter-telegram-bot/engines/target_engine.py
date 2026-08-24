"""Hunter Bot - Target Intelligence Engine (Phase 2.10).

Deterministic, explainable target zone generation from real market structure.
Additive - consumes SwingIntelligence, TechnicalIntelligence, and
IntradayIntelligence outputs without replacing them. No targets from arbitrary
percentages.

Targets are zone-based (zone_low / zone_high), never single magical prices.
Quality considers structural evidence, distance, R:R, level strength, and
timeframe alignment. Confidence is separate from Score. Targets are gated by
valid entry/invalidation.

Causality: every computation depends only on bars at or before the last bar.
Appending future bars can never change the output at an earlier bar.
"""
from typing import Optional, List
from models.ticker import TickerData
from models.target import (
    TargetZone,
    Target,
    TargetScore,
    TargetScoreComponent,
    TargetConfidence,
    TargetResult,
)
from models.swing import SwingIntelligence
from models.technical import TechnicalIntelligence
from models.intraday import IntradayIntelligence
from utils.logger import LOGGER


# Nominal weights; renormalized over available components at scoring time.
TARGET_SCORE_WEIGHTS = {
    "Structure": 25,
    "DistanceQuality": 15,
    "RiskReward": 20,
    "LevelStrength": 15,
    "TimeframeAlignment": 10,
    "VolumeConfirmation": 5,
    "CatalystContext": 5,
    "SetupQuality": 5,
}


class TargetEngine:
    """Deterministic target zone generation from real market structure.

    Consumes SwingIntelligence, TechnicalIntelligence, IntradayIntelligence.
    Does NOT replace Swing/Intraday/Technical/Risk/Trap/Decision engines.
    """

    def build(self, swing=None, technical=None, intraday=None, data=None,
              entry_price=None, invalidation=None):
        result = TargetResult()
        result.missing_data = []

        if entry_price is None or invalidation is None:
            result.status = "UNAVAILABLE"
            result.direction = "LONG" if entry_price and invalidation is None else "LONG"
            result.evidence.append("no valid entry or invalidation price")
            return result

        result.entry = entry_price
        result.direction = "LONG" if entry_price > invalidation else "SHORT"

        candidates = []
        if swing is not None:
            candidates.extend(self._from_swing(swing, entry_price))
        if technical is not None:
            candidates.extend(self._from_technical(technical, entry_price))
        if intraday is not None:
            candidates.extend(self._from_intraday(intraday, entry_price))

        candidates = self._cluster_zones(candidates)
        tp1, tp2, tp3 = self._rank_targets(candidates, result.direction)
        result.tp1, result.tp2, result.tp3 = tp1, tp2, tp3

        if result.direction == "LONG" and tp1 is not None and entry_price > invalidation:
            result.invalidation = invalidation
            if (tp1.zone.zone_low - entry_price) > 0:
                result.risk_reward = (tp1.zone.zone_low - entry_price) / (entry_price - invalidation)
        elif result.direction == "SHORT" and tp1 is not None and invalidation > entry_price:
            result.invalidation = invalidation
            if (entry_price - tp1.zone.zone_high) > 0:
                result.risk_reward = (entry_price - tp1.zone.zone_high) / (invalidation - entry_price)

        result.score = self._score_targets(result)
        result.confidence = self._target_confidence(result)

        if tp1 is not None and result.risk_reward is not None and result.risk_reward > 0:
            result.status = "READY"
        elif tp1 is not None:
            result.status = "WATCH"
        else:
            result.status = "UNAVAILABLE"
        return result

    def _from_swing(self, swing, entry):
        out = []
        for lv in swing.levels:
            if lv.level_type == "RESISTANCE" and lv.price > entry:
                zw = max(abs(lv.price) * 0.004, 0.1)
                zl = round(lv.price - zw / 2, 2)
                zh = round(lv.price + zw / 2, 2)
                if zl > entry:
                    out.append(self._mk("TP1", zl, zh, "SWING_RESISTANCE", lv, entry, "LONG"))
            elif lv.level_type == "SUPPORT" and lv.price < entry:
                zw = max(abs(lv.price) * 0.004, 0.1)
                zl = round(lv.price - zw / 2, 2)
                zh = round(lv.price + zw / 2, 2)
                if zh < entry:
                    out.append(self._mk("TP1", zl, zh, "SWING_SUPPORT", lv, entry, "SHORT"))
        return out

    def _from_technical(self, technical, entry):
        out = []
        sr = technical.support_resistance
        trend = technical.trend
        atr = technical.volatility.atr if technical.volatility else None
        if sr and sr.nearest_resistance and sr.nearest_resistance.distance_pct and sr.nearest_resistance.distance_pct > 0:
            res = sr.nearest_resistance
            zw = max(atr * 1.5 if atr else entry * 0.005, 0.1)
            zl = round(res.price - zw / 2, 2)
            zh = round(res.price + zw / 2, 2)
            if zl > entry:
                out.append(self._mk("TP1", zl, zh, "MAJOR_RESISTANCE", res, entry, "LONG"))
        if sr and sr.nearest_support and sr.nearest_support.distance_pct and sr.nearest_support.distance_pct < 0:
            sup = sr.nearest_support
            zw = max(atr * 1.5 if atr else entry * 0.005, 0.1)
            zl = round(sup.price - zw / 2, 2)
            zh = round(sup.price + zw / 2, 2)
            if zh < entry:
                out.append(self._mk("TP1", zl, zh, "MAJOR_SUPPORT", sup, entry, "SHORT"))
        if atr and atr > 0 and trend:
            proj = atr * 2.0
            if trend.direction.value == "BULLISH":
                zl = round(entry + proj * 0.8, 2)
                zh = round(entry + proj * 1.2, 2)
                if zl > entry:
                    out.append(Target(tp_id="TP2", zone=TargetZone(
                        zone_low=zl, zone_high=zh, source="ATR_PROJECTION",
                        source_type="ATR_PROJECTION", distance=round(zl - entry, 2),
                        distance_pct=round((zl - entry) / entry * 100, 2),
                        confidence="WEAK", quality="WEAK",
                        evidence="ATR projection 2xATR above entry",
                        timeframe="1d", direction="LONG"), status="READY"))
            elif trend.direction.value == "BEARISH":
                zl = round(entry - proj * 1.2, 2)
                zh = round(entry - proj * 0.8, 2)
                if zh < entry:
                    out.append(Target(tp_id="TP2", zone=TargetZone(
                        zone_low=zl, zone_high=zh, source="ATR_PROJECTION",
                        source_type="ATR_PROJECTION", distance=round(entry - zh, 2),
                        distance_pct=round((entry - zh) / entry * 100, 2),
                        confidence="WEAK", quality="WEAK",
                        evidence="ATR projection 2xATR below entry",
                        timeframe="1d", direction="SHORT"), status="READY"))
        return out

    def _from_intraday(self, intraday, entry):
        out = []
        lv = intraday.levels
        if lv.vwap is not None:
            zw = max(entry * 0.005, 0.5)
            zl = round(lv.vwap - zw / 2, 2)
            zh = round(lv.vwap + zw / 2, 2)
            if entry < lv.vwap:
                out.append(Target(tp_id="TP1", zone=TargetZone(
                    zone_low=zl, zone_high=zh, source="INTRADAY_LEVEL",
                    source_type="VWAP", distance=round(zl - entry, 2),
                    distance_pct=round((zl - entry) / entry * 100, 2),
                    confidence="WEAK", quality="WEAK",
                    evidence="Intraday VWAP resistance",
                    timeframe="1m", direction="LONG"), status="READY"))
            elif entry > lv.vwap:
                out.append(Target(tp_id="TP1", zone=TargetZone(
                    zone_low=zl, zone_high=zh, source="INTRADAY_LEVEL",
                    source_type="VWAP", distance=round(entry - zh, 2),
                    distance_pct=round((entry - zh) / entry * 100, 2),
                    confidence="WEAK", quality="WEAK",
                    evidence="Intraday VWAP support",
                    timeframe="1m", direction="SHORT"), status="READY"))
        if lv.opening_range_high is not None and entry > lv.opening_range_high:
            zw = max(entry * 0.003, 0.3)
            zl = round(lv.opening_range_high - zw / 2, 2)
            zh = round(lv.opening_range_high + zw / 2, 2)
            out.append(Target(tp_id="TP1", zone=TargetZone(
                zone_low=zl, zone_high=zh, source="RANGE_HIGH",
                source_type="OPENING_RANGE_BREAKOUT", distance=round(zl - entry, 2),
                distance_pct=round((zl - entry) / entry * 100, 2),
                confidence="VALID", quality="VALID",
                evidence="Opening range high breakout",
                timeframe="1m", direction="LONG"), status="READY"))
        if lv.opening_range_low is not None and entry < lv.opening_range_low:
            zw = max(entry * 0.003, 0.3)
            zl = round(lv.opening_range_low - zw / 2, 2)
            zh = round(lv.opening_range_low + zw / 2, 2)
            out.append(Target(tp_id="TP1", zone=TargetZone(
                zone_low=zl, zone_high=zh, source="RANGE_LOW",
                source_type="OPENING_RANGE_BREAKDOWN", distance=round(entry - zh, 2),
                distance_pct=round((entry - zh) / entry * 100, 2),
                confidence="VALID", quality="VALID",
                evidence="Opening range low breakdown",
                timeframe="1m", direction="SHORT"), status="READY"))
        return out

    def _mk(self, tpid, zl, zh, src, lv, entry, direction):
        return Target(tp_id=tpid, zone=TargetZone(
            zone_low=zl, zone_high=zh, source=src, source_type=src,
            distance=round(zl - entry, 2) if direction == "LONG" else round(entry - zh, 2),
            distance_pct=round((zl - entry) / entry * 100, 2) if direction == "LONG" else round((entry - zh) / entry * 100, 2),
            confidence="VALID", quality="VALID",
            evidence="Swing level %s strength %s" % (lv.price, lv.strength),
            timeframe="1d", direction=direction), status="READY")

    def _cluster_zones(self, candidates):
        if not candidates:
            return candidates
        candidates.sort(key=lambda t: abs(t.zone.distance))
        clustered = []
        used = set()
        for i, c1 in enumerate(candidates):
            if i in used:
                continue
            cluster = [c1]
            used.add(i)
            w1 = c1.zone.zone_high - c1.zone.zone_low
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                c2 = candidates[j]
                if c2.zone.direction != c1.zone.direction:
                    continue
                w2 = c2.zone.zone_high - c2.zone.zone_low
                if abs(c1.zone.distance - c2.zone.distance) <= (w1 + w2) / 2:
                    cluster.append(c2)
                    used.add(j)
            if cluster:
                all_low = min(c.zone.zone_low for c in cluster)
                all_high = max(c.zone.zone_high for c in cluster)
                best = min(cluster, key=lambda c: abs(c.zone.distance_pct) if c.zone.distance_pct else 999)
                clustered.append(Target(tp_id=best.tp_id, zone=TargetZone(
                    zone_low=all_low, zone_high=all_high, source=best.zone.source,
                    source_type=best.zone.source_type, distance=best.zone.distance,
                    distance_pct=best.zone.distance_pct, confidence=best.zone.confidence,
                    quality=best.zone.quality,
                    evidence="; ".join(c.zone.evidence for c in cluster),
                    timeframe=best.zone.timeframe, direction=best.zone.direction), status="READY"))
        return clustered

    def _rank_targets(self, candidates, direction):
        valid = [c for c in candidates if c.zone.direction == direction]
        if not valid:
            return None, None, None
        conf_priority = {"VALID": 3, "WEAK": 2, "UNAVAILABLE": 1}

        def key(t):
            d = t.zone.distance_pct or 999
            return (-conf_priority.get(t.zone.confidence, 0), abs(d), -len(t.zone.evidence))

        valid.sort(key=key)
        tp1 = valid[0] if len(valid) >= 1 else None
        tp2 = valid[1] if len(valid) >= 2 else None
        tp3 = valid[2] if len(valid) >= 3 else None
        return tp1, tp2, tp3

    def _score_targets(self, result):
        comps = []
        if result.tp1 and result.tp1.zone.confidence in ("VALID", "EXCELLENT"):
            comps.append(TargetScoreComponent("Structure", 25, 80, result.tp1.zone.source_type))
        else:
            comps.append(TargetScoreComponent("Structure", 25, None, "no valid target structure"))
        d = result.tp1.zone.distance_pct if result.tp1 else None
        if d:
            if 3 <= abs(d) <= 8:
                comps.append(TargetScoreComponent("DistanceQuality", 15, 80, "%.1f%% from entry" % abs(d)))
            elif abs(d) < 3:
                comps.append(TargetScoreComponent("DistanceQuality", 15, 50, "%.1f%% too tight" % abs(d)))
            else:
                comps.append(TargetScoreComponent("DistanceQuality", 15, 60, "%.1f%% from entry" % abs(d)))
        else:
            comps.append(TargetScoreComponent("DistanceQuality", 15, None, "no distance"))
        if result.risk_reward and result.risk_reward > 0:
            rr = result.risk_reward
            v = 90 if rr >= 2.0 else (70 if rr >= 1.5 else 40)
            comps.append(TargetScoreComponent("RiskReward", 20, v, "R:R %.1f" % rr))
        else:
            comps.append(TargetScoreComponent("RiskReward", 20, None, "no R:R"))
        if result.tp1 and result.tp1.zone.quality in ("VALID", "EXCELLENT"):
            comps.append(TargetScoreComponent("LevelStrength", 15, 70, result.tp1.zone.quality))
        else:
            comps.append(TargetScoreComponent("LevelStrength", 15, None, "no level strength"))
        comps.append(TargetScoreComponent("TimeframeAlignment", 10, 50, "1d"))
        comps.append(TargetScoreComponent("VolumeConfirmation", 5, 40, "none"))
        comps.append(TargetScoreComponent("CatalystContext", 5, None, "none"))
        comps.append(TargetScoreComponent("SetupQuality", 5, None, "none"))
        avail = [c for c in comps if c.value is not None]
        tw = sum(c.weight for c in avail)
        total = int(round(sum((c.value or 0) * c.weight for c in avail) / tw)) if tw else 0
        return TargetScore(total=max(0, min(100, total)), components=comps)

    def _target_confidence(self, result):
        comps = []
        if result.tp1:
            v = 80 if result.tp1.zone.confidence == "VALID" else 30
            comps.append(TargetScoreComponent("Evidence", 25, v, "structural"))
            rr = result.risk_reward or 0
            comps.append(TargetScoreComponent("RiskReward", 20, max(0, min(80, int(rr * 20))), "R:R"))
            comps.append(TargetScoreComponent("Timeframe", 15, 60, result.tp1.zone.timeframe))
            comps.append(TargetScoreComponent("DataCompleteness", 15, 100 - min(30, len(result.missing_data) * 10), "missing"))
        else:
            comps.append(TargetScoreComponent("DataCompleteness", 50, 0, "no target"))
        avail = [c for c in comps if c.value is not None]
        tw = sum(c.weight for c in avail)
        total = int(round(sum((c.value or 0) * c.weight for c in avail) / tw)) if tw else 0
        return TargetConfidence(value=max(0, min(100, total)), components=comps)
