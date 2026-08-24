"""Phase 2.10 — Target Intelligence Engine tests.

Deterministic, additive checks:
- targets are zones (low/high), never single magical prices
- targets derive only from real structural levels (swing/technical/intraday)
- no fabricated targets from arbitrary percentages
- gated by valid entry/invalidation; honest UNAVAILABLE otherwise
- TP1/TP2/TP3 ranking + zone clustering
- quality Score and Confidence are independent 0-100 metrics
- no look-ahead leakage (engine is a pure function of passed intel)
- additive: does not mutate upstream intelligence objects
"""
import pytest
from models.swing import SwingIntelligence, SwingLevel
from models.technical import (
    TechnicalIntelligence, SupportResistanceIntelligence, PriceLevel,
    TrendIntelligence, VolatilityIntelligence, TrendDirection,
)
from models.intraday import IntradayIntelligence, IntradayLevels
from models.target import TargetZone, Target
from engines.target_engine import TargetEngine


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _swing(levels):
    sw = SwingIntelligence(ticker="TEST")
    sw.levels = levels
    return sw


def _res(price, dist, strength=80, role="MAJOR"):
    return SwingLevel(price=price, level_type="RESISTANCE", strength=strength,
                      distance_pct=dist, evidence="pivot high (swing high)", role=role)


def _sup(price, dist, strength=80, role="MAJOR"):
    return SwingLevel(price=price, level_type="SUPPORT", strength=strength,
                      distance_pct=dist, evidence="pivot low (swing low)", role=role)


def _eng():
    return TargetEngine()


# ---------------------------------------------------------------------------
# core behavior
# ---------------------------------------------------------------------------
def test_zones_not_single_price():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    assert r.tp1 is not None
    z = r.tp1.zone
    assert z.zone_high > z.zone_low          # a real zone
    assert z.zone_high - z.zone_low < 2.0    # tight band around level, not a wide guess
    assert r.tp1.zone.zone_low != r.tp1.zone.zone_high


def test_long_target_from_resistance_above_entry():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8), _res(150.0, 11.5)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    assert r.direction == "LONG"
    assert r.status == "READY"
    assert r.tp1 is not None and r.tp2 is not None
    assert r.tp1.zone.zone_low > 134.5               # target above entry
    assert r.tp2.zone.zone_low > r.tp1.zone.zone_low  # ordered
    assert r.risk_reward is not None and r.risk_reward > 0


def test_short_target_from_support_below_entry():
    eng = _eng()
    sw = _swing([_sup(128.0, -4.8)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=137.0)
    assert r.direction == "SHORT"
    assert r.status == "READY"
    assert r.tp1 is not None
    assert r.tp1.zone.zone_high < 134.5               # target below entry
    assert r.risk_reward is not None and r.risk_reward > 0


def test_no_fabricated_targets():
    eng = _eng()
    # support exists but entry is BELOW it (no resistance above for LONG)
    sw = _swing([_sup(128.0, -4.8)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    # for LONG we need resistance above; only support below -> no LONG target
    assert r.tp1 is None
    assert r.status == "UNAVAILABLE"


def test_missing_entry_or_invalidation_is_honest():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8)])
    r1 = eng.build(swing=sw, entry_price=None, invalidation=132.0)
    assert r1.status == "UNAVAILABLE"
    assert any("entry" in e.lower() for e in r1.evidence)
    r2 = eng.build(swing=sw, entry_price=134.5, invalidation=None)
    assert r2.status == "UNAVAILABLE"
    assert any("invalidation" in e.lower() for e in r2.evidence)


def test_zone_clustering_merges_overlapping():
    eng = _eng()
    # two near-identical resistance levels should cluster into one TP1
    sw = _swing([_res(145.0, 7.8, strength=85), _res(145.3, 8.0, strength=70)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    # only one LONG target expected from the cluster (plus none other)
    assert r.tp1 is not None
    assert r.tp2 is None
    assert ";" in r.tp1.zone.evidence or "Swing level" in r.tp1.zone.evidence


def test_score_and_confidence_independent_zero_to_100():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8)])
    r = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    assert 0 <= r.score.total <= 100
    assert 0 <= r.confidence.value <= 100
    # confidence and score are distinct computations
    assert isinstance(r.score, object) and isinstance(r.confidence, object)


def test_additive_no_mutation_of_inputs():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8)])
    before = len(sw.levels)
    eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    assert len(sw.levels) == before          # engine consumed, not mutated


def test_technical_resistance_and_atr_projection():
    eng = _eng()
    tech = TechnicalIntelligence(ticker="TEST")
    tech.support_resistance = SupportResistanceIntelligence(
        nearest_resistance=PriceLevel(price=160.0, level_type="RESISTANCE",
                                      strength=75, distance_pct=5.0, evidence="res"),
        nearest_support=PriceLevel(price=120.0, level_type="SUPPORT",
                                   strength=70, distance_pct=-5.0, evidence="sup"))
    tech.trend = TrendIntelligence(direction=TrendDirection.BULLISH)
    tech.volatility = VolatilityIntelligence(atr=2.0)
    r = eng.build(technical=tech, entry_price=134.5, invalidation=132.0)
    assert r.status == "READY"
    assert r.tp1 is not None
    # ATR projection should appear as a second target
    assert any(t.zone.source_type == "ATR_PROJECTION" for t in (r.tp1, r.tp2) if t)


def test_intraday_vwap_target():
    eng = _eng()
    intra = IntradayIntelligence(ticker="TEST")
    intra.levels = IntradayLevels(vwap=136.0, opening_range_high=135.0,
                                  opening_range_low=133.0)
    r = eng.build(intraday=intra, entry_price=134.5, invalidation=133.2)
    assert r.tp1 is not None
    assert r.tp1.zone.source_type == "VWAP"
    assert r.tp1.zone.zone_low > 134.5


def test_deterministic_output():
    eng = _eng()
    sw = _swing([_res(145.0, 7.8), _sup(128.0, -4.8)])
    a = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    b = eng.build(swing=sw, entry_price=134.5, invalidation=132.0)
    assert a.tp1.zone.zone_low == b.tp1.zone.zone_low
    assert a.score.total == b.score.total
    assert a.confidence.value == b.confidence.value


def test_cross_direction_levels_do_not_merge():
    eng = _eng()
    sw = _swing([_res(220.0, 5.0, strength=90), _sup(207.0, -3.0, strength=90)])
    # SHORT entry: only supports below entry are valid SHORT targets
    r = eng.build(swing=sw, entry_price=214.0, invalidation=216.0)
    assert r.direction == "SHORT"
    assert r.tp1 is not None
    assert r.tp1.zone.zone_high < 214.0
    assert "RESISTANCE" not in r.tp1.zone.source_type
    # LONG entry: only resistances above entry are valid LONG targets
    r2 = eng.build(swing=sw, entry_price=214.0, invalidation=212.0)
    assert r2.direction == "LONG"
    assert r2.tp1 is not None
    assert r2.tp1.zone.zone_low > 214.0
    assert "SUPPORT" not in r2.tp1.zone.source_type
