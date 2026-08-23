"""Phase 2.8 — Intraday Hunter tests.

All scenarios are built from deterministic synthetic 1m bars (fixed seeds,
no randomness at runtime) expanded onto a real ET session grid. Assertions
cover setups, entry intelligence, honesty under missing data, scoring,
and risk integration.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from engines.intraday_engine import IntradayEngine
from models.ticker import TickerData
from models.intraday import IntradaySetup, IntradayLevels
from core.session_clock import MarketSession
from models.session import SessionSnapshot


# ---------------------------------------------------------------- helpers

def _expand_to_1m(block_closes, block_highs, block_lows, block_vols):
    """Expand 5m blocks into identical 1m bars on a real 09:30 ET grid."""
    rows = []
    for j, c in enumerate(block_closes):
        for k in range(5):
            rows.append((c, block_highs[j], block_lows[j], c, block_vols[j] / 5.0))
    idx = pd.date_range("2026-08-21 09:30", periods=len(rows), freq="1min",
                        tz="America/New_York")
    df = pd.DataFrame(rows, columns=["Close", "High", "Low", "O", "V"], index=idx)
    df = df.rename(columns={"O": "Open", "V": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def make_data(bars, price=None, avg_vol=6_000_000, prev_close=None, reg_volume=None):
    data = TickerData(
        ticker="TEST",
        timestamp=bars.index[-1].to_pydatetime() if bars is not None else datetime(2026, 8, 21, 16),
        current_price=price if price is not None else (float(bars["Close"].iloc[-1]) if bars is not None else None),
        previous_close=prev_close,
        avg_volume_20d=avg_vol,
        regular=SessionSnapshot(session_type=MarketSession.REGULAR),
        intraday_bars=bars,
    )
    if reg_volume is None and bars is not None:
        reg_volume = int(float(bars["Volume"].sum()))
    data.regular.volume = reg_volume
    return data


def blocks(base_closes, hi_eps=0.05, lo_eps=0.05, override_high=None,
           override_low=None, vol=1_000.0, last_vol_mult=3.0):
    n = len(base_closes)
    highs = [(c + hi_eps) for c in base_closes]
    lows = [(c - lo_eps) for c in base_closes]
    if override_high:
        for i, v in override_high.items():
            highs[i] = v
    if override_low:
        for i, v in override_low.items():
            lows[i] = v
    vols = [vol] * n
    vols[-1] = vol * last_vol_mult
    return highs, lows, vols


@pytest.fixture
def engine():
    return IntradayEngine()


def detected(intel):
    return {s.name: s for s in intel.setups if s.detected}


# ---------------------------------------------------------------- timeframes

def test_resample_and_primary_timeframe(engine):
    idx = pd.date_range("2026-08-21 09:30", periods=120, freq="1min",
                        tz="America/New_York")
    close = np.full(120, 100.0)
    bars = pd.DataFrame({"Open": close, "High": close + 0.01, "Low": close - 0.01,
                         "Close": close,
                         "Volume": np.full(120, 100.0)}, index=idx)
    intel = engine.build(make_data(bars))
    frames = {t.timeframe: t for t in intel.timeframes}
    assert frames["1m"].bars_available == 120
    assert frames["5m"].bars_available == 24
    assert intel.timeframe == "5m"
    assert intel.data_status == "OK"

    # resample volume integrity
    b5 = engine._resample(engine._normalize_bars(bars), "5min")
    assert len(b5) == 24
    assert float(b5["Volume"].iloc[0]) == 500.0


def test_primary_falls_back_to_1m_then_15m(engine):
    def mk(n):
        idx = pd.date_range("2026-08-21 09:30", periods=n, freq="1min",
                            tz="America/New_York")
        close = np.full(n, 100.0)
        return pd.DataFrame({"Open": close, "High": close + 0.01,
                             "Low": close - 0.01, "Close": close,
                             "Volume": np.full(n, 100.0)}, index=idx)
    assert engine.build(make_data(mk(40))).timeframe == "1m"
    assert engine.build(make_data(mk(390))).timeframe == "15m"


# ---------------------------------------------------------------- setups

def test_orb_bullish_entry_ready(engine):
    closes = ([100.2] * 3 + list(np.linspace(100.6, 101.4, 20)) + [103.0])
    highs, lows, vols = blocks([float(c) for c in closes])
    bars = _expand_to_1m([float(c) for c in closes], highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "OPENING_RANGE_BREAKOUT" in d
    orb = d["OPENING_RANGE_BREAKOUT"]
    assert orb.direction == "BULLISH" and orb.anchor_basis == "OPENING_RANGE_HIGH"
    assert intel.primary_setup().name == "OPENING_RANGE_BREAKOUT"

    en = intel.entry
    assert en.status == "READY" and en.side == "LONG"
    assert en.entry_zone_low == pytest.approx(round(float(lows[0] * 0 + orb.anchor_price), 4))
    assert en.invalidation_basis == "OPENING_RANGE_LOW"
    assert en.risk_distance_pct is not None and en.risk_distance_pct > 0
    assert 0 <= en.confidence <= 100


def test_breakout_anchor_from_real_pivot(engine):
    # OR spikes to ~110 (price stays inside the range so ORB stays silent);
    # early pivot high 105.05; tall idx15 keeps the consolidation window wide
    # so neither CONSOLIDATION_BREAK nor FAILED_BREAKOUT can hijack; fresh
    # cross to 107.3 on the final bar.
    closes = [100, 100, 100,
              100.2, 100.4, 100.6, 100.8,
              105.0, 104.5, 104.0, 103.7, 103.5,
              103.4, 103.3, 103.2,
              108.0, 103.0, 102.9, 102.8, 102.7,
              102.6, 102.5, 102.4,
              107.3]
    assert len(closes) == 24
    highs, lows, vols = blocks(closes, override_high={0: 109.95, 1: 110.0, 2: 110.05})
    bars = _expand_to_1m(closes, highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "BREAKOUT" in d
    bo = d["BREAKOUT"]
    assert bo.direction == "BULLISH"
    assert bo.anchor_price == pytest.approx(105.05)
    assert bo.anchor_basis == "INTRADAY_PIVOT_HIGH"
    assert intel.primary_setup().name == "BREAKOUT"
    assert intel.entry.status == "READY"
    assert intel.entry.entry_zone_low == pytest.approx(105.05)


def test_breakout_pullback_holding_above_level(engine):
    closes = [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 105.0,
              104.5, 104.0, 103.7, 103.5, 103.4, 103.3, 103.2, 103.1,
              103.0, 102.9, 102.8, 102.7, 106.0, 104.9, 105.2, 105.3]
    highs, lows, vols = blocks(closes)
    bars = _expand_to_1m(closes, highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "BREAKOUT_PULLBACK" in d
    pb = d["BREAKOUT_PULLBACK"]
    assert pb.direction == "BULLISH"
    assert pb.anchor_price == pytest.approx(105.05)
    assert "BREAKOUT" not in d  # no fresh cross on the final bar


def test_vwap_reclaim(engine):
    # decline, recovery, then a flat hold above the OR low and below the
    # consolidation high so only VWAP_RECLAIM can be primary.
    closes = [float(x) for x in np.linspace(101, 98, 12)] \
        + [float(x) for x in np.linspace(98.2, 100.8, 7)] + [100.6] * 5
    assert len(closes) == 24
    highs, lows, vols = blocks(closes)
    bars = _expand_to_1m(closes, highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "VWAP_RECLAIM" in d
    rec = d["VWAP_RECLAIM"]
    assert rec.anchor_basis == "VWAP"
    assert intel.levels.vwap is not None
    assert intel.levels.vwap_source is not None
    en = intel.entry
    assert en.status == "READY"
    assert en.invalidation_basis in ("INTRADAY_SUPPORT", "RECENT_LOW",
                                     "OPENING_RANGE_LOW", "ATR_INTRADAY")
    assert en.invalidation_price < en.entry_zone_low


def test_vwap_rejection_short(engine):
    # price inside the opening range (no ORB), early dip inside the
    # consolidation window (no break), then a failed push back above VWAP.
    closes = [100.0, 100.0, 100.0] + [round(100.0 + i * 0.05, 2) for i in range(1, 9)] \
        + [98.0] + [100.30] * 9 + [100.30, 100.15, 100.02]
    assert len(closes) == 24
    highs, lows, vols = blocks([float(c) for c in closes])
    bars = _expand_to_1m([float(c) for c in closes], highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "OPENING_RANGE_BREAKOUT" not in d          # price inside OR: honest silence
    assert "VWAP_REJECTION" in d
    rej = d["VWAP_REJECTION"]
    assert rej.direction == "BEARISH"
    assert intel.primary_setup().name == "VWAP_REJECTION"
    en = intel.entry
    assert en.status == "READY" and en.side == "SHORT"
    assert en.invalidation_basis in ("INTRADAY_RESISTANCE", "RECENT_HIGH",
                                     "OPENING_RANGE_HIGH", "ATR_INTRADAY")
    assert en.invalidation_price > en.entry_zone_high


def test_momentum_continuation_higher_lows(engine):
    closes = list(np.linspace(100, 103, 24))
    highs, lows, vols = blocks([float(c) for c in closes])
    bars = _expand_to_1m([float(c) for c in closes], highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "MOMENTUM_CONTINUATION" in d
    mc = d["MOMENTUM_CONTINUATION"]
    assert mc.direction == "BULLISH"
    assert intel.momentum_volume.momentum_direction == "UP"


def test_volume_expansion_neutral_when_direction_unclear(engine):
    closes = [100 + (0.01 if i % 2 else -0.01) for i in range(24)]
    highs, lows, vols = blocks(closes)
    bars = _expand_to_1m(closes, highs, lows, vols)
    data = make_data(bars, avg_vol=5_000)   # forces RVOL >> 2
    intel = engine.build(data)
    ve = [s for s in intel.setups if s.name == "VOLUME_EXPANSION"]
    assert ve and ve[0].direction == "NEUTRAL" and ve[0].detected is False
    assert "informational" in " ".join(ve[0].evidence)


def test_breakdown_bearish(engine):
    closes = [99, 98.8, 98.6, 98.4, 97.0, 95.0, 95.5, 96.0, 96.2, 96.1,
              96.0, 95.9, 95.8, 95.7, 95.6, 95.5, 95.4, 95.5, 95.4,
              95.4, 95.4, 95.3, 95.2, 93.0]
    highs, lows, vols = blocks(closes)
    bars = _expand_to_1m(closes, highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "BREAKDOWN" in d
    bd = d["BREAKDOWN"]
    assert bd.direction == "BEARISH"
    assert bd.anchor_price == pytest.approx(94.95)
    assert bd.anchor_basis == "INTRADAY_PIVOT_LOW"
    assert intel.entry.status == "READY"
    assert intel.entry.side == "SHORT"


def test_failed_breakout_sets_trap_flag(engine):
    closes = [99, 99.5, 100, 100.5, 101, 101.5, 102, 105, 104.8, 104.5,
              104, 103, 102, 106.5, 104, 103, 102, 101, 100, 99, 98,
              97, 96, 95]
    highs, lows, vols = blocks(closes)
    bars = _expand_to_1m(closes, highs, lows, vols)
    intel = engine.build(make_data(bars))
    d = detected(intel)
    assert "FAILED_BREAKOUT" in d
    fb = d["FAILED_BREAKOUT"]
    assert fb.direction == "BEARISH" and fb.quality == "CONFIRMED"
    assert "INTRADAY_FAILED_BREAKOUT" in intel.trap_flags


# ---------------------------------------------------------------- honesty

def test_no_intraday_data_full_honesty(engine):
    intel = engine.build(make_data(None, price=50.0))
    assert intel.data_status == "NO_INTRADAY"
    lv = intel.levels
    assert lv.premarket_high is None and lv.opening_range_high is None \
        and lv.vwap is None and lv.recent_high is None
    assert intel.setups == [] or all(not s_.detected for s_ in intel.setups)
    assert intel.entry.status == "UNKNOWN"
    assert intel.score.total <= 49          # degraded data can never look strong
    assert all(c.value is None for c in intel.score.components)


def test_insufficient_history_flagged(engine):
    idx = pd.date_range("2026-08-21 09:30", periods=10, freq="1min",
                        tz="America/New_York")
    close = np.full(10, 100.0)
    bars = pd.DataFrame({"Open": close, "High": close + 0.01,
                         "Low": close - 0.01, "Close": close,
                         "Volume": np.full(10, 100.0)}, index=idx)
    intel = engine.build(make_data(bars))
    assert intel.data_status == "INSUFFICIENT_INTRADAY"
    assert any("only_10" in r for r in intel.data_reasons)


def test_missing_volume_component_unavailable_not_bullish(engine):
    idx = pd.date_range("2026-08-21 09:30", periods=120, freq="1min",
                        tz="America/New_York")
    close = np.full(120, 100.0)
    bars = pd.DataFrame({"Open": close, "High": close + 0.01,
                         "Low": close - 0.01, "Close": close,
                         "Volume": np.zeros(120)}, index=idx)
    intel = engine.build(make_data(bars, avg_vol=None))
    vol_comp = [c for c in intel.score.components if c.name == "Volume"][0]
    assert vol_comp.value is None
    assert vol_comp.reason and "unavailable" in vol_comp.reason
    assert vol_comp.contribution is None
    assert intel.momentum_volume.volume_spike is False
    assert intel.momentum_volume.rvol is None


def test_missing_vwap_no_setups_fabricated(engine):
    closes = list(np.linspace(101, 98, 12)) + list(np.linspace(98.2, 100.8, 12))
    highs, lows, vols = blocks([float(c) for c in closes])
    bars = _expand_to_1m([float(c) for c in closes], highs, lows, vols)
    data = make_data(bars)
    data.regular.volume = 0                    # blocks computed fallback too
    intel = engine.build(data)
    assert intel.levels.vwap is None
    assert intel.levels.vwap_source is None
    assert not [s for s in intel.setups if s.name.startswith("VWAP")]
    vw_comp = [c for c in intel.score.components if c.name == "VWAP"][0]
    assert vw_comp.value is None and vw_comp.reason == "vwap_unavailable"


def test_no_fabricated_levels_anywhere(engine):
    intel = engine.build(make_data(None, price=42.0))
    srcs = intel.levels.sources()
    assert srcs == {}
    assert intel.entry.entry_zone_low is None
    assert intel.entry.invalidation_price is None
    assert intel.entry.confidence == 0
    assert intel.entry.risk_distance_abs is None


# ---------------------------------------------------------------- entry unit paths

def test_entry_unknown_paths_direct(engine):
    lv = IntradayEngine()._build_levels(make_data(None, price=10.0), pd.DataFrame(),
                                        pd.DataFrame(), None)
    e = IntradayEngine()
    p_none = e._build_entry(None, 10.0, lv, None, None, pd.DataFrame())
    assert p_none.status == "UNKNOWN" and p_none.reason == "no_detected_setup"

    setup = IntradaySetup("BREAKOUT", "BULLISH", True)
    p_nolevel = e._build_entry(setup, 10.0, lv, None, None, pd.DataFrame())
    assert p_nolevel.status == "UNKNOWN"
    assert p_nolevel.reason == "required_level_unavailable"

    setup2 = IntradaySetup("BREAKOUT", "BULLISH", True,
                           anchor_price=10.0, anchor_basis="X")
    p_chase = e._build_entry(setup2, 9.5, lv, None, 0.05, pd.DataFrame())
    assert p_chase.status == "UNKNOWN"
    assert p_chase.reason == "price_not_in_valid_entry_position"


def test_atr_fallback_invalidation_when_no_structure(engine):
    e = IntradayEngine()
    lv = IntradayLevels()
    stop, basis = e._structural_stop_below(100.0, lv, atr_i=0.8, exclude=None)
    assert basis == "ATR_INTRADAY"
    assert stop == pytest.approx(100.0 - 0.8)


# ---------------------------------------------------------------- scoring

def _all_scenarios():
    out = []
    c1 = [float(x) for x in ([100.2] * 3 + list(np.linspace(100.6, 101.4, 20)) + [103.0])]
    h, l, v = blocks(c1)
    out.append(_expand_to_1m(c1, h, l, v))
    c2 = [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 105.0, 104.5,
          104.0, 103.7, 103.5, 103.4, 103.3, 103.2, 103.1, 103.0, 102.9,
          102.8, 102.7, 102.6, 102.5, 102.4, 107.0]
    h, l, v = blocks(c2)
    out.append(_expand_to_1m(c2, h, l, v))
    c3 = list(np.linspace(101, 98, 12)) + list(np.linspace(98.2, 100.8, 12))
    h, l, v = blocks([float(x) for x in c3])
    out.append(_expand_to_1m([float(x) for x in c3], h, l, v))
    return out


def test_score_bounds_across_scenarios(engine):
    for bars in _all_scenarios():
        intel = engine.build(make_data(bars))
        assert 0 <= intel.score.total <= 100
        for c in intel.score.components:
            assert c.value is None or 0 <= c.value <= 100


def test_degraded_data_caps_score_even_with_strong_technical(engine):
    class FakeTI:
        timeframe = "1d"

        class score:
            total = 90

    class FakeTech:
        intelligence = FakeTI()

    intel = engine.build(make_data(None, price=50.0), technical=FakeTech(),
                         trap_risk=10, trap_warnings=["EXTREME_GAP"])
    tech_comp = [c for c in intel.score.components if c.name == "TechnicalScore"][0]
    assert tech_comp.value == 90                       # real input honored
    assert intel.score.total <= 49                     # but never looks strong
    assert intel.score.renormalized is True


def test_trap_and_catalyst_components(engine):
    bars = _all_scenarios()[0]

    class Reaction:
        reaction_label = "NEGATIVE_REACTION"

    intel = engine.build(make_data(bars), reaction=Reaction(),
                         trap_risk=70, trap_warnings=["EXTREME_GAP"])
    comps = {c.name: c for c in intel.score.components}
    assert comps["Catalyst"].value == 15
    assert comps["RiskTrap"].value == 30
    assert comps["RiskTrap"].contribution > 0

    class GoodReaction:
        reaction_label = "POSITIVE_REACTION"

    intel2 = engine.build(make_data(bars), reaction=GoodReaction(), trap_risk=0)
    comps2 = {c.name: c for c in intel2.score.components}
    assert comps2["Catalyst"].value == 80
    assert comps2["RiskTrap"].value == 100


def test_renormalization_contributions_sum(engine):
    bars = _all_scenarios()[0]
    intel = engine.build(make_data(bars))       # no catalyst/technical/trap inputs
    sc = intel.score
    assert sc.available_weight < 100.0
    assert sc.renormalized is True
    contrib_sum = sum(c.contribution for c in sc.components if c.contribution)
    assert contrib_sum == pytest.approx(sc.total, abs=0.5)


# ---------------------------------------------------------------- determinism

def test_deterministic_outputs(engine):
    bars = _all_scenarios()[0]
    i1 = engine.build(make_data(bars))
    i2 = engine.build(make_data(bars))
    assert i1.score.total == i2.score.total
    assert [(c.name, c.value, c.contribution) for c in i1.score.components] == \
           [(c.name, c.value, c.contribution) for c in i2.score.components]
    assert [(s.name, s.detected, s.quality, s.anchor_price) for s in i1.setups] == \
           [(s.name, s.detected, s.quality, s.anchor_price) for s in i2.setups]
    a, b = i1.entry, i2.entry
    assert (a.status, a.setup, a.side, a.entry_zone_low, a.entry_zone_high,
            a.invalidation_price, a.invalidation_basis, a.risk_distance_pct,
            a.confidence) == \
           (b.status, b.setup, b.side, b.entry_zone_low, b.entry_zone_high,
            b.invalidation_price, b.invalidation_basis, b.risk_distance_pct,
            b.confidence)


def test_confirmations_exposed_and_evaluable(engine):
    bars = _all_scenarios()[0]
    intel = engine.build(make_data(bars))
    en = intel.entry
    if en.status == "READY":
        assert len(en.confirmations) >= 3
        for chk in en.confirmations:
            assert chk.met in (True, False, None)
        met = sum(1 for c in en.confirmations if c.met)
        evaluable = sum(1 for c in en.confirmations if c.met is not None)
        expected = (55 if detected(intel)[en.setup].quality == "CONFIRMED" else 40)
        if evaluable:
            expected += int(15 * (met / evaluable))
        assert en.confidence == expected
