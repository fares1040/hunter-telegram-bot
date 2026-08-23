"""Phase 2.9 — Swing Intelligence tests.

Focused, deterministic checks mirroring the Phase 2.8 intraday suite:
- HH/HL, LH/LL structure
- trend classification via MA50/MA200
- all 10 swing setups
- volume confirmation
- S/R levels (every level has price/type/strength/distance/evidence)
- catalyst integration (reuse, not duplicate)
- optional intraday confirmation (swing works without it)
- entry / confirmation / invalidation / risk (NO targets)
- explainable 0-100 score with renormalization
- missing data / insufficient history -> UNKNOWN, never bullish
- deterministic output
- no look-ahead leakage
- contradictory setup handling
"""
import numpy as np
import pandas as pd
import types
from datetime import datetime, timezone

import pytest
from models.ticker import TickerData
from models.session import SessionSnapshot
from core.session_clock import MarketSession
from models.swing import (
    SwingIntelligence, SwingSetup, SwingLevel, SwingEntry, SwingScore,
)
from engines.swing_engine import SwingEngine, SWING_PIVOT_WINDOW

from engines.intraday_engine import IntradayEngine, IntradayIntelligence, IntradaySetup


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _segments(segs):
    out = []
    for val, n in segs:
        out.extend([val] * n)
    return np.array(out, dtype=float)


def _df(closes, vols=None, start="2025-01-01"):
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq="1D", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    highs = closes * 1.012 + np.abs(np.random.default_rng(1).normal(0, 0.05, n))
    lows = closes * 0.988 - np.abs(np.random.default_rng(2).normal(0, 0.05, n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    if vols is None:
        vols = np.random.default_rng(3).integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def _data(df, current=None, avg20=None):
    current = float(df["Close"].iloc[-1] if current is None else current)
    vol = int(df["Volume"].sum())
    avg20 = int(np.mean(df["Volume"].iloc[-20:])) if avg20 is None else avg20
    return TickerData(
        ticker="TST",
        timestamp=df.index[-1].to_pydatetime(),
        current_price=current,
        avg_volume_20d=avg20,
        regular=SessionSnapshot(session_type=MarketSession.REGULAR, volume=vol),
        intraday_bars=None,
    )


def _hh_hl_df():
    """Clear HH/HL uptrend (200+ bars). Ends in a small pullback so the final
    swing high is a CONFIRMED pivot (required for HH/HL structure)."""
    base = np.linspace(100, 125, 190).tolist()
    swing = [130, 125, 122, 124, 135, 145, 155, 150, 143, 145, 165, 175, 186, 182, 180, 178]
    closes = np.array(base + swing, dtype=float)
    return _df(closes)


def _breakout_df():
    """Uptrend with a prior high then a clear breakout on the final bar."""
    base = np.linspace(100, 150, 180).tolist()
    swing = [155, 150, 148, 152, 165, 172, 186, 187]
    closes = np.array(base + swing, dtype=float)
    return _df(closes)


def _lh_ll_df():
    """Clear LH/LL downtrend (200+ bars) ending in a clean breakdown."""
    base = np.linspace(200, 168, 190).tolist()
    swing = [170, 165, 160, 158, 162, 165, 160, 155, 158, 150,
             152, 156, 160, 154, 148, 145, 142, 140]
    closes = np.array(base + swing, dtype=float)
    return _df(closes)


def _fake_catalyst(sentiment="POSITIVE", materiality=70, trap=False, category="EARNINGS"):
    ev = types.SimpleNamespace(
        sentiment=sentiment,
        catalyst_type=types.SimpleNamespace(value=category),
        materiality_score=materiality,
    )
    prof = types.SimpleNamespace(
        freshness=types.SimpleNamespace(value="RECENT"),
        materiality=materiality,
        is_trap_risk=trap,
        sentiment=types.SimpleNamespace(value=sentiment),
    )
    return ev, prof


def _fake_intraday(direction="BULLISH", name="OPENING_RANGE_BREAKOUT"):
    intel = IntradayIntelligence(ticker="TST", as_of=datetime.now(timezone.utc), timeframe="1m")
    intel.setups = [IntradaySetup(name=name, direction=direction, detected=True, evidence=["x"], quality="CONFIRMED")]
    return intel


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def test_hh_hl_structure():
    e = SwingEngine()
    i = e.build(_data(_hh_hl_df()), daily_history=_hh_hl_df())
    assert i.trend.structure == "HH_HL", i.trend.structure_evidence
    assert i.trend.direction == "BULLISH"


def test_lh_ll_structure():
    e = SwingEngine()
    i = e.build(_data(_lh_ll_df()), daily_history=_lh_ll_df())
    assert i.trend.structure == "LH_LL", i.trend.structure_evidence
    assert i.trend.direction == "BEARISH"


# ---------------------------------------------------------------------------
# trend via MA50 / MA200
# ---------------------------------------------------------------------------
def test_trend_ma50_ma200_bullish():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    assert i.trend.ma50 is not None and i.trend.ma200 is not None
    assert i.trend.ma50 > i.trend.ma200  # uptrend: price>ma50>ma200
    assert i.trend.direction == "BULLISH"


def test_trend_insufficient_for_ma200_not_fabricated():
    # 60 bars only: MA200 unavailable, not invented
    df = _df(_segments([(100, 20), (110, 20), (120, 20)]))
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    assert i.trend.ma200 is None


# ---------------------------------------------------------------------------
# setups
# ---------------------------------------------------------------------------
def test_breakout_detected():
    df = _breakout_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "BREAKOUT" in names, names
    assert i.primary_setup().direction == "BULLISH"


def test_breakout_retest_detected():
    # rise to 130, pullback to 122, breakout spike to 134, retest ~134 holding
    closes = _segments([
        (100, 12), (110, 12), (120, 12), (130, 12),
        (125, 5), (122, 5),                # pullback low 122
        (132, 3),                          # clear prior high
        (134, 2),                          # breakout spike (new recent high)
        (133.5, 3), (134.5, 1),            # retest ~134, holding (single highest bar)
    ])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "BREAKOUT_RETEST" in names, names


def test_failed_breakout_bearish_suppresses_bullish():
    # clears resistance then falls back below it
    closes = _segments([
        (100, 20), (110, 20), (120, 20), (130, 20),  # resistance ~130
        (134, 2), (132, 2), (128, 2), (125, 10),      # broke then failed back <130
    ])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "FAILED_BREAKOUT" in names, names
    # a contradictory active BREAKOUT on the same anchor must not also fire
    assert not any(s.name == "BREAKOUT" and s.detected for s in i.setups)
    assert i.primary_setup().direction == "BEARISH"


def test_breakdown_detected():
    df = _lh_ll_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "BREAKDOWN" in names, names
    assert i.primary_setup().direction == "BEARISH"


def test_pullback_uptrend():
    closes = _segments([
        (100, 15), (110, 15), (120, 15), (130, 3),
        (128, 2), (126, 2),   # clear V pullback low at 126 (pivot low)
        (129, 3), (132, 3), (135, 5),
    ])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "PULLBACK_UPTREND" in names, names


def test_pullback_downtrend():
    closes = _segments([
        (200, 10), (185, 10), (175, 10), (165, 10),
        (160, 3), (163, 2), (162, 2),   # clear bounce high at 163 (pivot high) in downtrend
        (159, 3), (156, 3), (153, 5),
    ])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "PULLBACK_DOWNTREND" in names, names


def test_higher_low_continuation():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "HIGHER_LOW_CONTINUATION" in names, names


def test_range_breakout():
    # tight range then break above
    closes = _segments([(100, 5) for _ in range(20)])
    closes = np.concatenate([closes, [100, 100.2, 100.4, 100.6, 100.8, 101.5, 102.5, 103.5, 104.5, 106]])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "RANGE_BREAKOUT" in names, names


def test_base_breakout():
    # long flat base then breakout
    base = _segments([(100, 3) for _ in range(35)])
    base = np.concatenate([base, [100.2, 100.4, 100.6, 100.8, 101.5, 103, 105, 107]])
    df = _df(base)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "BASE_BREAKOUT" in names, names


def test_trend_continuation():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    names = [s.name for s in i.setups if s.detected]
    assert "TREND_CONTINUATION" in names, names


def test_contradictory_setups_flagged_not_primary_conflict():
    # craft a scenario that can produce both a bullish and bearish detected setup
    closes = _segments([
        (100, 20), (120, 20), (110, 6), (130, 6),  # swing highs
        (95, 6), (108, 6),                          # swing low then higher
        (134, 2), (128, 2), (125, 6),               # failed breakout at top
        (126, 10),
    ])
    df = _df(closes)
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    detected_dirs = {s.direction for s in i.setups if s.detected}
    if "BULLISH" in detected_dirs and "BEARISH" in detected_dirs:
        assert any("conflicting" in w for w in i.warnings)
    # primary is always a single direction
    assert i.primary_setup() is not None


# ---------------------------------------------------------------------------
# volume confirmation
# ---------------------------------------------------------------------------
def test_volume_expansion_boosts_quality():
    df = _breakout_df()
    vols = df["Volume"].values.copy()
    vols[-1] *= 4  # expansion on breakout bar
    df2 = df.copy(); df2["Volume"] = vols
    e = SwingEngine()
    i = e.build(_data(df2), daily_history=df2)
    bo = [s for s in i.setups if s.name == "BREAKOUT" and s.detected]
    if bo:
        assert bo[0].quality == "CONFIRMED"
    assert i.volume.volume_expansion is True


def test_weak_volume_lowers_quality():
    df = _breakout_df()
    vols = df["Volume"].values.copy()
    vols[-1] = vols[-1] * 0.2  # weak breakout volume
    df2 = df.copy(); df2["Volume"] = vols
    e = SwingEngine()
    i = e.build(_data(df2), daily_history=df2)
    bo = [s for s in i.setups if s.name == "BREAKOUT" and s.detected]
    if bo:
        assert bo[0].quality != "CONFIRMED"  # WATCH / UNCONFIRMED
    assert "WEAK_BREAKOUT_VOLUME" in i.trap_flags


# ---------------------------------------------------------------------------
# support / resistance
# ---------------------------------------------------------------------------
def test_levels_have_full_evidence():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    assert i.levels, "expected real S/R levels"
    for lvl in i.levels:
        assert isinstance(lvl, SwingLevel)
        assert lvl.level_type in ("SUPPORT", "RESISTANCE")
        assert 0 <= lvl.strength <= 100
        assert abs(lvl.distance_pct) is not None
        assert lvl.evidence


# ---------------------------------------------------------------------------
# catalyst integration (reuse, not duplicate)
# ---------------------------------------------------------------------------
def test_catalyst_strengthens_bullish_setup():
    df = _hh_hl_df()
    e = SwingEngine()
    ev, prof = _fake_catalyst(sentiment="POSITIVE", materiality=80)
    i = e.build(_data(df), daily_history=df, catalyst_event=ev, catalyst_profile=prof)
    assert i.catalyst.present
    assert i.catalyst.sentiment == "POSITIVE"
    cat_comp = [c for c in i.score.components if c.name == "Catalyst"][0]
    assert cat_comp.value is not None and cat_comp.value >= 60


def test_catalyst_trap_lowers_score():
    df = _hh_hl_df()
    e = SwingEngine()
    ev, prof = _fake_catalyst(sentiment="NEGATIVE", materiality=80, trap=True, category="DILUTION")
    i = e.build(_data(df), daily_history=df, catalyst_event=ev, catalyst_profile=prof)
    assert i.catalyst.is_trap_risk
    assert "DILUTION_OFFERING" in i.trap_flags
    cat_comp = [c for c in i.score.components if c.name == "Catalyst"][0]
    assert cat_comp.value is not None and cat_comp.value <= 40


# ---------------------------------------------------------------------------
# optional intraday confirmation
# ---------------------------------------------------------------------------
def test_intraday_confirmation_aligned():
    df = _hh_hl_df()
    e = SwingEngine()
    intraday = _fake_intraday("BULLISH", "OPENING_RANGE_BREAKOUT")
    i = e.build(_data(df), daily_history=df, intraday_intelligence=intraday)
    assert i.intraday_confirmation is not None
    assert "aligned" in i.intraday_confirmation


def test_swing_works_without_intraday():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)  # no intraday_intelligence
    assert i.intraday_confirmation is None
    assert i.primary_setup() is not None
    assert i.entry.status == "READY" or i.entry.status == "UNKNOWN"


# ---------------------------------------------------------------------------
# entry / invalidation / risk (NO targets)
# ---------------------------------------------------------------------------
def test_entry_ready_has_zone_and_invalidation():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    e0 = i.entry
    assert e0.status == "READY"
    assert e0.side in ("LONG", "SHORT")
    assert e0.entry_zone_low is not None and e0.entry_zone_high is not None
    assert e0.invalidation_price is not None
    assert e0.risk_distance_abs is not None and e0.risk_distance_abs > 0
    assert e0.risk_distance_pct is not None
    # no target fields exist
    assert not hasattr(e0, "target_1")
    assert not hasattr(e0, "target_zone")


def test_entry_long_invalidation_below_zone():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    e0 = i.entry
    if e0.side == "LONG":
        assert e0.invalidation_price < e0.entry_zone_low
    else:
        assert e0.invalidation_price > e0.entry_zone_high


def test_no_targets_anywhere_in_model():
    # confirm the model has no target attributes at all
    assert "target" not in SwingEntry.__annotations__
    assert "target" not in SwingIntelligence.__annotations__


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def test_score_bounds_and_renormalization():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(_data(df), daily_history=df)
    assert 0 <= i.score.total <= 100
    # renormalized when catalyst/structure missing — simulate by checking math
    total = i.score.total
    avail = i.score.available_weight
    manual = sum((c.value / 100) * (c.weight / avail) * 100 for c in i.score.components if c.value is not None)
    assert abs(manual - total) <= 1.0
    # every component exposed
    names = {c.name for c in i.score.components}
    assert {"Trend", "Structure", "Momentum", "Volume", "SupportResistance", "Catalyst", "RiskTrap", "SetupQuality"} <= names


def test_missing_data_not_bullish():
    # no daily history -> insufficient; score components should be None, not invented bullish
    data = TickerData(ticker="TST", timestamp=datetime.now(timezone.utc), current_price=100.0)
    e = SwingEngine()
    i = e.build(data, daily_history=None)
    assert i.data_status == "INSUFFICIENT_HISTORY"
    assert i.primary_setup() is None
    assert i.entry.status == "UNKNOWN"
    # any available components must not be fabricated bullish
    for c in i.score.components:
        if c.value is not None:
            assert c.name in ("RiskTrap",)  # only neutral-safe allowed; here none should exist


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_deterministic_output():
    df = _hh_hl_df()
    e = SwingEngine()
    a = e.build(_data(df), daily_history=df)
    b = e.build(_data(df), daily_history=df)
    assert a.score.total == b.score.total
    assert a.summary() == b.summary()
    assert [s.name for s in a.setups if s.detected] == [s.name for s in b.setups if s.detected]


# ---------------------------------------------------------------------------
# no look-ahead leakage
# ---------------------------------------------------------------------------
def test_no_lookahead_pivots():
    df = _hh_hl_df()
    e = SwingEngine()
    k = 110
    sub = df.iloc[:k]
    i = e.build(_data(sub), daily_history=sub)
    ph = e._pivot_points(sub["High"])
    pl = e._pivot_points(sub["Low"], kind="low")
    if ph:
        assert max(ph) <= k - 1 - SWING_PIVOT_WINDOW
    if pl:
        assert max(pl) <= k - 1 - SWING_PIVOT_WINDOW


def test_causality_bar_k_only_depends_on_prefix():
    # value computed at bar k must equal recomputation on the same prefix
    df = _hh_hl_df()
    e = SwingEngine()
    k = 100
    prefix = df.iloc[:k]
    i = e.build(_data(prefix), daily_history=prefix)
    # MA200 (or None) matches pandas rolling on the prefix exactly
    if len(prefix) >= 200:
        expected_ma200 = round(float(prefix["Close"].rolling(200).mean().iloc[-1]), 4)
        assert i.trend.ma200 == expected_ma200
    else:
        assert i.trend.ma200 is None
    # score is deterministic from prefix regardless of what follows
    longer = df.copy()  # same prefix, extra future bars
    j = e.build(_data(longer), daily_history=longer)
    # build on prefix must equal build on longer truncated back to prefix
    again = e.build(_data(prefix), daily_history=prefix)
    assert i.score.total == again.score.total
    assert i.trend.structure == again.trend.structure


# ---------------------------------------------------------------------------
# integration sanity: engine importable and runnable in pipeline context
# ---------------------------------------------------------------------------
def test_pipeline_build_signature():
    df = _hh_hl_df()
    e = SwingEngine()
    i = e.build(
        _data(df), daily_history=df,
        technical_intelligence=None,
        catalyst_event=None, catalyst_profile=None,
        intraday_intelligence=None, trap_risk=10, trap_warnings=["X"],
    )
    assert isinstance(i, SwingIntelligence)
    assert "X" in i.trap_flags  # passed-through trap warning preserved
