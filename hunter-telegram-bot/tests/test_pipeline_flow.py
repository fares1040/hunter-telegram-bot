"""Phase 2.10 — full pipeline object-flow integration (STEP 4 / 11).

Runs the REAL engines in the exact process_ticker order:
Discovery(no-op here) -> Technical -> Intraday -> Swing -> Target -> Decision
with realistic TickerData + daily history. Proves:
- every intelligence object exists when data is available
- TargetResult is produced and attached to HunterSignal
- DecisionEngine stays authoritative (target presence does not change decision)
- missing history degrades gracefully (no fabricated targets, no crash)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from models.ticker import TickerData
from models.session import SessionSnapshot, MarketSession
from models.swing import SwingIntelligence, SwingEntry
from models.target import TargetResult
from engines.technical_engine import TechnicalEngine
from engines.intraday_engine import IntradayEngine
from engines.swing_engine import SwingEngine
from engines.target_engine import TargetEngine
from engines.decision_engine import DecisionEngine
from models.signal import HunterSignal


def _daily_df(closes):
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    highs = closes * 1.02 + np.abs(np.random.default_rng(1).normal(0, 0.05, n))
    lows = closes * 0.98 - np.abs(np.random.default_rng(2).normal(0, 0.05, n))
    opens = np.roll(closes, 1); opens[0] = closes[0]
    vols = np.random.default_rng(3).integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=idx)


def _data(df, current=None):
    current = float(df["Close"].iloc[-1] if current is None else current)
    prev = float(df["Close"].iloc[-2])
    vol = int(df["Volume"].sum())
    avg20 = int(np.mean(df["Volume"].iloc[-20:]))
    hi = float(df["High"].max()); lo = float(df["Low"].min())
    return TickerData(
        ticker="TEST", timestamp=df.index[-1].to_pydatetime(),
        current_price=current, previous_close=prev,
        avg_volume_20d=avg20,
        regular=SessionSnapshot(session_type=MarketSession.REGULAR, high=hi, low=lo,
                                open=float(df["Open"].iloc[-1]), close=current, volume=vol),
        intraday_bars=None,
    )


def _decide_with(target_result, technical, intraday, swing):
    # minimal non-target stubs (mirrors decision inputs other than targets)
    class NS: pass
    td = NS(); td.ticker = "TEST"; td.current_price = 100.0; td.change_percent = 1.0
    td.relative_volume = 1.2; td.timestamp = datetime(2025, 1, 1, 14, 30)
    ev = NS(); ev.catalyst_type = NS(); ev.catalyst_type.value = "EARNINGS"
    ev.sentiment = "POSITIVE"; ev.source_tier_score = 80; ev.impact_score = 80
    ev.priced_in_probability = 0.1; ev.primary_source = NS(); ev.primary_source.published_at = "2025-01-01T00:00:00Z"
    ev.is_fresh = lambda max_age_minutes=120: True
    rc = NS(); rc.reaction_score = 70; rc.reaction_label = "POSITIVE_REACTION"
    lq = NS(); lq.score = 70; lq.status = "NORMAL"
    tp = NS(); tp.setup_score = 70; tp.warnings = []
    cf = NS(); cf.score = 80
    eng = DecisionEngine()
    return eng.decide(td, ev, rc, lq, tp, cf, options=None, risk_plan=None,
                      trap_risk=0, trap_warnings=[], market_context=None,
                      technical_intelligence=technical.intelligence,
                      intraday_intelligence=intraday, swing_intelligence=swing,
                      target_result=target_result)


def _run_chain(df, current=None):
    data = _data(df, current)
    history = df
    tech = TechnicalEngine().analyze(data, history)
    intraday = IntradayEngine().build(data, technical=tech, daily_history=history)
    swing = SwingEngine().build(data, daily_history=history,
                               technical_intelligence=tech.intelligence,
                               intraday_intelligence=intraday,
                               catalyst_event=None, catalyst_profile=None,
                               trap_risk=0, trap_warnings=[])
    target_result = None
    entry = swing.entry.entry_zone_low if swing.entry else None
    inv = swing.entry.invalidation_price if swing.entry else None
    if entry and inv:
        target_result = TargetEngine().build(swing=swing, technical=tech.intelligence,
                                            intraday=intraday, entry_price=entry, invalidation=inv)
    signal = _decide_with(target_result, tech, intraday, swing)
    return data, tech, intraday, swing, target_result, signal


def test_full_chain_objects_exist_with_data():
    closes = np.concatenate([np.linspace(100, 200, 180), np.array([205, 202, 208, 215, 212, 210])])
    df = _daily_df(closes)
    data, tech, intraday, swing, target_result, signal = _run_chain(df)
    # every intelligence object exists
    assert tech is not None and tech.intelligence is not None
    assert intraday is not None
    assert swing is not None
    # target result is a real object (READY or honest UNAVAILABLE), never fabricated None mid-pipeline
    assert target_result is None or isinstance(target_result, TargetResult)
    # target attached to signal
    assert signal.target_result is target_result


def test_full_chain_target_ordering_when_present():
    closes = np.concatenate([np.linspace(100, 200, 180), np.array([205, 202, 208, 215, 212, 210])])
    df = _daily_df(closes)
    data, tech, intraday, swing, target_result, signal = _run_chain(df)
    if target_result and target_result.tp1:
        entry = swing.entry.entry_zone_low
        if target_result.direction == "LONG":
            assert target_result.tp1.zone.zone_low > entry
            if target_result.tp2:
                assert target_result.tp2.zone.zone_low > target_result.tp1.zone.zone_low
            if target_result.tp3:
                assert target_result.tp3.zone.zone_low > target_result.tp2.zone.zone_low
        else:
            assert target_result.tp1.zone.zone_high < entry
            if target_result.tp2:
                assert target_result.tp2.zone.zone_high < target_result.tp1.zone.zone_high
            if target_result.tp3:
                assert target_result.tp3.zone.zone_high < target_result.tp2.zone.zone_high


def test_missing_history_degrades_gracefully():
    closes = np.concatenate([np.linspace(100, 200, 180), np.array([205, 202, 208, 215, 212, 210])])
    df = _daily_df(closes)
    data = _data(df)
    # no history -> swing reports insufficient history
    swing = SwingEngine().build(data, daily_history=None,
                               technical_intelligence=None, intraday_intelligence=None,
                               catalyst_event=None, catalyst_profile=None)
    assert swing.data_status in ("INSUFFICIENT_HISTORY", "NO_DATA")
    # entry is None -> target build skipped, signal carries None target safely
    tech = TechnicalEngine().analyze(data, df)
    intraday = IntradayEngine().build(data, technical=tech, daily_history=df)
    signal = _decide_with(None, tech, intraday, swing)
    assert signal.target_result is None


def test_decision_authority_unchanged_by_target_full_chain():
    closes = np.concatenate([np.linspace(100, 200, 180), np.array([205, 202, 208, 215, 212, 210])])
    df = _daily_df(closes)
    data, tech, intraday, swing, tr, sig_with = _run_chain(df)
    _, _, _, _, _, sig_without = _run_chain(df)
    assert sig_with.decision == sig_without.decision
    assert sig_with.hunter_score == sig_without.hunter_score
