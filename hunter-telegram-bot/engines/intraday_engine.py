"""Hunter Bot — Intraday Intelligence Engine (Phase 2.8).

Deterministic intraday layer built on REAL market data:
- timeframes: 1m (provider bars) and 5m/15m derived by resampling the same
  real 1m bars. No synthetic data is ever generated.
- levels, setups, momentum/volume measurements, entry intelligence
  (entry zone / confirmation / invalidation / risk distance), and an
  explainable 0-100 score that renormalizes over available components.

The engine NEVER sends messages. The decision/alert layer stays authoritative.
Targets are intentionally out of scope (Phase 2.10).
"""
from __future__ import annotations

import logging
from typing import Optional, List, Tuple

import pandas as pd

from models.ticker import TickerData
from models.intraday import (
    ConfirmationCheck,
    EntryPlan,
    IntradayIntelligence,
    IntradayLevels,
    IntradayMomentumVolume,
    IntradayScore,
    IntradayScoreComponent,
    IntradaySetup,
    TimeframeAnalysis,
)

LOGGER = logging.getLogger("hunter")

REGULAR_START_MINUTES = 570          # 09:30 ET in minutes-from-midnight
ORB_WINDOW_MINUTES = 15              # standard opening range window
PIVOT_WINDOW = 3                     # bars each side for intraday swing pivots
RECENT_WINDOW = 60                   # bars for recent_high/low on the analysis TF
VOL_SPIKE_MULTIPLE = 2.0             # last-bar vol vs rolling mean
VOL_CONFIRM_RATIO = 1.5              # breakout volume vs prior baseline
MOMENTUM_LOOKBACK = 6                # bars for acceleration measurement


class IntradayEngine:
    """Real-data intraday analysis. Deterministic: identical input -> identical output."""

    # ---------------- public API ----------------

    def build(
        self,
        data: TickerData,
        technical=None,                     # TechnicalProfile (with .intelligence)
        daily_history: Optional[pd.DataFrame] = None,
        catalyst_event=None,                # models.news.CatalystEvent or None
        reaction=None,                      # ReactionMetrics or None
        liquidity=None,                     # LiquidityProxyResult or None
        risk_plan=None,                     # RiskPlan or None
        trap_risk: Optional[int] = None,
        trap_warnings: Optional[List[str]] = None,
    ) -> IntradayIntelligence:
        intel = IntradayIntelligence(ticker=data.ticker, as_of=data.timestamp)
        bars_1m = self._normalize_bars(data.intraday_bars)

        if bars_1m.empty:
            intel.data_status = "NO_INTRADAY"
            intel.data_reasons.append("no_intraday_bars_from_provider")
            self._finalize_empty(intel, data, technical, trap_risk, trap_warnings)
            return intel

        if len(bars_1m) < 30:
            intel.data_status = "INSUFFICIENT_INTRADAY"
            intel.data_reasons.append(f"only_{len(bars_1m)}_1m_bars")

        # ---- timeframes ----
        frames = {"1m": bars_1m}
        for tf, rule in (("5m", "5min"), ("15m", "15min")):
            res = self._resample(bars_1m, rule)
            if res is not None:
                frames[tf] = res
        intel.timeframes = [
            TimeframeAnalysis(tf, len(frames[tf]), True) if tf in frames
            else TimeframeAnalysis(tf, 0, False, "insufficient_bars")
            for tf in ("1m", "5m", "15m")
        ]
        primary_tf = self._select_primary(frames)
        intel.timeframe = primary_tf
        if intel.data_status == "OK" and primary_tf == "1m" and len(bars_1m) < 60:
            intel.warnings.append("primary_timeframe_fallback_to_1m")

        tf = frames[primary_tf]
        price = data.current_price or float(tf["Close"].iloc[-1])

        # ---- levels ----
        intel.levels = self._build_levels(data, bars_1m, tf, daily_history)

        # ---- momentum + volume ----
        intel.momentum_volume = self._momentum_volume(tf, data, intel.levels.vwap, price)

        # ---- setups ----
        intel.setups = self._detect_setups(tf, bars_1m, price, intel.levels, intel.momentum_volume)

        # ---- entry intelligence ----
        setup = intel.primary_setup()
        atr_i = self._intraday_atr(tf)
        intel.entry = self._build_entry(setup, price, intel.levels, intel.momentum_volume, atr_i, tf)

        # ---- score ----
        intel.score = self._build_score(
            intel, data, technical, catalyst_event, reaction, liquidity,
            risk_plan, trap_risk, trap_warnings,
        )

        # ---- intraday-specific trap flags (no duplication of TrapEngine) ----
        intel.trap_flags = self._intraday_trap_flags(intel)
        return intel

    def _finalize_empty(self, intel, data, technical, trap_risk, trap_warnings):
        intel.score = self._build_score(
            intel, data, technical, None, None, None, None, trap_risk, trap_warnings
        )

    # ---------------- data prep ----------------

    @staticmethod
    def _normalize_bars(bars) -> pd.DataFrame:
        if bars is None:
            return pd.DataFrame()
        try:
            df = bars.copy()
        except Exception:
            return pd.DataFrame()
        if df.empty or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            return pd.DataFrame()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                return pd.DataFrame()
        try:
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        except Exception:
            return pd.DataFrame()
        df = df.sort_index()
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df = df[df["Close"].notna()]
        return df

    @staticmethod
    def _resample(bars_1m: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        """Deterministic resample of real 1m bars to 5m/15m. Left-labeled."""
        if bars_1m.empty or len(bars_1m) < 15:
            return None
        agg = {"Open": "first", "High": "max", "Low": "min",
               "Close": "last", "Volume": "sum"}
        out = bars_1m.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["Close"])
        return out if not out.empty else None

    @staticmethod
    def _select_primary(frames: dict) -> str:
        if "15m" in frames and len(frames["15m"]) >= 20:
            return "15m"
        if "5m" in frames and len(frames["5m"]) >= 24:
            return "5m"
        return "1m"

    # ---------------- levels ----------------

    def _build_levels(self, data: TickerData, bars_1m: pd.DataFrame,
                      tf: pd.DataFrame, daily_history: Optional[pd.DataFrame]) -> IntradayLevels:
        lv = IntradayLevels()
        lv._or_window_minutes = ORB_WINDOW_MINUTES

        if data.premarket.high is not None:
            lv.premarket_high = float(data.premarket.high)
        if data.premarket.low is not None:
            lv.premarket_low = float(data.premarket.low)

        if daily_history is not None and not daily_history.empty and \
                {"High", "Low"}.issubset(daily_history.columns):
            prev = daily_history.sort_index().iloc[-1]
            lv.previous_day_high = round(float(prev["High"]), 4)
            lv.previous_day_low = round(float(prev["Low"]), 4)

        regular = pd.DataFrame()
        if not bars_1m.empty and isinstance(bars_1m.index, pd.DatetimeIndex):
            mins = bars_1m.index.hour * 60 + bars_1m.index.minute
            regular = bars_1m[mins >= REGULAR_START_MINUTES]
        if not regular.empty:
            orb = regular.iloc[: max(1, ORB_WINDOW_MINUTES)]
            lv.opening_range_high = round(float(orb["High"].max()), 4)
            lv.opening_range_low = round(float(orb["Low"].min()), 4)

        # VWAP: prefer provider session snapshot; fall back to session-to-date calc.
        if data.regular.vwap:
            lv.vwap = float(data.regular.vwap)
            lv.vwap_source = "provider_regular_session"
        elif data.regular.volume and data.regular.volume > 0 and not regular.empty:
            tp = (regular["High"] + regular["Low"] + regular["Close"]) / 3.0
            vol_sum = float(regular["Volume"].sum())
            if vol_sum > 0:
                lv.vwap = round(float((tp * regular["Volume"]).sum() / vol_sum), 4)
                lv.vwap_source = "computed_regular_session_bars"

        if len(tf) >= 2 * PIVOT_WINDOW + 1:
            highs = self._pivots(tf["High"], kind="high")
            lows = self._pivots(tf["Low"], kind="low")
            price_now = float(tf["Close"].iloc[-1])
            # Role-aware S/R (display): pivot highs below price act as flipped support;
            # pivot lows above price act as flipped resistance. Standard market structure.
            sup_cands = [float(tf["Low"].iloc[i]) for i in lows
                         if tf["Low"].iloc[i] < price_now]
            sup_cands += [float(tf["High"].iloc[i]) for i in highs
                          if tf["High"].iloc[i] < price_now]
            res_cands = [float(tf["High"].iloc[i]) for i in highs
                         if tf["High"].iloc[i] > price_now]
            res_cands += [float(tf["Low"].iloc[i]) for i in lows
                          if tf["Low"].iloc[i] > price_now]
            lv.intraday_support = round(max(sup_cands), 4) if sup_cands else None
            lv.intraday_resistance = round(min(res_cands), 4) if res_cands else None

            tail = tf.tail(RECENT_WINDOW)
            lv.recent_high = round(float(tail["High"].max()), 4)
            lv.recent_low = round(float(tail["Low"].min()), 4)
        return lv

    @staticmethod
    def _pivots(series: pd.Series, kind: str) -> List[int]:
        n = len(series)
        w = PIVOT_WINDOW
        pivots: List[int] = []
        vals = series.values
        for i in range(w, n - w):
            left = vals[i - w:i]
            right = vals[i + 1:i + 1 + w]
            if kind == "high":
                if all(vals[i] > v for v in left) and all(vals[i] >= v for v in right):
                    pivots.append(i)
            else:
                if all(vals[i] < v for v in left) and all(vals[i] <= v for v in right):
                    pivots.append(i)
        return pivots

    # ---------------- momentum & volume ----------------

    def _momentum_volume(self, tf: pd.DataFrame, data: TickerData,
                         vwap: Optional[float], price: float) -> IntradayMomentumVolume:
        mv = IntradayMomentumVolume()
        closes = tf["Close"]
        k = min(MOMENTUM_LOOKBACK, max(2, len(closes) // 3))
        if len(closes) >= 2 * k:
            last = float(closes.iloc[-k:].mean())
            prior = float(closes.iloc[-2 * k:-k].mean())
            if prior > 0:
                mv.price_acceleration = round((last - prior) / prior * 100, 3)
                if mv.price_acceleration > 0.05:
                    mv.momentum_direction = "UP"
                elif mv.price_acceleration < -0.05:
                    mv.momentum_direction = "DOWN"
                else:
                    mv.momentum_direction = "FLAT"

        vols = tf["Volume"]
        if len(vols) >= 2 * k and float(vols.iloc[-2 * k:-k].mean()) > 0:
            last_avg = float(vols.iloc[-k:].mean())
            prior_avg = float(vols.iloc[-2 * k:-k].mean())
            mv.volume_acceleration = round(last_avg / prior_avg, 3)

        if len(vols) > 20 and float(vols.iloc[-21:-1].mean()) > 0:
            mean20 = float(vols.iloc[-21:-1].mean())
            ratio = float(vols.iloc[-1]) / mean20
            mv.volume_spike_ratio = round(ratio, 2)
            mv.volume_spike = ratio >= VOL_SPIKE_MULTIPLE

        mv.rvol = data.relative_volume
        if data.dollar_volume and len(tf) > 0:
            mv.dollar_volume_minute = round(float(data.dollar_volume) / len(tf), 0)

        if vwap is not None and price:
            mv.price_vs_vwap_pct = round((price - vwap) / vwap * 100, 2)
            mv.above_vwap = price > vwap
        return mv

    @staticmethod
    def _intraday_atr(tf: pd.DataFrame) -> Optional[float]:
        if len(tf) < 5:
            return None
        c = tf["Close"]
        tr = pd.concat([
            tf["High"] - tf["Low"],
            (tf["High"] - c.shift(1)).abs(),
            (tf["Low"] - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        val = float(tr.tail(14).mean())
        return round(val, 4) if val == val and val > 0 else None

    # ---------------- setups ----------------

    def _detect_setups(self, tf: pd.DataFrame, bars_1m: pd.DataFrame, price: float,
                       lv: IntradayLevels, mv: IntradayMomentumVolume) -> List[IntradaySetup]:
        s: List[IntradaySetup] = []
        closes = tf["Close"]
        n = len(closes)
        if n < 5 or price is None:
            s.append(IntradaySetup("NO_SETUP", "NEUTRAL", False,
                                   ["insufficient_intraday_history"]))
            return s

        vol_ok = bool(mv.volume_acceleration and mv.volume_acceleration >= VOL_CONFIRM_RATIO) \
            or bool(mv.volume_spike)
        last_close = float(closes.iloc[-1])
        prior_close = float(closes.iloc[-2]) if n > 1 else last_close

        # Real swing pivots on the analysis timeframe (value, index-free).
        pivot_highs = sorted({round(float(tf["High"].iloc[i]), 6)
                              for i in self._pivots(tf["High"], kind="high")})
        pivot_lows = sorted({round(float(tf["Low"].iloc[i]), 6)
                             for i in self._pivots(tf["Low"], kind="low")})
        lookback12 = closes.tail(12)

        # --- OPENING RANGE BREAKOUT ---
        if lv.opening_range_high is not None and lv.opening_range_low is not None:
            minutes_into_reg = self._minutes_into_regular(bars_1m)
            or_complete = minutes_into_reg is not None and minutes_into_reg >= ORB_WINDOW_MINUTES
            ev = [f"ORH {lv.opening_range_high} / ORL {lv.opening_range_low} "
                  f"(first {ORB_WINDOW_MINUTES}m)"]
            if or_complete and price > lv.opening_range_high:
                q = "CONFIRMED" if vol_ok else "UNCONFIRMED"
                if not vol_ok:
                    ev.append("volume confirmation unavailable or weak")
                s.append(IntradaySetup(
                    "OPENING_RANGE_BREAKOUT", "BULLISH", True, ev, q,
                    anchor_price=lv.opening_range_high, anchor_basis="OPENING_RANGE_HIGH"))
            elif or_complete and price < lv.opening_range_low:
                q = "CONFIRMED" if vol_ok else "UNCONFIRMED"
                s.append(IntradaySetup(
                    "OPENING_RANGE_BREAKOUT", "BEARISH", True, ev, q,
                    anchor_price=lv.opening_range_low, anchor_basis="OPENING_RANGE_LOW"))
            elif not or_complete:
                s.append(IntradaySetup("OPENING_RANGE_BREAKOUT", "NEUTRAL", False,
                                       ["opening_range_not_complete_yet"]))

        # --- BREAKOUT: close crossed above a real pivot high this bar ---
        crossed_up = [v for v in pivot_highs if prior_close <= v < last_close]
        if crossed_up:
            v = max(crossed_up)
            s.append(IntradaySetup(
                "BREAKOUT", "BULLISH", True,
                [f"closed {last_close:.4f} above intraday pivot high {v}",
                 f"volume_confirmed={vol_ok}"],
                "CONFIRMED" if vol_ok else "UNCONFIRMED",
                anchor_price=v, anchor_basis="INTRADAY_PIVOT_HIGH"))

        # --- BREAKDOWN: close crossed below a real pivot low this bar ---
        crossed_down = [v for v in pivot_lows if prior_close >= v > last_close]
        if crossed_down:
            v = min(crossed_down)
            s.append(IntradaySetup(
                "BREAKDOWN", "BEARISH", True,
                [f"closed {last_close:.4f} below intraday pivot low {v}",
                 f"volume_confirmed={vol_ok}"],
                "CONFIRMED" if vol_ok else "UNCONFIRMED",
                anchor_price=v, anchor_basis="INTRADAY_PIVOT_LOW"))

        # --- BREAKOUT + PULLBACK: holding just above a cleared pivot high ---
        pullback_levels = [v for v in pivot_highs
                           if float(lookback12.max()) > v and v < price <= v * 1.01]
        if pullback_levels:
            v = max(pullback_levels)
            s.append(IntradaySetup(
                "BREAKOUT_PULLBACK", "BULLISH", True,
                [f"cleared pivot high {v}, pulling back and holding above it "
                 f"(price {price:.4f})"],
                "CONFIRMED" if vol_ok else "UNCONFIRMED",
                anchor_price=v, anchor_basis="CLEARED_PIVOT_HIGH"))

        # --- VWAP family ---
        if lv.vwap is not None:
            v = lv.vwap
            was_below = any(float(c) < v for c in closes.tail(30)[:-1]) if n > 1 else False
            was_above = any(float(c) > v for c in closes.tail(30)[:-1]) if n > 1 else False
            if was_below and price > v:
                s.append(IntradaySetup(
                    "VWAP_RECLAIM", "BULLISH", True,
                    [f"crossed above VWAP {v:.4f} after trading below"],
                    "CONFIRMED" if bool(mv.volume_spike or (mv.volume_acceleration or 0) >= 1.25) else "UNCONFIRMED",
                    anchor_price=v, anchor_basis="VWAP"))
            elif was_above and price < v and min(
                    abs(float(c) - v) / v for c in closes.tail(5)) < 0.003:
                s.append(IntradaySetup(
                    "VWAP_REJECTION", "BEARISH", True,
                    [f"approached VWAP {v:.4f} and closed back below"], "CONFIRMED",
                    anchor_price=v, anchor_basis="VWAP"))
            elif price > v and was_above and mv.price_vs_vwap_pct is not None \
                    and 0 <= mv.price_vs_vwap_pct <= 1.0 \
                    and last_close >= float(closes.iloc[-2]):
                s.append(IntradaySetup(
                    "VWAP_BOUNCE", "BULLISH", True,
                    [f"held VWAP {v:.4f} on pullback ({mv.price_vs_vwap_pct}% above) "
                     f"and closed up"], "CONFIRMED",
                    anchor_price=v, anchor_basis="VWAP"))

        # --- MOMENTUM CONTINUATION ---
        if mv.momentum_direction == "UP" and mv.above_vwap and n >= 8:
            lows_tail = tf["Low"].tail(6)
            higher_lows = all(lows_tail.iloc[i] <= lows_tail.iloc[i + 1] + 1e-9
                              for i in range(len(lows_tail) - 2))
            if higher_lows:
                s.append(IntradaySetup(
                    "MOMENTUM_CONTINUATION", "BULLISH", True,
                    [f"acceleration {mv.price_acceleration}% over last {MOMENTUM_LOOKBACK} bars",
                     f"above VWAP, higher-low sequence intact"],
                    "CONFIRMED" if vol_ok else "UNCONFIRMED",
                    anchor_price=lv.recent_low, anchor_basis="RECENT_LOW"))
        elif mv.momentum_direction == "DOWN" and mv.above_vwap is False and n >= 8:
            highs_tail = tf["High"].tail(6)
            lower_highs = all(highs_tail.iloc[i] >= highs_tail.iloc[i + 1] - 1e-9
                              for i in range(len(highs_tail) - 2))
            if lower_highs:
                s.append(IntradaySetup(
                    "MOMENTUM_CONTINUATION", "BEARISH", True,
                    [f"acceleration {mv.price_acceleration}% over last {MOMENTUM_LOOKBACK} bars",
                     "below VWAP, lower-high sequence intact"],
                    "CONFIRMED" if vol_ok else "UNCONFIRMED",
                    anchor_price=lv.recent_high, anchor_basis="RECENT_HIGH"))

        # --- VOLUME EXPANSION (directional only when price agrees) ---
        if mv.rvol is not None and mv.rvol >= 2.0:
            direction = "BULLISH" if mv.momentum_direction == "UP" else \
                "BEARISH" if mv.momentum_direction == "DOWN" else "NEUTRAL"
            ev = [f"RVOL {mv.rvol}x avg"]
            if direction != "NEUTRAL":
                ev.append(f"price direction agrees ({direction})")
            else:
                ev.append("price direction unclear — informational only")
            s.append(IntradaySetup("VOLUME_EXPANSION", direction,
                                   direction != "NEUTRAL", ev,
                                   "CONFIRMED" if direction != "NEUTRAL" else "UNCONFIRMED"))

        # --- CONSOLIDATION BREAK ---
        cons_window = min(20, max(10, n // 3))
        if n > cons_window + 2:
            seg = tf.iloc[-(cons_window + 3):-3]
            width = float(seg["High"].max() - seg["Low"].min())
            if width > 0 and price > float(seg["High"].max()):
                edge = round(float(seg["High"].max()), 4)
                s.append(IntradaySetup(
                    "CONSOLIDATION_BREAK", "BULLISH", True,
                    [f"broke above consolidation high {edge} (width {width:.4f})",
                     f"volume_confirmed={vol_ok}"],
                    "CONFIRMED" if vol_ok else "UNCONFIRMED",
                    anchor_price=edge, anchor_basis="CONSOLIDATION_HIGH"))
            elif width > 0 and price < float(seg["Low"].min()):
                edge = round(float(seg["Low"].min()), 4)
                s.append(IntradaySetup(
                    "CONSOLIDATION_BREAK", "BEARISH", True,
                    [f"broke below consolidation low {edge} (width {width:.4f})",
                     f"volume_confirmed={vol_ok}"],
                    "CONFIRMED" if vol_ok else "UNCONFIRMED",
                    anchor_price=edge, anchor_basis="CONSOLIDATION_LOW"))

        # --- FAILED BREAKOUT: pivot high breached in lookback, back below now ---
        failed = [v for v in pivot_highs
                  if any(float(c) > v for c in lookback12) and price < v]
        if failed:
            v = max(failed)
            s.append(IntradaySetup(
                "FAILED_BREAKOUT", "BEARISH", True,
                [f"closes breached {v} then failed back below (now {price:.4f})"],
                "CONFIRMED",
                anchor_price=v, anchor_basis="FAILED_PIVOT_HIGH"))

        if not any(x.detected for x in s):
            s.append(IntradaySetup("NO_SETUP", "NEUTRAL", False,
                                   ["no intraday setup conditions met with available data"]))
        return s

    @staticmethod
    def _minutes_into_regular(bars_1m: pd.DataFrame) -> Optional[int]:
        if bars_1m.empty:
            return None
        mins = bars_1m.index.hour * 60 + bars_1m.index.minute
        reg = mins >= REGULAR_START_MINUTES
        if not reg.any():
            return None
        return int(mins[-1] - REGULAR_START_MINUTES) + 1

    # ---------------- entry intelligence ----------------

    def _build_entry(self, setup: Optional[IntradaySetup], price: float,
                     lv: IntradayLevels, mv: IntradayMomentumVolume,
                     atr_i: Optional[float], tf: pd.DataFrame) -> EntryPlan:
        plan = EntryPlan()
        if setup is None or not setup.detected or price is None:
            plan.reason = "no_detected_setup" if setup is None or not setup.detected \
                else "missing_price"
            return plan

        name = setup.name
        side = "LONG" if setup.direction == "BULLISH" else "SHORT"
        level = setup.anchor_price
        basis = setup.anchor_basis
        invalidation, inv_basis = None, None

        if level is None:
            plan.reason = "required_level_unavailable"
            plan.setup, plan.side = name, side
            plan.confirmations = self._confirmations(mv, lv, setup.quality)
            return plan

        if name == "OPENING_RANGE_BREAKOUT":
            invalidation = lv.opening_range_low if side == "LONG" else lv.opening_range_high
            inv_basis = "OPENING_RANGE_LOW" if side == "LONG" else "OPENING_RANGE_HIGH"
        elif name in ("VWAP_RECLAIM", "VWAP_BOUNCE"):
            # Entry at/above VWAP; stop belongs BELOW it (structural or ATR).
            invalidation, inv_basis = self._structural_stop_below(lv.vwap, lv, atr_i,
                                                                  exclude=lv.vwap)
        elif name == "VWAP_REJECTION":
            # Short into VWAP; stop belongs ABOVE it.
            invalidation, inv_basis = self._structural_stop_above(lv.vwap, lv, atr_i,
                                                                  exclude=lv.vwap)
        elif side == "LONG":
            invalidation, inv_basis = self._structural_stop_below(level, lv, atr_i,
                                                                  exclude=level)
        else:
            invalidation, inv_basis = self._structural_stop_above(level, lv, atr_i,
                                                                  exclude=level)

        cap = 0.01 if atr_i is None else max(0.002, (atr_i / price) * 0.75)
        if side == "LONG":
            zone_low, zone_high = level, min(price, level * (1 + cap))
            valid = price > level and invalidation is not None and invalidation < zone_low
            risk_ref = (zone_low, invalidation)
        else:
            zone_low, zone_high = max(price, level * (1 - cap)), level
            valid = price < level and invalidation is not None and invalidation > zone_high
            risk_ref = (invalidation, zone_high)

        if not valid:
            plan.reason = ("price_not_in_valid_entry_position"
                           if (side == "LONG" and price <= level) or
                              (side == "SHORT" and price >= level)
                           else "invalidation_does_not_define_positive_risk")
            plan.setup, plan.side = name, side
            plan.confirmations = self._confirmations(mv, lv, setup.quality)
            return plan

        risk_abs = abs(risk_ref[0] - risk_ref[1])
        risk_pct = risk_abs / zone_low * 100 if zone_low else None

        plan.status = "READY"
        plan.setup = name
        plan.side = side
        plan.entry_zone_low = round(zone_low, 4)
        plan.entry_zone_high = round(zone_high, 4)
        plan.invalidation_price = round(invalidation, 4)
        plan.invalidation_basis = inv_basis or basis
        plan.risk_distance_abs = round(risk_abs, 4)
        plan.risk_distance_pct = round(risk_pct, 2) if risk_pct is not None else None
        plan.confirmations = self._confirmations(mv, lv, setup.quality)
        met = sum(1 for c in plan.confirmations if c.met)
        evaluable = sum(1 for c in plan.confirmations if c.met is not None)
        base = 55 if setup.quality == "CONFIRMED" else 40
        if evaluable:
            plan.confidence = int(base + 15 * (met / evaluable))
        else:
            plan.confidence = base
        if risk_pct is not None and risk_pct > 3.0:
            plan.confidence = max(0, plan.confidence - 10)
            plan.evidence.append(f"WIDE_RISK: stop distance {plan.risk_distance_pct}%")
        plan.confidence = max(0, min(100, plan.confidence))
        plan.evidence.extend(setup.evidence[:2])
        plan.evidence.append(f"anchor {basis}={level}; stop basis {plan.invalidation_basis}")
        return plan

    def _structural_stop_below(self, bound: float, lv: IntradayLevels,
                               atr_i: Optional[float], exclude: Optional[float]):
        """Nearest real level strictly below `bound` (the entry-zone floor)."""
        cands: List[Tuple[float, str]] = []
        for nm, v in (("INTRADAY_SUPPORT", lv.intraday_support),
                      ("RECENT_LOW", lv.recent_low),
                      ("OPENING_RANGE_LOW", lv.opening_range_low)):
            if v is not None and v < bound and (exclude is None or abs(v - exclude) > 1e-6):
                cands.append((v, nm))
        if cands:
            return max(cands)
        if atr_i:
            return bound - max(atr_i, bound * 0.002), "ATR_INTRADAY"
        return None, ""

    def _structural_stop_above(self, bound: float, lv: IntradayLevels,
                               atr_i: Optional[float], exclude: Optional[float]):
        """Nearest real level strictly above `bound` (the entry-zone ceiling)."""
        cands: List[Tuple[float, str]] = []
        for nm, v in (("INTRADAY_RESISTANCE", lv.intraday_resistance),
                      ("RECENT_HIGH", lv.recent_high),
                      ("OPENING_RANGE_HIGH", lv.opening_range_high)):
            if v is not None and v > bound and (exclude is None or abs(v - exclude) > 1e-6):
                cands.append((v, nm))
        if cands:
            return min(cands)
        if atr_i:
            return bound + max(atr_i, bound * 0.002), "ATR_INTRADAY"
        return None, ""

    @staticmethod
    def _confirmations(mv: Optional[IntradayMomentumVolume], lv: IntradayLevels,
                       quality: str) -> List[ConfirmationCheck]:
        if mv is None:
            mv = IntradayMomentumVolume()
        checks = [
            ConfirmationCheck(
                "volume_expansion",
                (mv.volume_acceleration >= VOL_CONFIRM_RATIO) if mv.volume_acceleration is not None
                else (True if mv.volume_spike else None),
                f"accel={mv.volume_acceleration} spike_ratio={mv.volume_spike_ratio}"),
            ConfirmationCheck(
                "vwap_alignment", mv.above_vwap,
                f"{mv.price_vs_vwap_pct}% vs VWAP" if mv.price_vs_vwap_pct is not None else ""),
            ConfirmationCheck("setup_quality", quality == "CONFIRMED",
                              f"setup quality={quality}"),
        ]
        return checks

    # ---------------- score ----------------

    SCORE_WEIGHTS = {
        "Structure": 20.0,
        "Momentum": 20.0,
        "Volume": 15.0,
        "VWAP": 10.0,
        "Liquidity": 10.0,
        "Catalyst": 10.0,
        "TechnicalScore": 10.0,
        "RiskTrap": 5.0,
    }

    def _build_score(self, intel: IntradayIntelligence, data: TickerData,
                     technical, catalyst_event, reaction, liquidity,
                     risk_plan, trap_risk, trap_warnings) -> IntradayScore:
        comps: List[IntradayScoreComponent] = []
        mv = intel.momentum_volume

        # Structure — detected setup quality
        setup = intel.primary_setup()
        if setup and setup.detected:
            val = 85 if (setup.quality == "CONFIRMED" and setup.direction != "NEUTRAL") else \
                65 if setup.direction != "NEUTRAL" else 40
        else:
            val, reason = None, "no_setup_detected"
        comps.append(IntradayScoreComponent("Structure", self.SCORE_WEIGHTS["Structure"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["Structure"] / 100, 2) if val is not None else None))

        # Momentum
        if mv.price_acceleration is not None:
            a = mv.price_acceleration
            val = 80 if a > 0.5 else 60 if a > 0.05 else 40 if a > -0.05 else 20
        else:
            val, reason = None, "insufficient_history_for_acceleration"
        comps.append(IntradayScoreComponent("Momentum", self.SCORE_WEIGHTS["Momentum"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["Momentum"] / 100, 2) if val is not None else None))

        # Volume
        vol_signals = []
        if mv.rvol is not None:
            vol_signals.append(min(100.0, mv.rvol / 3.0 * 100))
        if mv.volume_acceleration is not None:
            vol_signals.append(min(100.0, mv.volume_acceleration / 2.0 * 100))
        if mv.volume_spike:
            vol_signals.append(90.0)
        if vol_signals:
            val = int(max(vol_signals))
        else:
            val, reason = None, "rvol_and_intraday_volume_unavailable"
        comps.append(IntradayScoreComponent("Volume", self.SCORE_WEIGHTS["Volume"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["Volume"] / 100, 2) if val is not None else None))

        # VWAP
        if mv.price_vs_vwap_pct is not None:
            p = mv.price_vs_vwap_pct
            val = 80 if p > 0.5 else 65 if p > 0 else 45 if p > -0.5 else 25
        else:
            val, reason = None, "vwap_unavailable"
        comps.append(IntradayScoreComponent("VWAP", self.SCORE_WEIGHTS["VWAP"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["VWAP"] / 100, 2) if val is not None else None))

        # Liquidity — reuse TickerData dollar volume + liquidity proxy when provided
        dv = data.dollar_volume
        liq_val = getattr(liquidity, "status", None)
        if dv is not None and dv > 0:
            val = 80 if dv >= 50_000_000 else 60 if dv >= 10_000_000 else 35
            if liq_val == "WEAK":
                val = min(val, 30)
        elif liq_val:
            val = 30 if liq_val == "WEAK" else 55
        else:
            val, reason = None, "dollar_volume_unavailable"
        comps.append(IntradayScoreComponent("Liquidity", self.SCORE_WEIGHTS["Liquidity"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["Liquidity"] / 100, 2) if val is not None else None))

        # Catalyst — reaction-based, never assumed
        label = getattr(reaction, "reaction_label", None) if reaction else None
        if label in ("STRONG_POSITIVE_REACTION", "POSITIVE_REACTION"):
            val = 80
        elif label == "NEGATIVE_REACTION":
            val = 15
        elif label in ("WEAK_REACTION", "NEUTRAL"):
            val = 45
        elif label == "DATA_INSUFFICIENT" or label is None:
            val, reason = None, "reaction_data_insufficient_or_no_event"
        else:
            val, reason = None, "unknown_reaction_state"
        comps.append(IntradayScoreComponent("Catalyst", self.SCORE_WEIGHTS["Catalyst"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["Catalyst"] / 100, 2) if val is not None else None))

        # TechnicalScore — reuse Phase 2.7 intelligence score
        ti_score = getattr(getattr(technical, "intelligence", None), "score", None) if technical else None
        if ti_score is not None and getattr(ti_score, "total", None) is not None and ti_score.total > 0:
            val = int(ti_score.total)
        else:
            val, reason = None, "technical_intelligence_unavailable"
        comps.append(IntradayScoreComponent("TechnicalScore", self.SCORE_WEIGHTS["TechnicalScore"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["TechnicalScore"] / 100, 2) if val is not None else None))

        # RiskTrap — inverse of trap burden (existing TrapEngine + flags)
        flags = list(trap_warnings or [])
        tr = trap_risk if trap_risk is not None else None
        if tr is not None:
            val = int(max(0, 100 - tr))
        elif flags:
            val = 60
        else:
            val, reason = None, "trap_inputs_unavailable"
        comps.append(IntradayScoreComponent("RiskTrap", self.SCORE_WEIGHTS["RiskTrap"],
                                            val, reason if val is None else None,
                                            round(val * self.SCORE_WEIGHTS["RiskTrap"] / 100, 2) if val is not None else None))

        avail_w = sum(c.weight for c in comps if c.value is not None)
        total = 0
        for c in comps:
            if c.value is not None and avail_w > 0:
                c.contribution = round((c.value / 100.0) * (c.weight / avail_w) * 100, 2)
                total += c.value * c.weight / avail_w
        score = IntradayScore(total=int(round(total)), components=comps)
        if intel.data_status != "OK":
            score.total = min(score.total, 49)   # degraded data can never look strong
        return score

    # ---------------- intraday-specific trap flags ----------------

    @staticmethod
    def _intraday_trap_flags(intel: IntradayIntelligence) -> List[str]:
        flags: List[str] = []
        names = [s.name for s in intel.setups if s.detected]
        if "FAILED_BREAKOUT" in names:
            flags.append("INTRADAY_FAILED_BREAKOUT")
        entry = intel.entry
        if entry.status == "READY":
            unmet = [c.name for c in entry.confirmations if c.met is False]
            if "volume_expansion" in unmet:
                flags.append("INTRADAY_WEAK_VOLUME_CONFIRMATION")
        mv = intel.momentum_volume
        if mv.price_vs_vwap_pct is not None and mv.price_vs_vwap_pct > 6:
            flags.append("INTRADAY_OVEREXTENDED_VS_VWAP")
        if intel.data_status != "OK":
            flags.append("INTRADAY_DATA_DEGRADED")
        return flags
