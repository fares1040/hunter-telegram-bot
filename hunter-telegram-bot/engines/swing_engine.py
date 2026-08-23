"""Hunter Bot — Swing Intelligence Engine (Phase 2.9).

Real, deterministic swing analysis built from daily history. Reuses the
Phase 2.7 TechnicalIntelligence (MA/MACD/RSI), Phase 2.6 Catalyst context,
and Phase 2.8 Intraday confirmation where supplied, but performs its own
swing-specific structure / S/R / setups / entry / score so it can also run
fully standalone on daily bars (no intraday dependency).

No targets are produced (Phase 2.10 scope). No Telegram output. The
DecisionEngine remains authoritative.

Causality: every computation depends only on bars at or before the last bar.
Confirmed pivots exclude the trailing SWING_PIVOT_WINDOW bars. Appending
future bars can never change the output at an earlier bar.
"""
import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from models.ticker import TickerData
from models.swing import (
    SwingIntelligence, SwingLevel, SwingTrend, SwingSetup, SwingMomentum,
    SwingVolume, SwingEntry, SwingConfirmationCheck, SwingScore,
    SwingScoreComponent, SwingCatalystContext,
)
from utils.logger import LOGGER

LOGGER = logging.getLogger("hunter")

SWING_MIN_HISTORY = 20                 # daily bars to produce any swing read
SWING_PIVOT_WINDOW = 3                 # daily bars each side for confirmed pivots
SWING_STRUCTURE_LOOKBACK = 120         # daily bars for HH/HL structure
SWING_SR_LOOKBACK = 90                 # daily bars for swing S/R pivots
SWING_MAJOR_LOOKBACK = 250             # daily bars for major high/low
BREAKOUT_THRESHOLD_PCT = 3.0           # price within 3% above a ceiling = fresh breakout
BREAKOUT_LOOKBACK = 20                  # bars for rolling recent high/low breakout detection
RETEST_TOL_PCT = 1.2                   # pullback that holds within this % of breakout level
PULLBACK_NEAR_PCT = 8.0                  # price within this % of support/resistance = pullback
FAILED_LOOKBACK = 20                   # bars to look back for a failed breakout
RANGE_LOOKBACK = 20                    # bars for range/base detection
BASE_LOOKBACK = 45
RANGE_TIGHT_PCT = 8.0                  # max high-low span % to call it a range/base
EXTENDED_PCT = 30.0                    # price > 30% above MA200 = extended
RISK_WIDE_PCT = 12.0                   # risk distance beyond this = excessive for swing


class SwingEngine:
    """Deterministic, real-data swing intelligence. Same input -> same output."""

    def build(
        self,
        data: TickerData,
        daily_history: Optional[pd.DataFrame] = None,
        technical_intelligence=None,               # models.technical.TechnicalIntelligence
        catalyst_event=None,                       # models.news.CatalystEvent
        catalyst_profile=None,                     # models.catalyst.CatalystProfile
        intraday_intelligence=None,                # IntradayIntelligence (optional confirmation)
        trap_risk: Optional[int] = None,
        trap_warnings: Optional[List[str]] = None,
    ) -> SwingIntelligence:
        intel = SwingIntelligence(ticker=data.ticker, as_of=data.timestamp)
        price = data.current_price
        if price is None and technical_intelligence is not None:
            price = technical_intelligence.current_price
        df = self._normalize_daily(daily_history)

        if df is None or len(df) < SWING_MIN_HISTORY or price is None:
            intel.data_status = "INSUFFICIENT_HISTORY"
            intel.data_reasons.append(
                "daily_history_too_short" if df is not None and len(df) < SWING_MIN_HISTORY
                else "no_daily_history"
            )
            if price is None:
                intel.data_reasons.append("no_current_price")
            self._fill_from_technical(intel, technical_intelligence, price)
            intel.catalyst = self._catalyst_context(catalyst_event, catalyst_profile)
            intel.score = self._build_score(intel, data, trap_risk, trap_warnings)
            intel.trap_flags = self._swing_trap_flags(intel, data, trap_risk, trap_warnings)
            return intel

        # ---- trend & structure ----
        intel.trend = self._build_trend(df, price, technical_intelligence)
        structure, struct_ev = self._swing_structure(df)
        if structure:
            intel.trend.structure = structure
            intel.trend.structure_evidence = struct_ev

        # ---- support / resistance ----
        intel.levels = self._build_levels(df, price)

        # ---- momentum & volume ----
        intel.momentum = self._build_momentum(df, technical_intelligence)
        intel.volume = self._build_volume(df, data, technical_intelligence)

        # ---- catalyst + intraday context ----
        intel.catalyst = self._catalyst_context(catalyst_event, catalyst_profile)
        intel.intraday_confirmation = self._intraday_confirm(intraday_intelligence, intel.trend.direction)

        # ---- setups ----
        intel.setups = self._detect_setups(df, price, intel, data)

        # ---- entry intelligence ----
        setup = intel.primary_setup()
        intel.entry = self._build_entry(setup, price, intel, data)

        # ---- risk / trap flags (reuses passed trap risk, adds swing-specific) ----
        intel.trap_flags = self._swing_trap_flags(intel, data, trap_risk, trap_warnings)

        # ---- score (after flags so RiskTrap penalty is applied) ----
        intel.score = self._build_score(intel, data, trap_risk, trap_warnings)

        # honesty ledger
        intel.missing_data = (
            intel.trend.missing + intel.momentum.missing + intel.volume.missing
            + intel.data_reasons
        )
        if intel.catalyst.present is False:
            intel.missing_data.append("no_catalyst_context")
        LOGGER.info(f"[Swing] {intel.summary()}")
        return intel

    # ------------------------------------------------------------------
    # data prep
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_daily(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
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

    @staticmethod
    def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b * 100, 2)

    @staticmethod
    def _safe_round(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return round(float(val), 4)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pivot_points(series: pd.Series, window: int = SWING_PIVOT_WINDOW, kind: str = "high") -> List[int]:
        """Deterministic, confirmed pivots. Excludes the trailing `window` bars
        so no pivot depends on the current (rightmost) bar -> no look-ahead."""
        pivots: List[int] = []
        values = series.values
        for i in range(window, len(values) - window):
            left = values[i - window:i]
            right = values[i + 1:i + 1 + window]
            if kind == "high":
                if all(values[i] > v for v in left) and all(values[i] >= v for v in right):
                    pivots.append(i)
            else:
                if all(values[i] < v for v in left) and all(values[i] <= v for v in right):
                    pivots.append(i)
        return pivots

    # ------------------------------------------------------------------
    # trend
    # ------------------------------------------------------------------
    def _fill_from_technical(self, intel: SwingIntelligence, tech, price):
        if tech is None:
            return
        t = intel.trend
        t.ma20 = tech.trend.ma20 if tech.trend else None
        t.ma50 = tech.trend.ma50 if tech.trend else None
        t.ma200 = tech.trend.ma200 if tech.trend else None
        if tech.trend:
            t.price_vs_ma50_pct = tech.trend.price_vs_ma50_pct
            t.price_vs_ma200_pct = tech.trend.price_vs_ma200_pct
            t.structure = tech.trend.structure
            t.structure_evidence = tech.trend.structure_evidence
        if price is not None and t.ma50 is not None:
            t.direction = "BULLISH" if price > t.ma50 else "BEARISH"

    def _build_trend(self, df: pd.DataFrame, price: float, tech) -> SwingTrend:
        t = SwingTrend()
        closes = df["Close"]
        # Prefer already-computed TechnicalIntelligence (Phase 2.7) when present.
        if tech is not None and tech.trend is not None:
            tt = tech.trend
            t.ma20 = tt.ma20
            t.ma50 = tt.ma50
            t.ma200 = tt.ma200
            t.price_vs_ma50_pct = tt.price_vs_ma50_pct
            t.price_vs_ma200_pct = tt.price_vs_ma200_pct
            t.ma_alignment = tt.ma_alignment
            t.ma50_slope_pct = tt.ma50_slope_pct
            t.ma200_slope_pct = None
            if tt.ma200 is None:
                t.missing.append("ma200_unavailable")
        else:
            t.ma20 = self._safe_round(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None
            t.ma50 = self._safe_round(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
            t.ma200 = self._safe_round(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
            t.price_vs_ma50_pct = self._pct_diff(price, t.ma50)
            t.price_vs_ma200_pct = self._pct_diff(price, t.ma200)
            if t.ma200 is None:
                t.missing.append("ma200_unavailable")
            # MA200 slope
            if len(closes) >= 205:
                s = closes.rolling(200).mean()
                prev = s.iloc[-6]
                if prev and not np.isnan(prev) and prev != 0:
                    t.ma200_slope_pct = round((s.iloc[-1] - prev) / prev * 100, 2)

        # MA alignment string
        parts = []
        if t.ma20 is not None:
            parts.append(("PRICE>" if price > t.ma20 else "PRICE<") + "MA20")
        if t.ma50 is not None:
            parts.append((">" if (t.ma20 or 0) > t.ma50 else "<") + "MA50")
        if t.ma200 is not None:
            above = (t.ma20 or 0) > t.ma200 and (t.ma50 or 0) > t.ma200
            parts.append((">" if above else "<") + "MA200")
        if parts:
            t.ma_alignment = ">".join(parts)

        # Direction — prioritize MA50/MA200
        if t.ma50 is not None and t.ma200 is not None:
            if price > t.ma50 > t.ma200:
                t.direction = "BULLISH"
            elif price < t.ma50 < t.ma200:
                t.direction = "BEARISH"
            elif (t.ma50 > t.ma200 and price < t.ma50) or (t.ma50 < t.ma200 and price > t.ma50):
                t.direction = "TRANSITION"
            else:
                t.direction = "NEUTRAL"
        elif price is not None and t.ma50 is not None:
            t.direction = "BULLISH" if price > t.ma50 else "BEARISH"
        else:
            t.direction = "UNKNOWN"
        return t

    def _swing_structure(self, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
        highs = df["High"].iloc[-SWING_STRUCTURE_LOOKBACK:]
        lows = df["Low"].iloc[-SWING_STRUCTURE_LOOKBACK:]
        ph = self._pivot_points(highs)[-2:]
        pl = self._pivot_points(lows, kind="low")[-2:]
        if len(ph) < 2 or len(pl) < 2:
            return None, ["fewer than two clean daily pivots in lookback"]
        h1, h2 = float(highs.iloc[ph[0]]), float(highs.iloc[ph[1]])
        l1, l2 = float(lows.iloc[pl[0]]), float(lows.iloc[pl[1]])
        evidence = [
            f"pivot high {round(h1, 2)} then {round(h2, 2)}",
            f"pivot low {round(l1, 2)} then {round(l2, 2)}",
        ]
        hh, hl = h2 > h1, l2 > l1
        lh, ll = h2 < h1, l2 < l1
        if hh and hl:
            return "HH_HL", evidence
        if lh and ll:
            return "LH_LL", evidence
        if (hh and ll) or (lh and hl):
            return "MIXED", evidence
        return None, evidence

    # ------------------------------------------------------------------
    # support / resistance
    # ------------------------------------------------------------------
    def _build_levels(self, df: pd.DataFrame, price: float) -> List[SwingLevel]:
        levels: List[SwingLevel] = []
        window = min(len(df), SWING_SR_LOOKBACK)
        hs = df["High"].iloc[-window:]
        ls = df["Low"].iloc[-window:]
        # swing pivot highs / lows
        for i in self._pivot_points(hs)[-4:]:
            v = float(hs.iloc[i])
            touches = int(((hs > v * 0.997) & (hs < v * 1.003)).sum())
            strength = min(50 + (touches - 1) * 10, 95)
            levels.append(SwingLevel(
                price=round(v, 2), level_type="RESISTANCE" if v >= price else "SUPPORT",
                strength=strength, distance_pct=self._pct_diff(v, price) or 0.0,
                evidence=f"daily swing high ({max(touches-1,1)} touches)", role="PIVOT",
            ))
        for i in self._pivot_points(ls, kind="low")[-4:]:
            v = float(ls.iloc[i])
            touches = int(((ls > v * 0.997) & (ls < v * 1.003)).sum())
            strength = min(50 + (touches - 1) * 10, 95)
            levels.append(SwingLevel(
                price=round(v, 2), level_type="RESISTANCE" if v >= price else "SUPPORT",
                strength=strength, distance_pct=self._pct_diff(v, price) or 0.0,
                evidence=f"daily swing low ({max(touches-1,1)} touches)", role="PIVOT",
            ))
        # major high / low over available history
        major = df.iloc[-min(len(df), SWING_MAJOR_LOOKBACK):]
        mh, ml = float(major["High"].max()), float(major["Low"].min())
        levels.append(SwingLevel(
            price=round(mh, 2), level_type="RESISTANCE" if mh >= price else "SUPPORT",
            strength=90, distance_pct=self._pct_diff(mh, price) or 0.0,
            evidence=f"major {min(len(df), SWING_MAJOR_LOOKBACK)}-day high", role="MAJOR",
        ))
        levels.append(SwingLevel(
            price=round(ml, 2), level_type="RESISTANCE" if ml >= price else "SUPPORT",
            strength=90, distance_pct=self._pct_diff(ml, price) or 0.0,
            evidence=f"major {min(len(df), SWING_MAJOR_LOOKBACK)}-day low", role="MAJOR",
        ))
        # role reversal: a former pivot high now below price = support (flipped)
        for lvl in list(levels):
            if lvl.role == "PIVOT" and "swing high" in lvl.evidence and lvl.price < price:
                lvl.role = "ROLE_REVERSAL"
                lvl.evidence += "; former resistance now support"
                lvl.strength = min(lvl.strength + 5, 100)
        # dedupe by 0.5%
        dedup: List[SwingLevel] = []
        for lvl in sorted(levels, key=lambda x: x.price):
            if any(abs(lvl.price - e.price) / price * 100 < 0.5 for e in dedup):
                continue
            dedup.append(lvl)
        return dedup

    # ------------------------------------------------------------------
    # momentum
    # ------------------------------------------------------------------
    def _build_momentum(self, df: pd.DataFrame, tech) -> SwingMomentum:
        m = SwingMomentum()
        if tech is not None and tech.momentum is not None and tech.momentum.rsi is not None:
            tm = tech.momentum
            m.rsi = tm.rsi
            m.roc_10 = tm.roc_10
            m.macd = tm.macd
            m.macd_signal = tm.macd_signal
            m.macd_hist = tm.macd_hist
            m.direction = tm.direction.value if hasattr(tm.direction, "value") else str(tm.direction)
            m.acceleration = tm.acceleration
            m.divergence = tm.divergence
            return m
        closes = df["Close"]
        if len(closes) < 15:
            m.missing.append("insufficient_history_for_momentum")
            return m
        m.rsi = self._calc_rsi(closes)
        if len(closes) >= 11:
            m.roc_10 = self._safe_round(closes.pct_change(10).iloc[-1] * 100)
        if len(closes) >= 21:
            m.roc_20 = self._safe_round(closes.pct_change(20).iloc[-1] * 100)
        if len(closes) >= 35:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            sig = macd_line.ewm(span=9, adjust=False).mean()
            m.macd = self._safe_round(macd_line.iloc[-1])
            m.macd_signal = self._safe_round(sig.iloc[-1])
            m.macd_hist = self._safe_round(macd_line.iloc[-1] - sig.iloc[-1])
        m.direction = self._momentum_label(m.rsi, m.macd_hist)
        return m

    @staticmethod
    def _calc_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def _momentum_label(rsi, macd_hist) -> str:
        if rsi is None:
            return "NEUTRAL"
        if rsi >= 70:
            base = "STRONG"
        elif rsi >= 55:
            base = "POSITIVE"
        elif rsi > 45:
            base = "NEUTRAL"
        elif rsi >= 30:
            base = "NEGATIVE"
        else:
            base = "WEAK"
        if macd_hist is not None:
            order = ["WEAK", "NEGATIVE", "NEUTRAL", "POSITIVE", "STRONG"]
            idx = order.index(base)
            if macd_hist > 0 and idx < 4 and base in ("NEUTRAL", "NEGATIVE", "WEAK"):
                base = order[idx + 1]
            elif macd_hist < 0 and idx > 0 and base in ("NEUTRAL", "POSITIVE", "STRONG"):
                base = order[idx - 1]
        return base

    # ------------------------------------------------------------------
    # volume
    # ------------------------------------------------------------------
    def _build_volume(self, df: pd.DataFrame, data: TickerData, tech) -> SwingVolume:
        v = SwingVolume()
        vols = df["Volume"].replace(0, np.nan).dropna()
        if len(vols) < 5:
            v.missing.append("insufficient_volume_history")
            return v
        last = float(vols.iloc[-1])
        prior = float(vols.iloc[-21:-1].mean()) if len(vols) >= 21 else float(vols.iloc[:-1].mean())
        v.last_bar_rvol = round(last / prior, 2) if prior > 0 else None
        # session relative volume from provider when available
        if tech is not None and tech.volume is not None and tech.volume.rvol is not None:
            v.rvol = tech.volume.rvol
        elif data.relative_volume is not None:
            v.rvol = data.relative_volume
        elif data.avg_volume_20d and data.regular.volume:
            v.rvol = round(data.regular.volume / data.avg_volume_20d, 2)
        # breakout expansion: last bar vs trailing baseline of prior bars
        base = float(vols.iloc[-10:-1].mean()) if len(vols) >= 10 else prior
        v.volume_expansion = bool(last > base * 1.5) if base > 0 else False
        # pullback contraction: last bar well below trailing baseline
        recent = float(vols.iloc[-4:].mean()) if len(vols) >= 4 else last
        v.pullback_contraction = bool(recent < base * 0.7) if base > 0 else False
        if data.dollar_volume:
            v.dollar_volume = data.dollar_volume
        r = v.rvol
        if r is None and v.last_bar_rvol is not None:
            r = v.last_bar_rvol
        if r is None:
            v.regime = "NORMAL"
            v.missing.append("rvol_unavailable")
        elif r < 0.7:
            v.regime = "LOW"
        elif r <= 1.2:
            v.regime = "NORMAL"
        elif r <= 2:
            v.regime = "ELEVATED"
        elif r <= 4:
            v.regime = "HIGH"
        else:
            v.regime = "EXTREME"
        return v

    # ------------------------------------------------------------------
    # catalyst + intraday context (reuse, never duplicate)
    # ------------------------------------------------------------------
    @staticmethod
    def _catalyst_context(event, profile) -> SwingCatalystContext:
        ctx = SwingCatalystContext()
        if event is None and profile is None:
            return ctx
        ctx.present = True
        if event is not None:
            ctx.sentiment = getattr(event, "sentiment", None)
            ctx.category = getattr(event.catalyst_type, "value", None) if hasattr(event.catalyst_type, "value") else str(getattr(event.catalyst_type, "name", ""))
            ctx.materiality = getattr(event, "materiality_score", None)
        if profile is not None:
            ctx.freshness = getattr(profile.freshness, "value", None) if hasattr(profile.freshness, "value") else str(getattr(profile.freshness, ""))
            ctx.materiality = ctx.materiality if ctx.materiality else getattr(profile, "materiality", None)
            ctx.is_trap_risk = bool(getattr(profile, "is_trap_risk", False))
            if ctx.sentiment is None:
                ctx.sentiment = getattr(profile.sentiment, "value", None) if hasattr(profile.sentiment, "value") else str(getattr(profile.sentiment, ""))
        if ctx.is_trap_risk:
            ctx.notes.append("catalyst_flagged_as_trap_risk")
        return ctx

    @staticmethod
    def _intraday_confirm(intraday, swing_dir: str) -> Optional[str]:
        if intraday is None or getattr(intraday, "data_status", "OK") == "NO_INTRADAY":
            return None
        setup = intraday.primary_setup() if hasattr(intraday, "primary_setup") else None
        if setup is None:
            return None
        aligns = (setup.direction == "BULLISH" and swing_dir in ("BULLISH", "TRANSITION")) or \
                 (setup.direction == "BEARISH" and swing_dir in ("BEARISH", "TRANSITION"))
        verdict = "aligned" if aligns else "conflicting"
        return f"intraday {setup.name}:{setup.direction} {verdict}"

    # ------------------------------------------------------------------
    # setups (10 deterministic)
    # ------------------------------------------------------------------
    def _detect_setups(self, df: pd.DataFrame, price: float, intel: SwingIntelligence, data: TickerData) -> List[SwingSetup]:
        s: List[SwingSetup] = []
        levels = intel.levels
        trend = intel.trend
        mom = intel.momentum
        vol = intel.volume
        closes = df["Close"]
        last_close = float(closes.iloc[-1])

        resistance = [l for l in levels if l.level_type == "RESISTANCE"]
        support = [l for l in levels if l.level_type == "SUPPORT"]
        nearest_res = min(resistance, key=lambda l: abs(l.price - price)) if resistance else None
        nearest_sup = max(support, key=lambda l: l.price) if support else None

        # Former supports/resistances for reference; breakouts are detected from
        # the rolling recent high/low (a broken level can never be a *confirmed*
        # pivot, since later bars exceed it — so pivots alone miss breakouts).
        former_lows = [l for l in levels if l.role == "PIVOT" and "swing low" in l.evidence]
        major_high = [l for l in levels if l.role == "MAJOR" and "high" in l.evidence]
        major_low = [l for l in levels if l.role == "MAJOR" and "low" in l.evidence]

        look = min(BREAKOUT_LOOKBACK, len(closes) - 1)
        pre = closes.iloc[:-1]
        recent_high = float(pre.iloc[-look:].max()) if look >= 1 else None
        recent_low = float(pre.iloc[-look:].min()) if look >= 1 else None

        # ---------- FAILED BREAKOUT (bearish) ----------
        failed_anchor = None
        if recent_high is not None:
            broke = any(closes.iloc[-FAILED_LOOKBACK:-1] >= recent_high)
            if broke and last_close < recent_high:
                failed_anchor = recent_high
                s.append(SwingSetup(
                    name="FAILED_BREAKOUT", direction="BEARISH", detected=True,
                    evidence=[f"daily close cleared {round(recent_high,2)} then fell back to {round(last_close,2)}"],
                    quality="CONFIRMED" if not vol.volume_expansion else "UNCONFIRMED",
                    anchor_price=round(recent_high, 2), anchor_basis="recent_high",
                ))

        # breakout anchor = most recent resistance just cleared
        anchor_res = None
        if recent_high is not None and last_close > recent_high:
            pct = (last_close - recent_high) / recent_high * 100
            if pct <= BREAKOUT_THRESHOLD_PCT * 3:  # fresh, not a parabolic blow-off
                anchor_res = round(recent_high, 2)

        # ---------- BREAKOUT (bullish) ----------
        if anchor_res is not None and (failed_anchor is None or abs(anchor_res - failed_anchor) / price * 100 > 1.0):
            q = "CONFIRMED" if vol.volume_expansion else "WATCH"
            s.append(SwingSetup(
                name="BREAKOUT", direction="BULLISH", detected=True,
                evidence=[f"price {round(price,2)} closed above recent high {anchor_res}",
                          f"volume {'expanding' if vol.volume_expansion else 'not expanding'} (rvol {vol.rvol})"],
                quality=q, anchor_price=anchor_res, anchor_basis="recent_high",
            ))

        # ---------- BREAKOUT + RETEST (bullish) ----------
        if anchor_res is not None and (failed_anchor is None or abs(anchor_res - failed_anchor) / price * 100 > 1.0):
            # last close pulled back to within RETEST_TOL of the broken level and held
            if abs(last_close - anchor_res) / anchor_res * 100 <= RETEST_TOL_PCT and last_close >= anchor_res * 0.999:
                q = "CONFIRMED" if vol.volume_expansion else "WATCH"
                s.append(SwingSetup(
                    name="BREAKOUT_RETEST", direction="BULLISH", detected=True,
                    evidence=[f"retesting breakout level {anchor_res} (close {round(last_close,2)}), holding",
                              f"volume {'expanding' if vol.volume_expansion else 'contraction' if vol.pullback_contraction else 'normal'}"],
                    quality=q, anchor_price=anchor_res, anchor_basis="recent_high",
                ))

        # ---------- base / range detection ----------
        # Exclude the final 5 bars (the breakout itself) so the consolidation
        # window reflects the base/range that was actually broken.
        cons = df.iloc[:-5] if len(df) > 5 else df
        rb = min(len(cons), RANGE_LOOKBACK)
        range_high = float(cons["High"].iloc[-rb:].max())
        range_low = float(cons["Low"].iloc[-rb:].min())
        range_pct = (range_high - range_low) / price * 100 if price else 0
        tight_range = range_pct <= RANGE_TIGHT_PCT and rb >= 10

        # ---------- RANGE BREAKOUT ----------
        if tight_range:
            if last_close > range_high:
                q = "CONFIRMED" if vol.volume_expansion else "WATCH"
                s.append(SwingSetup(
                    name="RANGE_BREAKOUT", direction="BULLISH", detected=True,
                    evidence=[f"broke {rb}-day range top {round(range_high,2)} (span {round(range_pct,1)}%) on volume {'expansion' if vol.volume_expansion else 'normal'}"],
                    quality=q, anchor_price=range_high, anchor_basis="range_top",
                ))
            elif last_close < range_low:
                q = "CONFIRMED" if vol.volume_expansion else "WATCH"
                s.append(SwingSetup(
                    name="RANGE_BREAKOUT", direction="BEARISH", detected=True,
                    evidence=[f"broke {rb}-day range bottom {round(range_low,2)} (span {round(range_pct,1)}%) on volume {'expansion' if vol.volume_expansion else 'normal'}"],
                    quality=q, anchor_price=range_low, anchor_basis="range_bottom",
                ))

        # ---------- BASE BREAKOUT (bullish, longer base) ----------
        bb = min(len(cons), BASE_LOOKBACK)
        base_high = float(cons["High"].iloc[-bb:].max())
        base_low = float(cons["Low"].iloc[-bb:].min())
        base_pct = (base_high - base_low) / price * 100 if price else 0
        if base_pct <= RANGE_TIGHT_PCT * 1.4 and bb >= 30 and last_close > base_high:
            q = "CONFIRMED" if vol.volume_expansion else "WATCH"
            s.append(SwingSetup(
                name="BASE_BREAKOUT", direction="BULLISH", detected=True,
                evidence=[f"multi-week base {round(base_low,2)}-{round(base_high,2)} (span {round(base_pct,1)}%) then breakout above {round(base_high,2)}"],
                quality=q, anchor_price=base_high, anchor_basis="base_top",
            ))

        # ---------- HIGHER-LOW CONTINUATION (bullish) ----------
        if trend.structure == "HH_HL" and nearest_sup is not None and last_close > nearest_sup.price:
            s.append(SwingSetup(
                name="HIGHER_LOW_CONTINUATION", direction="BULLISH", detected=True,
                evidence=[f"structure HH_HL; price {round(price,2)} holding above higher low {nearest_sup.price}"],
                quality="CONFIRMED" if trend.direction == "BULLISH" else "UNCONFIRMED",
                anchor_price=nearest_sup.price, anchor_basis=nearest_sup.evidence,
            ))

        # ---------- TREND CONTINUATION ----------
        bull_trend = trend.direction == "BULLISH" and (mom.direction in ("POSITIVE", "STRONG") or (mom.macd_hist or 0) > 0)
        bear_trend = trend.direction == "BEARISH" and (mom.direction in ("NEGATIVE", "WEAK") or (mom.macd_hist or 0) < 0)
        if bull_trend:
            anc = nearest_sup.price if nearest_sup else anchor_res.price if anchor_res else None
            s.append(SwingSetup(name="TREND_CONTINUATION", direction="BULLISH", detected=True,
                                evidence=[f"{trend.direction} trend + momentum {mom.direction}"],
                                quality="CONFIRMED", anchor_price=anc, anchor_basis="structure_support"))
        elif bear_trend:
            anc = nearest_res.price if nearest_res else None
            s.append(SwingSetup(name="TREND_CONTINUATION", direction="BEARISH", detected=True,
                                evidence=[f"{trend.direction} trend + momentum {mom.direction}"],
                                quality="CONFIRMED", anchor_price=anc, anchor_basis="structure_resistance"))

        # ---------- PULLBACK in UPTREND (bullish) ----------
        # support sits below price -> distance_pct is negative; pullback means
        # price is within PULLBACK_NEAR_PCT ABOVE the support.
        if trend.direction == "BULLISH" and nearest_sup is not None and nearest_sup.distance_pct < 0 and abs(nearest_sup.distance_pct) <= PULLBACK_NEAR_PCT:
            q = "CONFIRMED" if vol.pullback_contraction else "UNCONFIRMED"
            s.append(SwingSetup(
                name="PULLBACK_UPTREND", direction="BULLISH", detected=True,
                evidence=[f"uptrend pullback into support {nearest_sup.price} ({abs(nearest_sup.distance_pct):.1f}% above)",
                          f"volume {'contracting' if vol.pullback_contraction else 'normal'}"],
                quality=q, anchor_price=nearest_sup.price, anchor_basis=nearest_sup.evidence,
            ))

        # ---------- PULLBACK in DOWNTREND (bearish) ----------
        # resistance sits above price -> distance_pct positive; bounce means price
        # is within PULLBACK_NEAR_PCT BELOW the resistance.
        if trend.direction == "BEARISH" and nearest_res is not None and nearest_res.distance_pct > 0 and nearest_res.distance_pct <= PULLBACK_NEAR_PCT:
            q = "CONFIRMED" if vol.pullback_contraction else "UNCONFIRMED"
            s.append(SwingSetup(
                name="PULLBACK_DOWNTREND", direction="BEARISH", detected=True,
                evidence=[f"downtrend bounce into resistance {nearest_res.price} ({nearest_res.distance_pct:.1f}% below)",
                          f"volume {'contracting' if vol.pullback_contraction else 'normal'}"],
                quality=q, anchor_price=nearest_res.price, anchor_basis=nearest_res.evidence,
            ))

        # ---------- BREAKDOWN (bearish) ----------
        # price closed below the recent swing low (real support just broken)
        anchor_sup = None
        if recent_low is not None and last_close < recent_low:
            anchor_sup = round(recent_low, 2)
        elif nearest_sup is not None and nearest_sup.distance_pct <= 0 and last_close < nearest_sup.price:
            anchor_sup = nearest_sup.price
        if anchor_sup is not None:
            q = "CONFIRMED" if vol.volume_expansion else "WATCH"
            s.append(SwingSetup(
                name="BREAKDOWN", direction="BEARISH", detected=True,
                evidence=[f"price {round(price,2)} closed below support {anchor_sup}"],
                quality=q, anchor_price=anchor_sup, anchor_basis="recent_low",
            ))

        # contradiction note
        dirs = {x.direction for x in s if x.detected}
        if "BULLISH" in dirs and "BEARISH" in dirs:
            intel.warnings.append("conflicting_setups_detected:" + ",".join(x.name for x in s if x.detected))

        if not any(x.detected for x in s):
            s.append(SwingSetup(name="NO_SETUP", direction="NEUTRAL", detected=False,
                                evidence=["no swing setup conditions met with available data"]))
        return s

    # ------------------------------------------------------------------
    # entry intelligence (NO targets)
    # ------------------------------------------------------------------
    def _build_entry(self, setup: Optional[SwingSetup], price: float, intel: SwingIntelligence, data: TickerData) -> SwingEntry:
        e = SwingEntry()
        if setup is None or not setup.detected or setup.anchor_price is None:
            e.status = "UNKNOWN"
            e.reason = "no reliable swing anchor/setup"
            return e
        anchor = float(setup.anchor_price)
        bull = setup.direction == "BULLISH"
        levels = intel.levels
        if bull:
            # invalidation = nearest support below the anchor (structure), else below anchor
            below = [l for l in levels if l.level_type == "SUPPORT" and l.price < anchor]
            inv = max(below, key=lambda l: l.price).price if below else anchor * 0.985
            inv_basis = "nearest_support_below_anchor" if below else "anchor_minus_1.5%"
            # entry zone: break above anchor + small buffer
            buf = max(anchor * 0.003, 0.01)
            e.entry_zone_low = round(anchor, 2)
            e.entry_zone_high = round(anchor + buf, 2)
            e.side = "LONG"
        else:
            above = [l for l in levels if l.level_type == "RESISTANCE" and l.price > anchor]
            inv = min(above, key=lambda l: l.price).price if above else anchor * 1.015
            inv_basis = "nearest_resistance_above_anchor" if above else "anchor_plus_1.5%"
            buf = max(anchor * 0.003, 0.01)
            e.entry_zone_high = round(anchor, 2)
            e.entry_zone_low = round(anchor - buf, 2)
            e.side = "SHORT"
        e.invalidation_price = round(inv, 2)
        e.invalidation_basis = inv_basis
        e.setup = setup.name
        if e.entry_zone_low and e.entry_zone_high and e.invalidation_price:
            if bull:
                risk = e.entry_zone_low - e.invalidation_price
            else:
                risk = e.invalidation_price - e.entry_zone_high
            if risk > 0:
                e.risk_distance_abs = round(risk, 2)
                base = e.entry_zone_low if bull else e.entry_zone_high
                e.risk_distance_pct = round(risk / base * 100, 2)
        # confirmation text
        conf = []
        if bull:
            conf.append(f"daily close above {anchor}")
        else:
            conf.append(f"daily close below {anchor}")
        if intel.volume.volume_expansion:
            conf.append("on expanding volume")
        elif intel.volume.pullback_contraction:
            conf.append("on contracting (pullback) volume")
        if intel.intraday_confirmation and "aligned" in intel.intraday_confirmation:
            conf.append("intraday aligned")
        e.confirmation = "; ".join(conf) if conf else None
        e.confirmations = [
            SwingConfirmationCheck("breakout_close", True, f"price vs {anchor}"),
            SwingConfirmationCheck("volume_support", intel.volume.volume_expansion or None,
                                   "expansion" if intel.volume.volume_expansion else "normal/contraction"),
            SwingConfirmationCheck("intraday_alignment",
                                   bool(intel.intraday_confirmation and "aligned" in intel.intraday_confirmation),
                                   intel.intraday_confirmation or "no intraday data"),
        ]
        # confidence
        conf_score = 50
        if setup.quality == "CONFIRMED":
            conf_score += 15
        elif setup.quality == "WATCH":
            conf_score += 5
        if intel.volume.volume_expansion:
            conf_score += 10
        if intel.catalyst.present and not intel.catalyst.is_trap_risk:
            if intel.catalyst.sentiment in ("POSITIVE",):
                conf_score += 15
            elif intel.catalyst.sentiment in ("MIXED", "NEUTRAL"):
                conf_score += 5
        elif intel.catalyst.is_trap_risk:
            conf_score -= 20
        if intel.intraday_confirmation and "aligned" in intel.intraday_confirmation:
            conf_score += 10
        if e.risk_distance_pct and e.risk_distance_pct > RISK_WIDE_PCT:
            conf_score -= 10
        e.confidence = max(0, min(100, int(conf_score)))
        e.evidence = [f"anchor={anchor} ({setup.anchor_basis})", f"setup={setup.name} quality={setup.quality}"]
        if e.risk_distance_pct:
            e.evidence.append(f"risk={e.risk_distance_pct}%")
        e.status = "READY" if (e.entry_zone_low and e.entry_zone_high and e.invalidation_price and e.risk_distance_abs and e.risk_distance_abs > 0) else "UNKNOWN"
        if e.status == "UNKNOWN":
            e.reason = "could not derive a positive-risk entry zone"
        return e

    # ------------------------------------------------------------------
    # score (0-100, renormalized, missing != bullish)
    # ------------------------------------------------------------------
    SCORE_WEIGHTS = {
        "Trend": 22, "Structure": 14, "Momentum": 16, "Volume": 12,
        "SupportResistance": 12, "Catalyst": 12, "RiskTrap": 7, "SetupQuality": 5,
    }

    def _build_score(self, intel: SwingIntelligence, data: TickerData, trap_risk, trap_warnings) -> SwingScore:
        comps: List[SwingScoreComponent] = []
        # Trend
        t = intel.trend
        if t.direction == "UNKNOWN":
            comps.append(SwingScoreComponent("Trend", self.SCORE_WEIGHTS["Trend"], None, reason="insufficient_history"))
        else:
            val = {"BULLISH": 78, "NEUTRAL": 52, "TRANSITION": 46, "BEARISH": 26}[t.direction]
            if t.price_vs_ma200_pct is not None and t.price_vs_ma200_pct > 0:
                val += 6
            if t.ma_alignment and "MA50>MA200" in t.ma_alignment and t.direction == "BULLISH":
                val += 6
            if t.ma_alignment and "MA50<MA200" in t.ma_alignment and t.direction == "BEARISH":
                val -= 6
            comps.append(SwingScoreComponent("Trend", self.SCORE_WEIGHTS["Trend"], max(0, min(100, int(val))),
                                             reason=f"{t.direction}; ma50/200 alignment"))
        # Structure
        st = t.structure
        if st is None:
            comps.append(SwingScoreComponent("Structure", self.SCORE_WEIGHTS["Structure"], None, reason="no_confirmed_pivots"))
        else:
            val = {"HH_HL": 85, "LH_LL": 25, "MIXED": 45}.get(st, 50)
            comps.append(SwingScoreComponent("Structure", self.SCORE_WEIGHTS["Structure"], val, reason=st))
        # Momentum
        m = intel.momentum
        if m.rsi is None:
            comps.append(SwingScoreComponent("Momentum", self.SCORE_WEIGHTS["Momentum"], None, reason="insufficient_history"))
        else:
            val = {"STRONG": 90, "POSITIVE": 70, "NEUTRAL": 50, "NEGATIVE": 30, "WEAK": 15}[m.direction]
            comps.append(SwingScoreComponent("Momentum", self.SCORE_WEIGHTS["Momentum"], val, reason=f"RSI {m.rsi}; {m.direction}"))
        # Volume
        v = intel.volume
        if v.rvol is None and v.last_bar_rvol is None:
            comps.append(SwingScoreComponent("Volume", self.SCORE_WEIGHTS["Volume"], None, reason="rvol_unavailable"))
        else:
            val = {"LOW": 35, "NORMAL": 55, "ELEVATED": 70, "HIGH": 82, "EXTREME": 90}[v.regime]
            comps.append(SwingScoreComponent("Volume", self.SCORE_WEIGHTS["Volume"], val, reason=f"rvol {v.rvol or v.last_bar_rvol} ({v.regime})"))
        # Support / Resistance
        if not intel.levels:
            comps.append(SwingScoreComponent("SupportResistance", self.SCORE_WEIGHTS["SupportResistance"], None, reason="no_levels"))
        else:
            val = 50
            ns = max([l for l in intel.levels if l.level_type == "SUPPORT"], key=lambda l: l.price) if any(l.level_type == "SUPPORT" for l in intel.levels) else None
            nr = min([l for l in intel.levels if l.level_type == "RESISTANCE"], key=lambda l: abs(l.price - (data.current_price or 0))) if any(l.level_type == "RESISTANCE" for l in intel.levels) else None
            if ns and -3 <= ns.distance_pct <= 0:
                val += 12
            if nr and nr.distance_pct >= 5:
                val += 12
            elif nr and nr.distance_pct < 2:
                val -= 8
            primary = intel.primary_setup()
            if primary and primary.direction == "BULLISH":
                val += 8
            elif primary and primary.direction == "BEARISH":
                val -= 8
            comps.append(SwingScoreComponent("SupportResistance", self.SCORE_WEIGHTS["SupportResistance"], max(0, min(100, int(val))), reason=f"{len(intel.levels)} levels"))
        # Catalyst
        c = intel.catalyst
        if not c.present:
            comps.append(SwingScoreComponent("Catalyst", self.SCORE_WEIGHTS["Catalyst"], None, reason="no_catalyst"))
        else:
            mat = c.materiality or 40
            if c.is_trap_risk:
                val = 20
            elif c.sentiment == "POSITIVE":
                val = min(55 + mat // 2, 95)
            elif c.sentiment == "NEGATIVE":
                val = max(20, 55 - mat // 2)
            else:
                val = 50
            comps.append(SwingScoreComponent("Catalyst", self.SCORE_WEIGHTS["Catalyst"], val, reason=f"{c.sentiment}; mat {mat}; trap={c.is_trap_risk}"))
        # RiskTrap
        penalty = self._swing_penalty(intel)
        if trap_risk is not None:
            risk_val = 100 - max(int(trap_risk), penalty)
            comps.append(SwingScoreComponent("RiskTrap", self.SCORE_WEIGHTS["RiskTrap"], max(0, min(100, int(risk_val))), reason=f"trap_risk={trap_risk}; swing_penalty={penalty}"))
        else:
            if penalty > 0:
                comps.append(SwingScoreComponent("RiskTrap", self.SCORE_WEIGHTS["RiskTrap"], max(0, min(100, 100 - penalty)), reason=f"swing_flags_penalty={penalty}"))
            else:
                comps.append(SwingScoreComponent("RiskTrap", self.SCORE_WEIGHTS["RiskTrap"], None, reason="no_risk_signal_available"))
        # SetupQuality
        primary = intel.primary_setup()
        if primary is None or not primary.detected:
            comps.append(SwingScoreComponent("SetupQuality", self.SCORE_WEIGHTS["SetupQuality"], None, reason="no_setup"))
        else:
            val = {"CONFIRMED": 85, "WATCH": 60, "UNCONFIRMED": 45}.get(primary.quality, 45)
            comps.append(SwingScoreComponent("SetupQuality", self.SCORE_WEIGHTS["SetupQuality"], val, reason=primary.name))

        available = [c for c in comps if c.value is not None]
        total_w = sum(c.weight for c in available)
        total = 0
        if total_w:
            raw = sum((c.value or 0) * c.weight for c in available) / total_w
            total = max(0, min(100, int(round(raw))))
            for c in available:
                c.contribution = round((c.value / 100) * (c.weight / total_w) * 100, 2)
        return SwingScore(total=total, components=comps)

    @staticmethod
    def _swing_penalty(intel: SwingIntelligence) -> int:
        p = 0
        if "FAILED_BREAKOUT" in intel.trap_flags:
            p += 25
        if "WEAK_BREAKOUT_VOLUME" in intel.trap_flags:
            p += 12
        if "EXTENDED_PRICE" in intel.trap_flags:
            p += 15
        if "MAJOR_RESISTANCE_TOO_CLOSE" in intel.trap_flags:
            p += 10
        if "BEARISH_CATALYST" in intel.trap_flags or "DILUTION_OFFERING" in intel.trap_flags:
            p += 18
        if "BROKEN_STRUCTURE" in intel.trap_flags:
            p += 20
        if "POOR_LIQUIDITY" in intel.trap_flags:
            p += 12
        if "EXCESSIVE_RISK_DISTANCE" in intel.trap_flags:
            p += 10
        return min(p, 95)

    def _swing_trap_flags(self, intel: SwingIntelligence, data: TickerData, trap_risk, trap_warnings) -> List[str]:
        flags: List[str] = []
        if trap_warnings:
            flags.extend(trap_warnings)
        if "FAILED_BREAKOUT" in [s.name for s in intel.setups if s.detected]:
            flags.append("FAILED_BREAKOUT")
        # weak breakout volume
        if any(s.name in ("BREAKOUT", "RANGE_BREAKOUT", "BASE_BREAKOUT") and s.detected for s in intel.setups) \
                and not intel.volume.volume_expansion:
            flags.append("WEAK_BREAKOUT_VOLUME")
        # extended price
        if intel.trend.ma200 and data.current_price and data.current_price > intel.trend.ma200 * (1 + EXTENDED_PCT / 100):
            flags.append("EXTENDED_PRICE")
        # major resistance too close
        res = [l for l in intel.levels if l.level_type == "RESISTANCE"]
        if res:
            nr = min(res, key=lambda l: abs(l.price - (data.current_price or 0)))
            if nr.distance_pct < 2:
                flags.append("MAJOR_RESISTANCE_TOO_CLOSE")
        # catalyst bearish / dilution
        if intel.catalyst.is_trap_risk:
            flags.append("DILUTION_OFFERING" if (intel.catalyst.category and "DILUTION" in str(intel.catalyst.category)) else "BEARISH_CATALYST")
        # broken structure: price below a higher-low support that defined HH_HL
        if intel.trend.structure == "HH_HL":
            sup = [l for l in intel.levels if l.level_type == "SUPPORT"]
            if sup:
                ns = max(sup, key=lambda l: l.price)
                if data.current_price and data.current_price < ns.price * 0.995:
                    flags.append("BROKEN_STRUCTURE")
        # poor liquidity
        if (data.avg_volume_20d and data.avg_volume_20d < 1_000_000) or (data.float_shares and data.float_shares < 20_000_000):
            flags.append("POOR_LIQUIDITY")
        # excessive risk distance
        if intel.entry.risk_distance_pct and intel.entry.risk_distance_pct > RISK_WIDE_PCT:
            flags.append("EXCESSIVE_RISK_DISTANCE")
        # de-dup preserve order
        seen = set()
        out = []
        for f in flags:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out
