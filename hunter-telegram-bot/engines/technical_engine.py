"""Hunter Bot — Technical Engine

Legacy TechnicalProfile pipeline (setup/score consumed by Risk/Trap/Decision)
is preserved byte-for-byte. Phase 2.7 adds a structured, explainable
TechnicalIntelligence layer built from the SAME loaded market data — no extra
network calls, pure CPU-side pandas math. Missing data yields
UNKNOWN/UNAVAILABLE values, never fabricated ones.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from models.ticker import TickerData
from models.technical import (
    MomentumIntelligence,
    MomentumLabel,
    PriceLevel,
    ScoreComponent,
    SetupDetection,
    SupportResistanceIntelligence,
    TechnicalIntelligence,
    TechnicalScore,
    TrendDirection,
    TrendIntelligence,
    VolatilityIntelligence,
    VolatilityRegime,
    VolumeIntelligence,
    VolumeRegime,
    VwapIntelligence,
)
from core.session_clock import SessionClock, MarketSession
from utils.logger import LOGGER


@dataclass
class TechnicalProfile:
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    ma200: Optional[float] = None
    rsi: Optional[float] = None
    vwap: Optional[float] = None
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    premarket_high: Optional[float] = None
    premarket_low: Optional[float] = None
    intraday_high: Optional[float] = None
    intraday_low: Optional[float] = None
    recent_swing_high: Optional[float] = None
    recent_swing_low: Optional[float] = None

    setup: str = "NO_SETUP"
    setup_score: int = 0
    trend_score: int = 0
    structure_note: str = ""
    warnings: List[str] = None
    intelligence: Optional["TechnicalIntelligence"] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TechnicalEngine:
    def analyze(self, data: TickerData, history_3mo: pd.DataFrame) -> TechnicalProfile:
        profile = TechnicalProfile()
        price = data.current_price

        if price is None:
            profile.structure_note = "DATA_INSUFFICIENT"
            return profile

        if history_3mo is None or history_3mo.empty:
            # Minimal analysis with just current data
            profile.vwap = data.regular.vwap or data.premarket.vwap
            profile.premarket_high = data.premarket.high
            profile.premarket_low = data.premarket.low
            profile.intraday_high = data.regular.high or data.premarket.high
            profile.intraday_low = data.regular.low or data.premarket.low
            self._classify_setup(profile, data)
            self._score_structure(profile, data)
            LOGGER.info(f"[Technical] {data.ticker} | Setup: {profile.setup} | Score: {profile.setup_score} (no history)")
            return profile

        closes = history_3mo["Close"]
        highs = history_3mo["High"]
        lows = history_3mo["Low"]

        profile.ma20 = self._safe_round(closes.rolling(20).mean().iloc[-1])
        profile.ma50 = self._safe_round(closes.rolling(50).mean().iloc[-1])
        if len(closes) >= 200:
            profile.ma200 = self._safe_round(closes.rolling(200).mean().iloc[-1])
        else:
            profile.ma200 = profile.ma50

        profile.rsi = self._calculate_rsi(closes)
        profile.atr = self._calculate_atr(highs, lows, closes)

        if len(closes) >= 20:
            std20 = closes.rolling(20).std().iloc[-1]
            if profile.ma20:
                profile.bb_upper = self._safe_round(profile.ma20 + 2 * std20)
                profile.bb_lower = self._safe_round(profile.ma20 - 2 * std20)

        if len(highs) >= 2:
            profile.prev_day_high = self._safe_round(highs.iloc[-2])
            profile.prev_day_low = self._safe_round(lows.iloc[-2])

        profile.premarket_high = data.premarket.high
        profile.premarket_low = data.premarket.low
        profile.intraday_high = data.regular.high or data.premarket.high
        profile.intraday_low = data.regular.low or data.premarket.low

        if len(highs) >= 10:
            profile.recent_swing_high = self._safe_round(highs.iloc[-10:].max())
            profile.recent_swing_low = self._safe_round(lows.iloc[-10:].min())

        profile.vwap = data.regular.vwap or data.premarket.vwap

        self._classify_setup(profile, data)
        self._score_structure(profile, data)

        try:
            profile.intelligence = self.build_intelligence(data, history_3mo)
        except Exception as e:  # intelligence must never break the legacy pipeline
            LOGGER.warning(f"[Technical] Intelligence build failed for {data.ticker}: {e}")

        LOGGER.info(f"[Technical] {data.ticker} | Setup: {profile.setup} | Score: {profile.setup_score}")
        return profile

    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return round(100 - (100 / (1 + rs)), 2)

    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Optional[float]:
        if len(highs) < period + 1:
            return None
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift(1))
        tr3 = abs(lows - closes.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return round(tr.rolling(period).mean().iloc[-1], 2)

    def _classify_setup(self, p: TechnicalProfile, data: TickerData):
        price = data.current_price
        if price is None:
            p.setup = "NO_SETUP"
            return

        breakout_levels = [l for l in [p.prev_day_high, p.recent_swing_high, p.premarket_high] if l]
        if breakout_levels:
            nearest_resistance = min(breakout_levels, key=lambda x: abs(x - price))
            if price > nearest_resistance * 1.02:
                p.setup = "BREAKOUT"
                p.structure_note = f"Price {price} above resistance {nearest_resistance}"
                return
            elif price > nearest_resistance * 0.97 and price < nearest_resistance * 1.02:
                p.setup = "RESISTANCE_TEST"
                p.structure_note = f"Testing resistance at {nearest_resistance}"
                return

        if p.vwap and price > p.vwap * 1.01 and data.change_percent and data.change_percent > 3:
            p.setup = "VWAP_RECLAIM"
            p.structure_note = f"Price reclaimed VWAP {p.vwap}"
            return

        if p.ma20 and price > p.ma20 and p.vwap and abs(price - p.vwap) / p.vwap < 0.02:
            p.setup = "BREAKOUT_PULLBACK"
            p.structure_note = "Pulling back to VWAP/MA20 after move"
            return

        if data.change_percent and data.change_percent > 5 and p.rsi and 50 < p.rsi < 75:
            p.setup = "MOMENTUM"
            p.structure_note = "Strong momentum, not overbought"
            return

        p.setup = "NO_SETUP"
        p.structure_note = "No clear technical setup"

    def _score_structure(self, p: TechnicalProfile, data: TickerData):
        price = data.current_price
        score = 50

        if p.ma20 and p.ma50 and price:
            if price > p.ma20 > p.ma50:
                score += 15
            elif price > p.ma20:
                score += 8
            elif price < p.ma50:
                score -= 10

        if p.ma200 and price and price > p.ma200:
            score += 5

        if p.rsi:
            if 50 <= p.rsi <= 70:
                score += 10
            elif p.rsi > 75:
                score -= 15
                p.warnings.append("OVERBOUGHT")
            elif p.rsi < 30:
                score -= 5
                p.warnings.append("OVERSOLD_MOMENTUM_RISK")

        if p.vwap and price:
            if price > p.vwap:
                score += 8
            else:
                score -= 10

        setup_bonus = {
            "BREAKOUT": 15,
            "VWAP_RECLAIM": 10,
            "MOMENTUM": 8,
            "BREAKOUT_PULLBACK": 5,
            "RESISTANCE_TEST": 0,
            "NO_SETUP": -10,
        }
        score += setup_bonus.get(p.setup, 0)

        if data.gap_percent and data.gap_percent > 20:
            score -= 10
            p.warnings.append("HUGE_GAP")

        if data.float_shares and data.float_shares < 50_000_000:
            p.warnings.append("LOW_FLOAT")

        p.setup_score = max(0, min(100, score))
        p.trend_score = max(0, min(100, score))

    def _safe_round(self, val) -> Optional[float]:
        if pd.isna(val) or val is None:
            return None
        return round(float(val), 2)

    # ------------------------------------------------------------------
    # Phase 2.7 — Structured Technical Intelligence
    # All calculations are CPU-side on already-loaded data. Any component
    # lacking sufficient history reports UNAVAILABLE instead of a guess.
    # ------------------------------------------------------------------

    REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    def _prepare_history(self, history: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if history is None or not isinstance(history, pd.DataFrame) or history.empty:
            return None
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in history.columns]
        if missing_cols:
            return None
        df = history[list(self.REQUIRED_COLUMNS)].copy()
        df = df.sort_index()
        df = df.dropna(subset=["Close"])
        return df if len(df) else None

    @staticmethod
    def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        """% of b from a to b: ((a-b)/b)*100, None-safe."""
        if a is None or b is None or b == 0:
            return None
        return round((a - b) / b * 100, 2)

    def _pivot_points(self, series: pd.Series, window: int = 3, kind: str = "high") -> List[int]:
        """Deterministic pivots. Highs: strict-left / >=-right peaks.
        Lows: strict-left / <=-right troughs."""
        pivots = []
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

    def build_intelligence(
        self,
        data: TickerData,
        history_3mo: Optional[pd.DataFrame],
        timeframe: str = "1d",
    ) -> TechnicalIntelligence:
        df = self._prepare_history(history_3mo)
        price = data.current_price
        intel = TechnicalIntelligence(ticker=data.ticker, timeframe=timeframe, current_price=price)

        intel.trend = self._trend_intelligence(df, price, timeframe)
        intel.momentum = self._momentum_intelligence(df)
        intel.volatility = self._volatility_intelligence(df, price)
        intel.vwap = self._vwap_intelligence(data)
        intel.volume = self._volume_intelligence(data, df)
        intel.support_resistance = self._sr_intelligence(df, price, data)
        intel.setups = self._detect_setups(intel, data, recent_closes=df["Close"].tail(5).tolist() if df is not None else None)
        detected = [s for s in intel.setups if s.detected]
        intel.primary_setup = detected[0] if detected else None
        for s in intel.setups:
            if s.name == "NO_SETUP":
                continue
        intel.missing_data = (
            intel.trend.missing + intel.momentum.missing + intel.volatility.missing
            + intel.volume.missing + intel.support_resistance.notes
        )
        intel.score = self._technical_score(intel)
        LOGGER.info(f"[TechnicalIntel] {intel.summary()}")
        return intel

    # ---------------- TREND ----------------
    def _trend_intelligence(self, df: Optional[pd.DataFrame], price: Optional[float], timeframe: str) -> TrendIntelligence:
        t = TrendIntelligence(timeframe=timeframe)
        if df is None or price is None or len(df) < 21:
            t.missing.append("insufficient_history_for_trend")
            return t

        closes = df["Close"]
        t.ma20 = self._safe_round(closes.rolling(20).mean().iloc[-1])
        t.ma50 = self._safe_round(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
        t.ma200 = self._safe_round(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
        t.price_vs_ma20_pct = self._pct_diff(price, t.ma20)
        t.price_vs_ma50_pct = self._pct_diff(price, t.ma50)
        t.price_vs_ma200_pct = self._pct_diff(price, t.ma200)

        if len(closes) >= 26:
            ma20_series = closes.rolling(20).mean()
            prev = ma20_series.iloc[-6]
            if prev and not pd.isna(prev) and prev != 0:
                t.ma20_slope_pct = round((ma20_series.iloc[-1] - prev) / prev * 100, 2)
        if len(closes) >= 56:
            ma50_series = closes.rolling(50).mean()
            prev = ma50_series.iloc[-6]
            if prev and not pd.isna(prev) and prev != 0:
                t.ma50_slope_pct = round((ma50_series.iloc[-1] - prev) / prev * 100, 2)

        parts = []
        if t.ma20:
            parts.append(("PRICE>" if price > t.ma20 else "PRICE<") + "MA20")
        if t.ma50:
            parts.append((">" if t.ma20 and t.ma20 > t.ma50 else "<") + "MA50")
        if t.ma200:
            parts.append((">" if (t.ma50 or t.ma20) and max(x for x in [t.ma50, t.ma20] if x) > t.ma200 else "<") + "MA200")
        t.ma_alignment = ">".join(parts) if parts else None

        structure, evidence = self._market_structure(df)
        t.structure = structure
        t.structure_evidence = evidence

        bullish_stack = price > (t.ma20 or price) and (t.ma20 or 0) > (t.ma50 or 0) and (t.ma50 or 0) > (t.ma200 or 0) and t.ma200 is not None
        bearish_stack = price < (t.ma20 or price) and (t.ma20 or np.inf) < (t.ma50 or np.inf) and (t.ma50 or np.inf) < (t.ma200 or np.inf) and t.ma200 is not None

        if structure == "HH_HL" and (t.price_vs_ma20_pct or 0) > 0:
            t.direction = TrendDirection.BULLISH
        elif structure == "LH_LL" and (t.price_vs_ma20_pct or 0) < 0:
            t.direction = TrendDirection.BEARISH
        elif bullish_stack and (t.price_vs_ma20_pct or 0) > 0:
            t.direction = TrendDirection.BULLISH
        elif bearish_stack and (t.price_vs_ma20_pct or 0) < 0:
            t.direction = TrendDirection.BEARISH
        elif structure in ("MIXED",):
            t.direction = TrendDirection.TRANSITION
        else:
            t.direction = TrendDirection.NEUTRAL
        return t

    def _market_structure(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[Optional[str], List[str]]:
        """Higher-High/Higher-Low style classification from recent pivots."""
        highs = df["High"].iloc[-lookback:]
        lows = df["Low"].iloc[-lookback:]
        offset = max(len(df) - lookback, 0)
        ph = self._pivot_points(highs)[-2:]
        pl = self._pivot_points(lows, kind="low")[-2:]
        if len(ph) < 2 or len(pl) < 2:
            return None, ["fewer than two clean pivots in lookback"]
        h1, h2 = highs.iloc[ph[0]], highs.iloc[ph[1]]
        l1, l2 = lows.iloc[pl[0]], lows.iloc[pl[1]]
        evidence = [
            f"pivot high {round(h1, 2)} then {round(h2, 2)}",
            f"pivot low {round(l1, 2)} then {round(l2, 2)}",
        ]
        hh = h2 > h1
        hl = l2 > l1
        lh = h2 < h1
        ll = l2 < l1
        if hh and hl:
            return "HH_HL", evidence
        if lh and ll:
            return "LH_LL", evidence
        if (hh and ll) or (lh and hl):
            return "MIXED", evidence
        return None, evidence

    # ---------------- MOMENTUM ----------------
    def _momentum_intelligence(self, df: Optional[pd.DataFrame]) -> MomentumIntelligence:
        m = MomentumIntelligence()
        if df is None or len(df) < 15:
            m.missing.append("insufficient_history_for_momentum")
            return m
        closes = df["Close"]
        m.rsi = self._calculate_rsi(closes)

        if len(closes) >= 11:
            m.roc_10 = self._safe_round(closes.pct_change(10).iloc[-1] * 100)
        if len(closes) >= 6:
            m.roc_5 = self._safe_round(closes.pct_change(5).iloc[-1] * 100)

        if len(closes) >= 35:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            m.macd = self._safe_round(macd_line.iloc[-1])
            m.macd_signal = self._safe_round(signal_line.iloc[-1])
            m.macd_hist = self._safe_round(macd_line.iloc[-1] - signal_line.iloc[-1])
        else:
            m.missing.append("macd_requires_35_bars")

        if len(closes) >= 25:
            rsi_now = m.rsi
            rsi_past = self._calculate_rsi(closes.iloc[:-5])
            if rsi_now is not None and rsi_past is not None:
                if rsi_now - rsi_past >= 3:
                    m.acceleration = "BUILDING"
                elif rsi_now - rsi_past <= -3:
                    m.acceleration = "FADING"

        m.divergence = self._rsi_divergence(df)

        if m.rsi is None:
            m.direction = MomentumLabel.NEUTRAL
            return m
        if m.rsi >= 70:
            base = MomentumLabel.STRONG
        elif m.rsi >= 55:
            base = MomentumLabel.POSITIVE
        elif m.rsi > 45:
            base = MomentumLabel.NEUTRAL
        elif m.rsi >= 30:
            base = MomentumLabel.NEGATIVE
        else:
            base = MomentumLabel.WEAK
        # MACD histogram can shift the label one step toward its side.
        if m.macd_hist is not None:
            hist_positive = m.macd_hist > 0
            order = [MomentumLabel.WEAK, MomentumLabel.NEGATIVE, MomentumLabel.NEUTRAL, MomentumLabel.POSITIVE, MomentumLabel.STRONG]
            idx = order.index(base)
            if hist_positive and idx < len(order) - 1 and base in (MomentumLabel.NEUTRAL, MomentumLabel.NEGATIVE, MomentumLabel.WEAK):
                base = order[idx + 1]
            elif not hist_positive and idx > 0 and base in (MomentumLabel.NEUTRAL, MomentumLabel.POSITIVE, MomentumLabel.STRONG):
                base = order[idx - 1]
        m.direction = base
        return m

    def _rsi_divergence(self, df: pd.DataFrame, lookback: int = 60) -> Optional[str]:
        """Only report divergence when two clean pivot lows exist and price/RSI
        clearly disagree. Never guessed from noisy micro-structure."""
        if len(df) < 40:
            return None
        closes = df["Close"].iloc[-lookback:]
        lows = df["Low"].iloc[-lookback:]
        pivots = self._pivot_points(lows, kind="low")[-2:]
        if len(pivots) < 2 or pivots[1] - pivots[0] < 5:
            return None
        i1, i2 = pivots
        price_ll = lows.iloc[i2] < lows.iloc[i1]
        rsi_i1 = self._calculate_rsi(closes.iloc[:i1 + 1]) if i1 >= 14 else None
        rsi_i2 = self._calculate_rsi(closes.iloc[:i2 + 1]) if i2 >= 14 else None
        if rsi_i1 is None or rsi_i2 is None:
            return None
        if price_ll and rsi_i2 > rsi_i1 + 2:
            return "BULLISH_RSI"
        if not price_ll and rsi_i2 < rsi_i1 - 2:
            return "BEARISH_RSI"
        return None

    # ---------------- VOLATILITY ----------------
    def _volatility_intelligence(self, df: Optional[pd.DataFrame], price: Optional[float]) -> VolatilityIntelligence:
        v = VolatilityIntelligence()
        if df is None or len(df) < 15 or price is None:
            v.missing.append("insufficient_history_for_volatility")
            return v
        highs, lows, closes = df["High"], df["Low"], df["Close"]
        v.atr = self._calculate_atr(highs, lows, closes)
        if v.atr:
            v.atr_pct = round(v.atr / price * 100, 2)

        if len(closes) >= 21:
            mid = closes.rolling(20).mean().iloc[-1]
            std = closes.rolling(20).std().iloc[-1]
            if not pd.isna(mid) and not pd.isna(std):
                v.bb_upper = self._safe_round(mid + 2 * std)
                v.bb_lower = self._safe_round(mid - 2 * std)
                if mid:
                    v.bb_width_pct = round((v.bb_upper - v.bb_lower) / mid * 100, 2)
                widths = ((closes.rolling(20).mean() + 2 * closes.rolling(20).std())
                          - (closes.rolling(20).mean() - 2 * closes.rolling(20).std())) / closes.rolling(20).mean() * 100
                valid_widths = widths.dropna()
                if len(valid_widths) >= 30:
                    current = valid_widths.iloc[-1]
                    v.width_percentile = round((valid_widths < current).sum() / len(valid_widths) * 100, 1)
                    recent_slope = valid_widths.iloc[-1] - valid_widths.iloc[-6] if len(valid_widths) >= 6 else 0
                    v.expanding = bool(v.width_percentile >= 60 and recent_slope > 0)
                    v.contracting = bool(v.width_percentile <= 40 and recent_slope < 0)
        else:
            v.missing.append("bollinger_requires_21_bars")

        if v.atr_pct is None and v.bb_width_pct is None:
            v.regime = VolatilityRegime.UNKNOWN
            return v
        if v.atr_pct is not None and v.atr_pct > 8:
            v.regime = VolatilityRegime.EXTREME
        elif v.width_percentile is not None:
            if v.width_percentile <= 15:
                v.regime = VolatilityRegime.SQUEEZE
            elif v.width_percentile >= 85:
                v.regime = VolatilityRegime.EXPANSION
            else:
                v.regime = VolatilityRegime.NORMAL
        elif v.atr_pct is not None:
            v.regime = VolatilityRegime.NORMAL if v.atr_pct <= 8 else VolatilityRegime.EXTREME
        return v

    # ---------------- VWAP ----------------
    def _vwap_intelligence(self, data: TickerData) -> VwapIntelligence:
        w = VwapIntelligence()
        price = data.current_price

        bars_vwap = None
        bars = data.intraday_bars
        if isinstance(bars, pd.DataFrame) and not bars.empty and {"High", "Low", "Close", "Volume"}.issubset(bars.columns):
            tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
            vol_sum = bars["Volume"].sum()
            if vol_sum and vol_sum > 0:
                bars_vwap = float((tp * bars["Volume"]).sum() / vol_sum)

        session_vwap = data.regular.vwap or data.premarket.vwap

        if bars_vwap is not None:
            w.vwap = round(bars_vwap, 2)
            w.source = "intraday_bars"
        elif session_vwap:
            w.vwap = session_vwap
            w.source = "regular_session" if data.regular.vwap else "premarket"
        else:
            w.status = "UNAVAILABLE"
            w.note = "no intraday bars or session VWAP from provider"
            return w

        if price:
            w.price_vs_vwap_pct = self._pct_diff(price, w.vwap)
            w.status = "ABOVE" if price > w.vwap else "BELOW"

        # Reclaim/rejection need intraday bars; without them we do not guess.
        if bars_vwap is not None and price:
            closes = data.intraday_bars["Close"]
            above_now = price > w.vwap
            was_below = bool((closes.iloc[:-1].tail(30) < w.vwap).any()) if len(closes) > 1 else False
            w.reclaim = bool(above_now and was_below)
            was_above = bool((closes.iloc[:-1].tail(30) > w.vwap).any()) if len(closes) > 1 else False
            w.rejection = bool(not above_now and was_above)
        elif w.note == "":
            w.note = "reclaim/rejection unavailable without intraday bars"
        return w

    # ---------------- VOLUME ----------------
    def _volume_intelligence(self, data: TickerData, df: Optional[pd.DataFrame]) -> VolumeIntelligence:
        v = VolumeIntelligence()
        today_vol = data.regular.volume or data.premarket.volume
        v.volume = int(today_vol) if today_vol else None
        v.dollar_volume = data.dollar_volume
        v.rvol = data.relative_volume

        if df is not None and len(df) >= 5 and "Volume" in df.columns:
            hist_vols = df["Volume"].dropna()
            hist_vols = hist_vols[hist_vols > 0]
            avg20 = data.avg_volume_20d or (hist_vols.tail(20).mean() if len(hist_vols) else None)
            v.average_volume = round(float(avg20), 0) if avg20 else None
            if v.rvol is None and v.average_volume and today_vol:
                v.rvol = round(today_vol / v.average_volume, 2)
            last_hist_vol = hist_vols.iloc[-1] if len(hist_vols) else None
            prior_avg = hist_vols.iloc[-21:-1].mean() if len(hist_vols) >= 21 else None
            if last_hist_vol and prior_avg and prior_avg > 0:
                v.spike_ratio = round(float(last_hist_vol / prior_avg), 2)
            if len(hist_vols) >= 10:
                recent = hist_vols.iloc[-5:].mean()
                prior = hist_vols.iloc[-10:-5].mean()
                if prior and prior > 0:
                    v.acceleration = round(float(recent / prior), 2)
        else:
            v.missing.append("volume_history_insufficient")
            if v.average_volume is None and data.avg_volume_20d:
                v.average_volume = float(data.avg_volume_20d)

        if v.rvol is None:
            v.missing.append("rvol_unavailable")
            v.regime = VolumeRegime.NORMAL
            return v
        r = v.rvol
        if r < 0.7:
            v.regime = VolumeRegime.LOW
        elif r <= 1.2:
            v.regime = VolumeRegime.NORMAL
        elif r <= 2:
            v.regime = VolumeRegime.ELEVATED
        elif r <= 4:
            v.regime = VolumeRegime.HIGH
        else:
            v.regime = VolumeRegime.EXTREME
        return v

    # ---------------- SUPPORT / RESISTANCE ----------------
    LEVEL_DEDUPE_PCT = 0.4

    def _sr_intelligence(self, df: Optional[pd.DataFrame], price: Optional[float], data: TickerData) -> SupportResistanceIntelligence:
        sr = SupportResistanceIntelligence()
        if price is None:
            sr.notes.append("price_unavailable")
            return sr
        candidates: List[Tuple[float, str, str]] = []  # (level, source_evidence, kind_hint)

        if df is not None and len(df) >= 10:
            highs, lows = df["High"], df["Low"]
            window = min(len(df), 90)
            h_slice, l_slice = highs.iloc[-window:], lows.iloc[-window:]
            offset = max(len(df) - window, 0)
            for i in self._pivot_points(h_slice)[-3:]:
                touches = int(((h_slice > h_slice.iloc[i] * 0.997) & (h_slice < h_slice.iloc[i] * 1.003)).sum())
                candidates.append((float(h_slice.iloc[i]), f"daily swing high ({max(touches - 1, 1)} touches)", "swing_high"))
            for i in self._pivot_points(l_slice, kind="low")[-3:]:
                touches = int(((l_slice > l_slice.iloc[i] * 0.997) & (l_slice < l_slice.iloc[i] * 1.003)).sum())
                candidates.append((float(l_slice.iloc[i]), f"daily swing low ({max(touches - 1, 1)} touches)", "swing_low"))
            candidates.append((float(highs.iloc[-10:].max()), "10-day high", "recent_high"))
            candidates.append((float(lows.iloc[-10:].min()), "10-day low", "recent_low"))
        else:
            sr.notes.append("history_insufficient_for_swing_levels")

        if df is not None and len(df) >= 2:
            candidates.append((float(df["High"].iloc[-2]), "previous day high", "pdh"))
            candidates.append((float(df["Low"].iloc[-2]), "previous day low", "pdl"))

        if data.premarket.high:
            candidates.append((data.premarket.high, "premarket high", "pmh"))
        if data.premarket.low:
            candidates.append((data.premarket.low, "premarket low", "pml"))

        levels: List[PriceLevel] = []
        for value, evidence, _kind in candidates:
            level_type = "SUPPORT" if value < price else "RESISTANCE"
            distance_pct = self._pct_diff(value, price)
            strength = 50
            if "touches" in evidence:
                try:
                    strength += min(int(evidence.split("(")[1].split()[0]) - 1, 3) * 10
                except (IndexError, ValueError):
                    pass
            if evidence.startswith("previous day"):
                strength += 10
            if evidence.startswith("10-day"):
                strength += 5
            merged = False
            for existing in levels:
                if abs(existing.price - value) / price * 100 < self.LEVEL_DEDUPE_PCT:
                    existing.evidence += f"; also {evidence}"
                    existing.strength = min(100, existing.strength + 10)
                    merged = True
                    break
            if not merged:
                levels.append(PriceLevel(
                    price=round(value, 2),
                    level_type=level_type,
                    strength=min(strength, 100),
                    distance_pct=distance_pct,
                    evidence=evidence,
                ))
        levels.sort(key=lambda l: l.price)
        sr.levels = levels
        below = [l for l in levels if l.level_type == "SUPPORT"]
        above = [l for l in levels if l.level_type == "RESISTANCE"]
        sr.nearest_support = max(below, key=lambda l: l.price) if below else None
        sr.nearest_resistance = min(above, key=lambda l: l.price) if above else None
        if not below:
            sr.notes.append("no_support_below_price_in_available_data")
        if not above:
            sr.notes.append("no_resistance_above_price_in_available_data")
        return sr

    # ---------------- PRICE ACTION SETUPS ----------------
    SETUP_PRIORITY = [
        "FAILED_BREAKOUT", "BREAKOUT", "HIGHER_HIGH_BREAKOUT", "LOWER_LOW_BREAKDOWN",
        "VWAP_RECLAIM", "VWAP_REJECTION", "PULLBACK", "RESISTANCE_TEST",
        "SUPPORT_TEST", "CONSOLIDATION", "RANGE",
    ]

    def _detect_setups(self, intel: TechnicalIntelligence, data: TickerData, recent_closes: Optional[List[float]] = None) -> List[SetupDetection]:
        s: List[SetupDetection] = []
        price = intel.current_price
        trend, mom = intel.trend, intel.momentum
        vol_int, vwap_int, sr = intel.volatility, intel.vwap, intel.support_resistance

        resistance = sr.nearest_resistance
        support = sr.nearest_support

        # FAILED_BREAKOUT — a recent close cleared resistance, price is back below it
        if (
            price and resistance and recent_closes and len(recent_closes) >= 3
            and any(c > resistance.price for c in recent_closes[-3:])
            and price < resistance.price
        ):
            s.append(SetupDetection(
                name="FAILED_BREAKOUT", direction="BEARISH", detected=True,
                evidence=[f"closes {[round(c, 2) for c in recent_closes[-3:]]} breached {resistance.price} then failed",
                          f"resistance evidence: {resistance.evidence}"],
            ))

        # BREAKOUT family
        # A *current* resistance barely above price = test.
        if price and resistance and 0 < resistance.distance_pct <= 1.5:
            s.append(SetupDetection(
                name="RESISTANCE_TEST", direction="NEUTRAL", detected=True,
                evidence=[f"testing resistance {resistance.price}, distance {resistance.distance_pct}%"],
            ))
        # A former ceiling (swing high / PDH / PMH) now sitting just below price
        # while price holds above it = breakout in force.
        former_ceilings = [
            l for l in sr.levels
            if l.level_type == "SUPPORT" and -3.0 <= l.distance_pct <= 0
            and any(tag in l.evidence for tag in ("swing high", "previous day high", "premarket high"))
        ]
        if price and former_ceilings:
            anchor = max(former_ceilings, key=lambda l: l.price)
            if trend.direction is TrendDirection.BULLISH:
                s.append(SetupDetection(
                    name="HIGHER_HIGH_BREAKOUT", direction="BULLISH", detected=True,
                    evidence=[f"price {price} holds above former resistance {anchor.price} ({anchor.evidence})",
                              f"structure {trend.structure}"],
                ))
            else:
                s.append(SetupDetection(
                    name="BREAKOUT", direction="BULLISH", detected=True,
                    evidence=[f"price {price} cleared former resistance {anchor.price} ({anchor.evidence})"],
                ))

        if support and support.distance_pct >= -1.5 and support.distance_pct <= 0:
            s.append(SetupDetection(
                name="SUPPORT_TEST", direction="NEUTRAL", detected=True,
                evidence=[f"testing support {support.price}, distance {support.distance_pct}%"],
            ))

        if vwap_int.reclaim:
            s.append(SetupDetection(
                name="VWAP_RECLAIM", direction="BULLISH", detected=True,
                evidence=[f"crossed above intraday VWAP {vwap_int.vwap} after trading below"],
            ))
        elif vwap_int.rejection:
            s.append(SetupDetection(
                name="VWAP_REJECTION", direction="BEARISH", detected=True,
                evidence=[f"failed at VWAP {vwap_int.vwap} and fell back below"],
            ))

        # PULLBACK — uptrend pulling into MA20/VWAP
        if (
            price and trend.ma20 and trend.price_vs_ma20_pct is not None
            and trend.direction is TrendDirection.BULLISH
            and 0 <= trend.price_vs_ma20_pct <= 2
        ):
            s.append(SetupDetection(
                name="PULLBACK", direction="BULLISH", detected=True,
                evidence=[f"uptrend with price {trend.price_vs_ma20_pct}% above MA20 {trend.ma20}"],
            ))

        # CONSOLIDATION / RANGE
        if vol_int.regime is VolatilityRegime.SQUEEZE:
            s.append(SetupDetection(
                name="CONSOLIDATION", direction="NEUTRAL", detected=True,
                evidence=[f"Bollinger width percentile {vol_int.width_percentile} (squeeze)"],
            ))
        elif vol_int.regime is VolatilityRegime.UNKNOWN and trend.direction is TrendDirection.NEUTRAL and price:
            s.append(SetupDetection(name="RANGE", direction="NEUTRAL", detected=False,
                                    evidence=["not enough volatility/trend context to define a range"]))
        else:
            s.append(SetupDetection(name="RANGE", direction="NEUTRAL", detected=False))

        # LOWER_LOW_BREAKDOWN
        if trend.structure == "LH_LL" and price and trend.price_vs_ma20_pct is not None and trend.price_vs_ma20_pct < -2:
            s.insert(0, SetupDetection(
                name="LOWER_LOW_BREAKDOWN", direction="BEARISH", detected=True,
                evidence=[f"LH_LL structure; price {trend.price_vs_ma20_pct}% below MA20",
                          f"evidence: {'; '.join(trend.structure_evidence[:2])}"],
            ))

        if not any(x.detected for x in s):
            s.append(SetupDetection(name="NO_SETUP", direction="NEUTRAL", detected=False,
                                    evidence=["no setup conditions met with available data"]))
        ordered = sorted(
            s,
            key=lambda x: self.SETUP_PRIORITY.index(x.name) if x.name in self.SETUP_PRIORITY else len(self.SETUP_PRIORITY),
        )
        return ordered

    # ---------------- TECHNICAL SCORE ----------------
    SCORE_WEIGHTS = {
        "Trend": 0.25,
        "Momentum": 0.25,
        "Volume": 0.15,
        "Volatility": 0.15,
        "Structure_SR": 0.15,
        "VWAP": 0.05,
    }

    def _technical_score(self, intel: TechnicalIntelligence) -> TechnicalScore:
        components: List[ScoreComponent] = []

        # --- Trend ---
        t = intel.trend
        if t.direction is TrendDirection.UNKNOWN:
            components.append(ScoreComponent("Trend", self.SCORE_WEIGHTS["Trend"], False, reason="insufficient_history"))
        else:
            val = 50
            if t.price_vs_ma20_pct is not None and t.price_vs_ma20_pct > 0:
                val += 12
            if t.price_vs_ma50_pct is not None and t.price_vs_ma50_pct > 0:
                val += 12
            if t.price_vs_ma200_pct is not None and t.price_vs_ma200_pct > 0:
                val += 10
            stack_parts = [p for p in (t.ma_alignment or "").split(">")]
            if all(p in stack_parts for p in ("PRICE>MA20",)) and ">MA50" in (t.ma_alignment or "") and t.direction is TrendDirection.BULLISH:
                val += 16
            if all(p in stack_parts for p in ("PRICE<MA20",)) and "<MA50" in (t.ma_alignment or "") and t.direction is TrendDirection.BEARISH:
                val -= 16
            if t.structure == "HH_HL":
                val += 10
            elif t.structure == "LH_LL":
                val -= 10
            components.append(ScoreComponent(
                "Trend", self.SCORE_WEIGHTS["Trend"], True,
                value=max(0, min(100, int(val))),
                reason=f"{t.direction.value}; structure={t.structure}",
            ))

        # --- Momentum ---
        m = intel.momentum
        if m.rsi is None:
            components.append(ScoreComponent("Momentum", self.SCORE_WEIGHTS["Momentum"], False, reason="insufficient_history"))
        else:
            mapped = {
                MomentumLabel.STRONG: 90, MomentumLabel.POSITIVE: 70, MomentumLabel.NEUTRAL: 50,
                MomentumLabel.NEGATIVE: 30, MomentumLabel.WEAK: 15,
            }[m.direction]
            reason = f"RSI {m.rsi}; {m.direction.value}"
            if m.divergence:
                reason += f"; divergence {m.divergence}"
            components.append(ScoreComponent("Momentum", self.SCORE_WEIGHTS["Momentum"], True, value=mapped, reason=reason))

        # --- Volume ---
        vol = intel.volume
        if vol.rvol is None:
            components.append(ScoreComponent("Volume", self.SCORE_WEIGHTS["Volume"], False, reason="rvol_unavailable"))
        else:
            mapped = {
                VolumeRegime.LOW: 35, VolumeRegime.NORMAL: 55, VolumeRegime.ELEVATED: 68,
                VolumeRegime.HIGH: 80, VolumeRegime.EXTREME: 88,
            }[vol.regime]
            components.append(ScoreComponent(
                "Volume", self.SCORE_WEIGHTS["Volume"], True, value=mapped,
                reason=f"RVOL {vol.rvol} ({vol.regime.value})",
            ))

        # --- Volatility ---
        vr = intel.volatility
        if vr.regime is VolatilityRegime.UNKNOWN:
            components.append(ScoreComponent("Volatility", self.SCORE_WEIGHTS["Volatility"], False, reason="insufficient_history"))
        else:
            mapped = {
                VolatilityRegime.SQUEEZE: 58, VolatilityRegime.EXPANSION: 72,
                VolatilityRegime.NORMAL: 65, VolatilityRegime.EXTREME: 40,
            }[vr.regime]
            components.append(ScoreComponent(
                "Volatility", self.SCORE_WEIGHTS["Volatility"], True, value=mapped,
                reason=f"ATR% {vr.atr_pct}; width pctile {vr.width_percentile}; {vr.regime.value}",
            ))

        # --- Structure & S/R ---
        sr = intel.support_resistance
        if intel.current_price is None or not sr.levels:
            components.append(ScoreComponent("Structure_SR", self.SCORE_WEIGHTS["Structure_SR"], False, reason="no_levels_available"))
        else:
            val = 50
            ns, nr = sr.nearest_support, sr.nearest_resistance
            if ns and 0 <= ns.price and -3 <= ns.distance_pct:
                val += 12  # defined risk nearby
            if not ns:
                val -= 10  # air pocket below
            if nr and nr.distance_pct >= 5:
                val += 12  # room to run
            elif nr and nr.distance_pct < 2:
                val -= 8   # capped overhead
            primary = intel.primary_setup
            if primary and primary.direction == "BULLISH" and primary.detected:
                val += 10
            elif primary and primary.direction == "BEARISH" and primary.detected:
                val -= 10
            components.append(ScoreComponent(
                "Structure_SR", self.SCORE_WEIGHTS["Structure_SR"], True,
                value=max(0, min(100, int(val))),
                reason=f"{len(sr.levels)} levels; setup={primary.name if primary else 'NONE'}",
            ))

        # --- VWAP ---
        w = intel.vwap
        if w.status == "UNAVAILABLE":
            components.append(ScoreComponent("VWAP", self.SCORE_WEIGHTS["VWAP"], False, reason="vwap_unavailable"))
        else:
            if w.status == "ABOVE":
                val = 85 if w.reclaim else 70
            else:
                val = 30 if w.rejection else 45
            components.append(ScoreComponent("VWAP", self.SCORE_WEIGHTS["VWAP"], True, value=val, reason=w.status))

        available = [c for c in components if c.available]
        total_weight = sum(c.weight for c in available)
        total = int(round(sum((c.value or 0) * c.weight for c in available) / total_weight)) if total_weight else 0
        return TechnicalScore(total=max(0, min(100, total)), components=components)
