"""Hunter Bot — Technical Engine"""
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

from models.ticker import TickerData
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
