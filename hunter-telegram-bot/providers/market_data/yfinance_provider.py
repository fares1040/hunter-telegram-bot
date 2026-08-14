"""
Hunter Bot — YFinance Provider (FREE TIER / FALLBACK)

LIMITATIONS:
- Delayed data (15-20 min)
- Premarket coverage: limited to 1m intervals, max 7 days
- No true real-time bid/ask
- No options flow, no dark pool
- Best used for: prototyping, fallback only
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
import pytz

from providers.market_data.base_provider import MarketDataProvider
from models.ticker import TickerData
from models.session import SessionSnapshot
from core.session_clock import SessionClock, MarketSession
from core.exceptions import ProviderError, DataInsufficientError
from utils.retry import async_retry
from utils.logger import LOGGER


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    @property
    def is_realtime(self) -> bool:
        return False

    @async_retry(max_retries=2, delay=1.0, exceptions=(Exception,))
    async def fetch_ticker(self, ticker: str, timestamp: Optional[datetime] = None) -> TickerData:
        anchor = timestamp or SessionClock.now()
        anchor_et = SessionClock.localize(anchor)

        loop = asyncio.get_event_loop()
        try:
            stock = await asyncio.to_thread(yf.Ticker, ticker)
        except Exception as e:
            raise ProviderError(f"yfinance init failed: {e}", provider=self.name, retryable=True)

        try:
            hist_1m = await loop.run_in_executor(
                None,
                lambda: stock.history(period="5d", interval="1m", prepost=True)
            )
        except Exception as e:
            raise ProviderError(f"yfinance history failed: {e}", provider=self.name, retryable=True)

        if hist_1m.empty:
            raise DataInsufficientError(f"No price data for {ticker}")

        hist_1m.index = hist_1m.index.tz_convert(SessionClock._tz)

        # Previous close MUST come from the last completed regular session,
        # never from premarket or after-hours candles.
        regular_all = hist_1m[
            (hist_1m.index.time >= pd.Timestamp("09:30").time())
            & (hist_1m.index.time < pd.Timestamp("16:00").time())
        ].copy()
        prior_regular = regular_all[regular_all.index.date < anchor_et.date()]
        if prior_regular.empty:
            raise DataInsufficientError(f"Cannot determine previous regular-session close for {ticker}")
        previous_close = float(prior_regular.groupby(prior_regular.index.date)["Close"].last().iloc[-1])

        # Only expose bars up to the requested anchor. This prevents a historical
        # test or event analysis from accidentally reading future candles.
        intraday_bars = hist_1m[hist_1m.index <= anchor_et].copy()
        if intraday_bars.empty:
            raise DataInsufficientError(f"No intraday bars available at anchor for {ticker}")

        # IMPORTANT: build session snapshots from the anchor-filtered bars.
        # This prevents historical/event analysis from seeing future candles.
        idx = intraday_bars.index
        pre_df = intraday_bars[
            (idx.time >= pd.Timestamp("04:00").time()) &
            (idx.time < pd.Timestamp("09:30").time()) &
            (idx.date == anchor_et.date())
        ]
        reg_df = intraday_bars[
            (idx.time >= pd.Timestamp("09:30").time()) &
            (idx.time < pd.Timestamp("16:00").time()) &
            (idx.date == anchor_et.date())
        ]
        ah_df = intraday_bars[
            (idx.time >= pd.Timestamp("16:00").time()) &
            (idx.time <= pd.Timestamp("20:00").time()) &
            (idx.date == anchor_et.date())
        ]

        premarket = self._build_snapshot(pre_df, MarketSession.PREMARKET)
        regular = self._build_snapshot(reg_df, MarketSession.REGULAR)
        after_hours = self._build_snapshot(ah_df, MarketSession.AFTER_HOURS)

        latest_row = intraday_bars.iloc[-1]
        current_price = float(latest_row["Close"])

        gap_pct = None
        if premarket.open and previous_close and previous_close > 0:
            gap_pct = round(((premarket.open - previous_close) / previous_close) * 100, 2)
        elif current_price and previous_close and previous_close > 0:
            gap_pct = round(((current_price - previous_close) / previous_close) * 100, 2)

        change_pct = None
        if current_price and previous_close and previous_close > 0:
            change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)

        info = {}
        try:
            info = await loop.run_in_executor(None, lambda: stock.info)
        except Exception as e:
            LOGGER.warning(f"[{self.name}] Failed to fetch info for {ticker}: {e}")

        market_cap = info.get("marketCap")
        float_shares = info.get("floatShares") or info.get("sharesOutstanding")
        short_interest = info.get("shortPercentOfFloat")
        avg_volume = info.get("averageVolume")

        if avg_volume is None:
            try:
                hist_20d = await loop.run_in_executor(
                    None, lambda: stock.history(period="20d", interval="1d")
                )
                if not hist_20d.empty:
                    avg_volume = int(hist_20d["Volume"].mean())
            except Exception:
                pass

        return TickerData(
            ticker=ticker,
            timestamp=anchor,
            previous_close=round(previous_close, 2) if previous_close else None,
            premarket=premarket,
            regular=regular,
            after_hours=after_hours,
            current_price=round(current_price, 2) if current_price else None,
            change_percent=change_pct,
            gap_percent=gap_pct,
            market_cap=market_cap,
            float_shares=float_shares,
            short_interest_pct=short_interest,
            avg_volume_20d=avg_volume,
            provider_name=self.name,
            intraday_bars=intraday_bars,
        )

    def _build_snapshot(self, df: pd.DataFrame, session_type: MarketSession) -> SessionSnapshot:
        if df.empty:
            return SessionSnapshot(session_type=session_type)

        vwap = None
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            typical = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = float((typical * df["Volume"]).sum() / df["Volume"].sum())

        return SessionSnapshot(
            session_type=session_type,
            high=float(df["High"].max()),
            low=float(df["Low"].min()),
            open=float(df["Open"].iloc[0]),
            close=float(df["Close"].iloc[-1]),
            volume=int(df["Volume"].sum()),
            vwap=round(vwap, 2) if vwap else None,
            timestamp_start=df.index[0].to_pydatetime(),
            timestamp_end=df.index[-1].to_pydatetime(),
        )

    async def health_check(self) -> bool:
        try:
            await self.fetch_ticker("SPY")
            return True
        except Exception:
            return False
