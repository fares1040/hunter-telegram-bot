"""Market/sector context helpers. Uses optional yfinance data as context only."""
from dataclasses import dataclass
from typing import Optional
import asyncio
import pandas as pd
import yfinance as yf

from providers.market_data.yfinance_concurrency import get_yfinance_semaphore

@dataclass
class MarketContext:
    regime: str = "UNKNOWN"
    regime_score: int = 50
    sector: str = "UNKNOWN"
    sector_strength: int = 50
    sector_change_pct: Optional[float] = None
    benchmark_change_pct: Optional[float] = None
    notes: list[str] = None
    def __post_init__(self):
        if self.notes is None:
            self.notes = []

class MarketContextEngine:
    BENCHMARKS = {"SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM"}
    SECTOR_ETFS = {
        "TECHNOLOGY": "XLK", "SEMICONDUCTORS": "SOXX", "BIOTECH": "XBI",
        "HEALTHCARE": "XLV", "FINANCIALS": "XLF", "ENERGY": "XLE",
        "COMMUNICATION": "XLC", "CONSUMER_DISCRETIONARY": "XLY",
        "INDUSTRIALS": "XLI", "REAL_ESTATE": "XLRE", "UTILITIES": "XLU",
        "CONSUMER_STAPLES": "XLP", "MATERIALS": "XLB",
    }

    async def analyze(self, ticker: str) -> MarketContext:
        ctx = MarketContext()
        try:
            frames = await asyncio.gather(*[self._daily_change(sym) for sym in self.BENCHMARKS.values()])
            spy, qqq, iwm = frames
            vals = [v for v in (spy, qqq, iwm) if v is not None]
            if not vals:
                return ctx
            avg = sum(vals)/len(vals)
            if avg >= 1.0:
                ctx.regime, ctx.regime_score = "RISK_ON", min(100, int(65 + avg*5))
            elif avg <= -1.0:
                ctx.regime, ctx.regime_score = "RISK_OFF", max(0, int(40 + avg*5))
            else:
                ctx.regime, ctx.regime_score = "NEUTRAL", 50
            ctx.benchmark_change_pct = round(avg, 2)

            sector = await self._infer_sector(ticker)
            ctx.sector = sector
            etf = self.SECTOR_ETFS.get(sector)
            if etf:
                change = await self._daily_change(etf)
                ctx.sector_change_pct = change
                if change is not None:
                    ctx.sector_strength = max(0, min(100, int(50 + change*8)))
            ctx.notes.append(f"Market regime: {ctx.regime}")
        except Exception:
            ctx.notes.append("Market context unavailable")
        return ctx

    async def _daily_change(self, symbol: str):
        async with get_yfinance_semaphore():
            def pull():
                df = yf.Ticker(symbol).history(period="5d", interval="1d")
                if len(df) < 2:
                    return None
                return float((df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100)
            return await asyncio.to_thread(pull)

    async def _infer_sector(self, ticker: str) -> str:
        async with get_yfinance_semaphore():
            def pull():
                try:
                    info = yf.Ticker(ticker).info
                    sector = (info.get("sector") or "").upper()
                    industry = (info.get("industry") or "").upper()
                    if "SEMICONDUCTOR" in industry:
                        return "SEMICONDUCTORS"
                    if "BIOTECH" in industry or "BIOTECH" in sector:
                        return "BIOTECH"
                    mapping = {
                        "TECHNOLOGY":"TECHNOLOGY", "HEALTHCARE":"HEALTHCARE",
                        "FINANCIAL SERVICES":"FINANCIALS", "ENERGY":"ENERGY",
                        "COMMUNICATION SERVICES":"COMMUNICATION",
                        "CONSUMER CYCLICAL":"CONSUMER_DISCRETIONARY",
                        "INDUSTRIALS":"INDUSTRIALS", "REAL ESTATE":"REAL_ESTATE",
                        "UTILITIES":"UTILITIES", "CONSUMER DEFENSIVE":"CONSUMER_STAPLES",
                        "BASIC MATERIALS":"MATERIALS",
                    }
                    return mapping.get(sector, "UNKNOWN")
                except Exception:
                    return "UNKNOWN"
            return await asyncio.to_thread(pull)
