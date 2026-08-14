"""Polygon.io real-time-capable stock market data provider."""
import aiohttp, pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from providers.market_data.base_provider import MarketDataProvider
from models.ticker import TickerData
from models.session import SessionSnapshot
from core.session_clock import SessionClock, MarketSession
from core.exceptions import ProviderError, DataInsufficientError

class PolygonProvider(MarketDataProvider):
    name="polygon"
    def __init__(self, api_key:str): self.api_key=api_key
    @property
    def is_realtime(self)->bool: return True

    async def _get_json(self,url,params):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(url,params=params) as r:
                if r.status!=200:
                    raise ProviderError(f"Polygon HTTP {r.status}",provider=self.name,retryable=r.status>=500)
                return await r.json()

    async def fetch_ticker(self,ticker:str,timestamp:Optional[datetime]=None)->TickerData:
        anchor=SessionClock.localize(timestamp or SessionClock.now())
        day=anchor.strftime("%Y-%m-%d")
        start=(anchor-timedelta(days=5)).strftime("%Y-%m-%d")
        url=f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{start}/{day}"
        data=await self._get_json(url,{"adjusted":"true","sort":"asc","limit":50000,"apiKey":self.api_key})
        rows=data.get("results") or []
        if not rows: raise DataInsufficientError(f"No Polygon minute data for {ticker}")
        df=pd.DataFrame([{
            "Open":r.get("o"),"High":r.get("h"),"Low":r.get("l"),"Close":r.get("c"),"Volume":r.get("v"),
        } for r in rows],index=pd.to_datetime([r["t"] for r in rows],unit="ms",utc=True).tz_convert(SessionClock._tz))
        df=df[df.index<=anchor]
        if df.empty: raise DataInsufficientError(f"No Polygon bars at anchor for {ticker}")
        regular_all=df[(df.index.time>=datetime.strptime("09:30","%H:%M").time())&(df.index.time<datetime.strptime("16:00","%H:%M").time())]
        prior=regular_all[regular_all.index.date<anchor.date()]
        if prior.empty: raise DataInsufficientError(f"No previous regular close for {ticker}")
        previous_close=float(prior.groupby(prior.index.date)["Close"].last().iloc[-1])
        today=df[df.index.date==anchor.date()]
        pre=today[(today.index.time>=datetime.strptime("04:00","%H:%M").time())&(today.index.time<datetime.strptime("09:30","%H:%M").time())]
        reg=today[(today.index.time>=datetime.strptime("09:30","%H:%M").time())&(today.index.time<datetime.strptime("16:00","%H:%M").time())]
        ah=today[(today.index.time>=datetime.strptime("16:00","%H:%M").time())&(today.index.time<=datetime.strptime("20:00","%H:%M").time())]
        current=float(df.iloc[-1]["Close"])
        gap=((pre.iloc[0]["Open"]-previous_close)/previous_close*100) if not pre.empty else ((current-previous_close)/previous_close*100)
        regvol=int(reg["Volume"].sum()) if not reg.empty else int(pre["Volume"].sum())
        avg= None
        if prior.groupby(prior.index.date)["Volume"].sum().shape[0]>=3:
            avg=int(prior.groupby(prior.index.date)["Volume"].sum().tail(20).mean())
        return TickerData(ticker=ticker.upper(),timestamp=anchor,previous_close=round(previous_close,2),
            premarket=self._snapshot(pre,MarketSession.PREMARKET),regular=self._snapshot(reg,MarketSession.REGULAR),
            after_hours=self._snapshot(ah,MarketSession.AFTER_HOURS),current_price=round(current,2),
            change_percent=round((current-previous_close)/previous_close*100,2),gap_percent=round(gap,2),
            avg_volume_20d=avg,provider_name=self.name,intraday_bars=df)

    def _snapshot(self,df,kind):
        if df.empty:return SessionSnapshot(session_type=kind)
        tp=(df["High"]+df["Low"]+df["Close"])/3; vol=int(df["Volume"].sum())
        vwap=float((tp*df["Volume"]).sum()/df["Volume"].sum()) if vol else None
        return SessionSnapshot(session_type=kind,high=float(df.High.max()),low=float(df.Low.min()),open=float(df.Open.iloc[0]),close=float(df.Close.iloc[-1]),volume=vol,vwap=vwap,timestamp_start=df.index[0].to_pydatetime(),timestamp_end=df.index[-1].to_pydatetime())

    async def health_check(self)->bool:
        try:
            await self.fetch_ticker("SPY"); return True
        except Exception: return False
