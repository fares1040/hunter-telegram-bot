"""Real event-time reaction engine with symmetric windows and no fabricated data."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
from models.ticker import TickerData
from models.news import CatalystEvent
from core.session_clock import SessionClock, MarketSession

@dataclass
class ReactionMetrics:
    price_before: Optional[float]=None; price_after: Optional[float]=None
    price_after_5m: Optional[float]=None; price_after_15m: Optional[float]=None; price_after_30m: Optional[float]=None
    price_change_pct: Optional[float]=None; volume_before: Optional[int]=None; volume_after: Optional[int]=None
    volume_after_5m: Optional[int]=None; volume_after_15m: Optional[int]=None; volume_after_30m: Optional[int]=None
    volume_ratio: Optional[float]=None; rvol: Optional[float]=None; gap_pct: Optional[float]=None
    vwap_at_news: Optional[float]=None; price_vs_vwap_pct: Optional[float]=None
    reaction_label: str="UNKNOWN"; reaction_score:int=0; data_sufficient:bool=False
    reaction_timestamp: Optional[datetime]=None  # window end of after_5m (bar-derived, UTC)

class MarketReactionEngine:
    def analyze(self,event: CatalystEvent,ticker_data:TickerData, trades: Optional[list]=None, realtime_max_age_seconds: int=30)->ReactionMetrics:
        m=ReactionMetrics()
        if not ticker_data.is_data_sufficient or not event.primary_source.published_at:
            m.reaction_label="DATA_INSUFFICIENT"; return m
        bars=self._normalize_bars(ticker_data.intraday_bars)
        if bars.empty:
            m.reaction_label="DATA_INSUFFICIENT"; return m
        news=SessionClock.localize(event.primary_source.published_at)
        session=SessionClock.get_session(news)
        if session==MarketSession.CLOSED:
            m.reaction_label="DATA_INSUFFICIENT"; return m
        before=self._window(bars,news-timedelta(minutes=15),news)
        after5=self._window(bars,news,news+timedelta(minutes=5))
        after15=self._window(bars,news,news+timedelta(minutes=15))
        after30=self._window(bars,news,news+timedelta(minutes=30))
        # A valid +5m window requires a meaningful set of post-news bars, not a single candle.
        if before.empty or len(before) < 3 or len(after5) < 3:
            m.reaction_label="DATA_INSUFFICIENT"; return m
        m.price_before=float(before.iloc[-1]["Close"])
        # Equal-duration 5m baseline vs 5m post-news volume.
        m.volume_before=int(before["Volume"].tail(len(after5)).sum()) if len(after5) else int(before["Volume"].sum())
        m.volume_after_5m=int(after5["Volume"].sum())
        m.volume_after_15m=int(after15["Volume"].sum()) if not after15.empty else None
        m.volume_after_30m=int(after30["Volume"].sum()) if not after30.empty else None
        m.price_after_5m=float(after5.iloc[-1]["Close"])
        m.price_after_15m=float(after15.iloc[-1]["Close"]) if not after15.empty else None
        m.price_after_30m=float(after30.iloc[-1]["Close"]) if not after30.empty else None
        m.price_after=m.price_after_15m if m.price_after_15m is not None else m.price_after_5m
        m.volume_after=m.volume_after_15m if m.volume_after_15m is not None else m.volume_after_5m
        m.price_change_pct=round((m.price_after-m.price_before)/m.price_before*100,2) if m.price_before else None
        if m.volume_before and m.volume_after_5m is not None:
            m.volume_ratio=round(m.volume_after_5m/max(1,m.volume_before),2)
        m.rvol=ticker_data.relative_volume; m.gap_pct=ticker_data.gap_percent
        vwap=self._vwap_at_or_before(bars,news)
        if vwap:
            m.vwap_at_news=round(vwap,4); m.price_vs_vwap_pct=round((m.price_after-vwap)/vwap*100,2)
        # Additive realtime trade evidence (fresh only, never overrides bar-based score upward when stale)
        if trades:
            fresh = [t for t in trades if t.freshness(realtime_max_age_seconds)=="FRESH" and t.is_valid]
            if fresh:
                # Realtime trade volume in the 5m window worth a small confirmation (capped)
                pass  # Keep bar score authoritative; freshness already validated. Future stage may refine.

        # Reaction timestamp = end of the 5m reaction window (last bar in after5), UTC normalized
        try:
            if not after5.empty:
                ts = after5.index[-1]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.now(timezone.utc).tzinfo)
                m.reaction_timestamp = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except Exception:
            m.reaction_timestamp = None
        # Need both price reaction and comparable volume to call the reaction strong.
        m.reaction_score=self._score(m)
        m.reaction_label=self._label(m.reaction_score)
        m.data_sufficient=True
        event.price_before=m.price_before; event.price_after=m.price_after; event.volume_before=m.volume_before; event.volume_after=m.volume_after
        return m

    @staticmethod
    def _normalize_bars(bars):
        if bars is None: return pd.DataFrame()
        try: df=bars.copy()
        except Exception: return pd.DataFrame()
        if df.empty or "Close" not in df.columns: return pd.DataFrame()
        if not isinstance(df.index,pd.DatetimeIndex):
            if "timestamp" in df.columns: df=df.set_index("timestamp")
            else: return pd.DataFrame()
        df.index=pd.to_datetime(df.index,utc=True).tz_convert("America/New_York")
        df=df.sort_index()
        if "Volume" not in df.columns: df["Volume"]=0
        return df

    @staticmethod
    def _window(df,start,end):
        s=SessionClock.localize(start); e=SessionClock.localize(end)
        return df[(df.index>=s)&(df.index<e)]

    def _vwap_at_or_before(self,df,target):
        t=SessionClock.localize(target); w=df[df.index<=t]
        if w.empty or w["Volume"].sum()<=0:return None
        tp=(w["High"]+w["Low"]+w["Close"])/3
        return float((tp*w["Volume"]).sum()/w["Volume"].sum())

    def _score(self,m):
        change=m.price_change_pct or 0; vr=m.volume_ratio or 0; score=50
        if change>=8: score+=25
        elif change>=3: score+=15
        elif change>0: score+=5
        elif change<=-5: score-=30
        elif change<0: score-=15
        if vr>=2: score+=20
        elif vr>=1.25: score+=10
        elif vr<0.75: score-=10
        if m.price_vs_vwap_pct is not None:
            if m.price_vs_vwap_pct>=2: score+=10
            elif m.price_vs_vwap_pct<=-2: score-=10
        return max(0,min(100,score))

    @staticmethod
    def _label(score):
        return "STRONG_POSITIVE_REACTION" if score>=80 else "POSITIVE_REACTION" if score>=60 else "WEAK_REACTION" if score>=40 else "NEUTRAL" if score>=20 else "NEGATIVE_REACTION"
