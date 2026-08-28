"""Polygon real-time quotes & trades — REST polling + optional WebSocket.

Polling path is always available (no extra deps). WebSocket is opt-in via
polygon_ws_enabled and uses aiohttp WS client. No values fabricated.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

from providers.market_data.polygon_provider import PolygonProvider
from models.market_quote import Quote, Trade, RealtimeBuffer, parse_polygon_quote, parse_polygon_trade
from core.exceptions import ProviderError

LOGGER = logging.getLogger("hunter")

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"
MAX_WS_RECONNECT_DELAY = 30.0


class PolygonRealtimeProvider(PolygonProvider):
    """Extends PolygonProvider with quote/trade realtime capability."""

    @property
    def supports_realtime_quotes(self) -> bool:
        return True

    @property
    def supports_realtime_trades(self) -> bool:
        return True

    # --- REST polling ---

    async def fetch_quotes(self, ticker: str, limit: int = 1) -> List[Quote]:
        url = f"https://api.polygon.io/v3/quotes/{ticker.upper()}"
        data = await self._get_json_with_retry(url, {"limit": limit, "apiKey": self.api_key})
        results = data.get("results") or []
        out: List[Quote] = []
        now = datetime.now(timezone.utc)
        for raw in results:
            raw["_source"] = "polygon_realtime_quote"
            q = parse_polygon_quote(raw, ingested_at=now)
            if q:
                out.append(q)
        return out

    async def fetch_trades(self, ticker: str, limit: int = 1) -> List[Trade]:
        url = f"https://api.polygon.io/v3/trades/{ticker.upper()}"
        data = await self._get_json_with_retry(url, {"limit": limit, "apiKey": self.api_key})
        results = data.get("results") or []
        out: List[Trade] = []
        now = datetime.now(timezone.utc)
        for raw in results:
            raw["_source"] = "polygon_realtime_trade"
            t = parse_polygon_trade(raw, ingested_at=now)
            if t:
                out.append(t)
        return out

    async def fetch_history(self, ticker: str, period: str = "3mo", interval: str = "1d"):
        """History via Polygon aggregates (respects provider abstraction)."""
        import pandas as pd
        from datetime import timedelta
        from core.session_clock import SessionClock
        # Map period/interval to polygon range
        period_days = {"1mo": 30, "3mo": 90, "2y": 730, "max": 3650}.get(period, 90)
        anchor = SessionClock.now()
        start = (anchor - timedelta(days=period_days)).strftime("%Y-%m-%d")
        day = anchor.strftime("%Y-%m-%d")
        # Map interval
        mult = 1
        span = "day"
        if interval == "1wk":
            mult = 1
            span = "week"
        elif interval == "1mo":
            mult = 1
            span = "month"
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{mult}/{span}/{start}/{day}"
        data = await self._get_json_with_retry(url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key})
        rows = data.get("results") or []
        if not rows:
            return None
        df = pd.DataFrame([{
            "Open": r.get("o"), "High": r.get("h"), "Low": r.get("l"),
            "Close": r.get("c"), "Volume": r.get("v"),
        } for r in rows], index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert(SessionClock._tz))
        return df.sort_index()

    # --- WebSocket (opt-in) ---

    def create_buffer(self, symbol: str, max_events: int = 1000) -> RealtimeBuffer:
        return RealtimeBuffer(symbol=symbol.upper(), max_events=max_events)

    async def stream_quotes_trades(self, symbols: List[str], buffer_map: Dict[str, RealtimeBuffer], stop_event: asyncio.Event):
        """Long-running WS subscription. Reconnects with backoff until stop_event is set."""
        import aiohttp
        delay = 1.0
        symbols_upper = [s.upper() for s in symbols]
        while not stop_event.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(POLYGON_WS_URL) as ws:
                        await ws.send_str(json.dumps({"action": "auth", "params": self.api_key}))
                        # Wait for auth response
                        msg = await asyncio.wait_for(ws.receive(), timeout=10)
                        # Subscribe
                        await ws.send_str(json.dumps({"action": "subscribe", "params": ",".join([f"Q.{s}" for s in symbols_upper] + [f"T.{s}" for s in symbols_upper])}))
                        delay = 1.0
                        LOGGER.info(f"[PolygonWS] Subscribed {symbols_upper}")
                        async for msg in ws:
                            if stop_event.is_set():
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                    if not isinstance(payload, list):
                                        payload = [payload]
                                    now = datetime.now(timezone.utc)
                                    for ev in payload:
                                        ev_type = ev.get("ev")
                                        sym = (ev.get("sym") or ev.get("T") or "").upper()
                                        buf = buffer_map.get(sym)
                                        if not buf:
                                            continue
                                        if ev_type == "Q":
                                            q = parse_polygon_quote(ev, ingested_at=now)
                                            if q:
                                                buf.add_quote(q)
                                        elif ev_type == "T":
                                            t = parse_polygon_trade(ev, ingested_at=now)
                                            if t:
                                                buf.add_trade(t)
                                except Exception as e:
                                    LOGGER.warning(f"[PolygonWS] parse error: {e}")
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.warning(f"[PolygonWS] disconnected: {e}, reconnect in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_WS_RECONNECT_DELAY)
