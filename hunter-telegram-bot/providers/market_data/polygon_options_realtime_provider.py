"""Polygon options realtime — REST trades/quotes + optional WS O.*"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict

from providers.market_data.polygon_options_provider import PolygonOptionsProvider
from models.option_realtime import OptionQuote, OptionTrade, parse_polygon_option_quote, parse_polygon_option_trade, OptionRealtimeBuffer
from core.exceptions import ProviderError

LOGGER = logging.getLogger("hunter")
WS_URL = "wss://socket.polygon.io/options"
MAX_RECONNECT = 30.0

class PolygonOptionsRealtimeProvider(PolygonOptionsProvider):
    async def fetch_option_quotes(self, ticker: str, limit: int = 20) -> List[OptionQuote]:
        url = f"https://api.polygon.io/v3/quotes/{ticker}"
        # Use parent retry logic via aiohttp direct (reuse pattern)
        import aiohttp
        params = {"limit": limit, "apiKey": self.api_key}
        now = datetime.now(timezone.utc)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as s:
                async with s.get(url, params=params) as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
        except Exception as e:
            LOGGER.warning(f"[PolyOpt] quotes failed {ticker}: {e}")
            return []
        out = []
        for raw in data.get("results") or []:
            raw["_source"] = "polygon_options_realtime_quote"
            q = parse_polygon_option_quote(raw, ingested_at=now)
            if q:
                out.append(q)
        return out

    async def fetch_option_trades(self, ticker: str, limit: int = 50) -> List[OptionTrade]:
        url = f"https://api.polygon.io/v3/trades/{ticker}"
        import aiohttp
        params = {"limit": limit, "apiKey": self.api_key}
        now = datetime.now(timezone.utc)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as s:
                async with s.get(url, params=params) as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
        except Exception as e:
            LOGGER.warning(f"[PolyOpt] trades failed {ticker}: {e}")
            return []
        out = []
        for raw in data.get("results") or []:
            raw["_source"] = "polygon_options_realtime_trade"
            t = parse_polygon_option_trade(raw, ingested_at=now)
            if t:
                out.append(t)
        return out

    async def stream_option_trades(self, tickers: List[str], buffer_map: Dict[str, OptionRealtimeBuffer], stop_event: asyncio.Event):
        import aiohttp
        delay = 1.0
        up = [t.upper() for t in tickers]
        while not stop_event.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL) as ws:
                        await ws.send_str(json.dumps({"action": "auth", "params": self.api_key}))
                        await asyncio.wait_for(ws.receive(), timeout=10)
                        await ws.send_str(json.dumps({"action": "subscribe", "params": ",".join([f"T.{s}" for s in up] + [f"Q.{s}" for s in up])}))
                        LOGGER.info(f"[PolyOptWS] subscribed {up}")
                        delay = 1.0
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
                                        sym = (ev.get("sym") or ev.get("T") or "").upper()
                                        # underlying from contract
                                        import re
                                        m = re.match(r"^([A-Z]+)", sym)
                                        und = m.group(1) if m else sym
                                        buf = buffer_map.get(und)
                                        if not buf:
                                            continue
                                        ev_t = ev.get("ev")
                                        if ev_t == "Q":
                                            q = parse_polygon_option_quote(ev, ingested_at=now)
                                            if q:
                                                buf.add_quote(q)
                                        elif ev_t == "T":
                                            t = parse_polygon_option_trade(ev, ingested_at=now)
                                            if t:
                                                buf.add_trade(t)
                                except Exception as e:
                                    LOGGER.warning(f"[PolyOptWS] parse {e}")
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.warning(f"[PolyOptWS] disconnected {e} reconnect {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay*2, MAX_RECONNECT)
