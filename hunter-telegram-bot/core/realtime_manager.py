"""Hunter Bot — Real-time WebSocket Manager for Polygon streaming."""
import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta

from config.settings import SETTINGS
from providers.market_data.polygon_realtime_provider import PolygonRealtimeProvider
from models.market_quote import Quote, Trade, RealtimeBuffer
from core.session_clock import SessionClock
from utils.logger import LOGGER


class RealtimeManager:
    """Manages Polygon WebSocket connection and per-symbol realtime buffers.
    
    Provides real-time quote/trade data to supplement REST market data.
    Handles connection lifecycle, reconnection, and graceful shutdown.
    """

    def __init__(self, market_provider: PolygonRealtimeProvider, symbols: List[str]):
        self.provider = market_provider
        self.symbols = [s.upper() for s in symbols]
        self._buffers: Dict[str, RealtimeBuffer] = {
            sym: market_provider.create_buffer(sym) for sym in self.symbols
        }
        self._stop_event = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._symbols_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the WebSocket streaming."""
        if self._running:
            return
        
        if not SETTINGS.realtime_enabled or not SETTINGS.polygon_ws_enabled:
            LOGGER.info("[RealtimeManager] Realtime/WS disabled in settings — skipping WebSocket start")
            return
        
        self._stop_event.clear()
        self._ws_task = asyncio.create_task(self._run_stream())
        self._running = True
        LOGGER.info(f"[RealtimeManager] Started WebSocket streaming for {self.symbols}")

    async def stop(self) -> None:
        """Stop the WebSocket streaming gracefully."""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._ws_task is not None:
            try:
                await asyncio.wait_for(self._ws_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                LOGGER.warning(f"[RealtimeManager] Error stopping WS task: {e}")
        self._running = False
        LOGGER.info("[RealtimeManager] WebSocket streaming stopped")

    async def _run_stream(self) -> None:
        """Run the WebSocket stream with reconnection logic."""
        delay = 1.0
        max_delay = 30.0
        
        while not self._stop_event.is_set():
            try:
                await self.provider.stream_quotes_trades(
                    self.symbols, self._buffers, self._stop_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._stop_event.is_set():
                    break
                LOGGER.warning(f"[RealtimeManager] Stream error: {e}, reconnect in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                if self._stop_event.is_set():
                    break

    async def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the streaming subscription."""
        symbol = symbol.upper()
        async with self._symbols_lock:
            if symbol in self._buffers:
                return
            self._buffers[symbol] = self.provider.create_buffer(symbol)
            self.symbols.append(symbol)
        
        if self._running and self._ws_task and not self._ws_task.done():
            # Restart stream to include new symbol
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._ws_task = asyncio.create_task(self._run_stream())

    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the streaming subscription."""
        symbol = symbol.upper()
        async with self._symbols_lock:
            if symbol not in self._buffers:
                return
            del self._buffers[symbol]
            self.symbols = [s for s in self.symbols if s != symbol]
        
        if self._running and self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._ws_task = asyncio.create_task(self._run_stream())

    def get_buffer(self, symbol: str) -> Optional[RealtimeBuffer]:
        """Get the realtime buffer for a symbol."""
        return self._buffers.get(symbol.upper())

    def get_fresh_quotes(self, symbol: str, max_age_seconds: int = 30) -> List[Quote]:
        """Get fresh quotes for a symbol."""
        buf = self.get_buffer(symbol)
        if not buf:
            return []
        return buf.fresh_quotes(max_age_seconds)

    def get_fresh_trades(self, symbol: str, max_age_seconds: int = 30) -> List[Trade]:
        """Get fresh trades for a symbol."""
        buf = self.get_buffer(symbol)
        if not buf:
            return []
        return buf.fresh_trades(max_age_seconds)

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get the latest quote for a symbol."""
        buf = self.get_buffer(symbol)
        if not buf:
            return None
        return buf.latest_quote()

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """Get the latest trade for a symbol."""
        buf = self.get_buffer(symbol)
        if not buf:
            return None
        return buf.latest_trade()

    def build_session_snapshots_from_realtime(self, symbol: str, current_price: float, previous_close: float) -> tuple:
        """Build premarket/regular/after_hours session snapshots from realtime trades/quotes.
        
        Returns (premarket_snapshot, regular_snapshot, after_hours_snapshot).
        Returns (None, None, None) if insufficient realtime data.
        """
        from models.session import SessionSnapshot
        from core.session_clock import MarketSession
        from datetime import datetime
        
        buf = self.get_buffer(symbol)
        if not buf:
            return (None, None, None)
        
        fresh_trades = buf.fresh_trades(30)
        if not fresh_trades:
            return (None, None, None)
        
        # Determine current session to know which snapshot to update
        now = SessionClock.now()
        session = SessionClock.get_session(now)
        
        # For now, build a basic snapshot from trades
        # In production, this would be more sophisticated with time-based aggregation
        prices = [t.price for t in fresh_trades if t.price is not None]
        volumes = [t.size for t in fresh_trades if t.size is not None]
        
        if not prices or not volumes:
            return (None, None, None)
        
        high = max(prices)
        low = min(prices)
        volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else None
        
        # Determine which session we're in and update that snapshot
        premarket = SessionSnapshot(session_type=MarketSession.PREMARKET)
        regular = SessionSnapshot(session_type=MarketSession.REGULAR)
        after_hours = SessionSnapshot(session_type=MarketSession.AFTER_HOURS)
        
        if session == MarketSession.PREMARKET:
            premarket = SessionSnapshot(
                session_type=MarketSession.PREMARKET,
                high=high, low=low, volume=volume, vwap=vwap,
                timestamp_start=fresh_trades[0].timestamp,
                timestamp_end=fresh_trades[-1].timestamp,
            )
        elif session == MarketSession.REGULAR:
            regular = SessionSnapshot(
                session_type=MarketSession.REGULAR,
                high=high, low=low, volume=volume, vwap=vwap,
                timestamp_start=fresh_trades[0].timestamp,
                timestamp_end=fresh_trades[-1].timestamp,
            )
        elif session == MarketSession.AFTER_HOURS:
            after_hours = SessionSnapshot(
                session_type=MarketSession.AFTER_HOURS,
                high=high, low=low, volume=volume, vwap=vwap,
                timestamp_start=fresh_trades[0].timestamp,
                timestamp_end=fresh_trades[-1].timestamp,
            )
        
        return (premarket, regular, after_hours)


async def create_realtime_manager(market_provider, symbols: List[str]) -> Optional['RealtimeManager']:
    """Factory to create and start RealtimeManager if enabled."""
    if not SETTINGS.realtime_enabled or not SETTINGS.polygon_ws_enabled:
        return None
    if not getattr(market_provider, 'supports_realtime_quotes', False):
        return None
    
    manager = RealtimeManager(market_provider, symbols)
    await manager.start()
    return manager
