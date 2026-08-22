"""Session-aware continuous scan scheduler."""
import asyncio
from typing import Dict, Optional
from config.settings import SETTINGS
from core.session_clock import SessionClock, MarketSession
from core.watchlist import WatchlistStore
from models.signal import HunterDecision
from utils.logger import LOGGER

DEFAULT_INTERVALS = {
    MarketSession.REGULAR: SETTINGS.scan_interval_regular,
    MarketSession.PREMARKET: SETTINGS.scan_interval_extended,
    MarketSession.AFTER_HOURS: SETTINGS.scan_interval_extended,
    MarketSession.CLOSED: SETTINGS.scan_interval_closed,
}


class ScanScheduler:
    """Runs the full pipeline over the watchlist + discovered candidates in a loop,
    pacing scans by market session."""

    def __init__(self, orchestrator, watchlist: WatchlistStore, intervals: Optional[Dict[MarketSession, int]] = None, ticker_pause: float = 1.0, discovery_engine=None, max_scan_batch: int = 15):
        self.orchestrator = orchestrator
        self.watchlist = watchlist
        self.intervals = intervals or dict(DEFAULT_INTERVALS)
        self.ticker_pause = ticker_pause
        self.discovery_engine = discovery_engine
        self.max_scan_batch = max_scan_batch
        self._stop = None
        self._stop_requested = False
        self.last_pass_summary: Optional[str] = None
        self.last_pool_summary: Optional[str] = None

    def stop(self) -> None:
        self._stop_requested = True
        if self._stop is not None:
            self._stop.set()

    async def _scan_targets(self) -> list:
        targets = self.watchlist.list()
        if self.discovery_engine is not None:
            try:
                pool = await self.discovery_engine.refresh()
                self.last_pool_summary = (
                    f"discovered={len(pool.candidates)} raw={pool.raw_count} "
                    f"dupes={pool.duplicate_count} invalid={pool.invalid_count}"
                )
                for symbol in pool.symbols():
                    if symbol not in targets:
                        targets.append(symbol)
            except Exception as e:
                LOGGER.error(f"[Scheduler] discovery refresh failed: {e}")
        return targets[: self.max_scan_batch]

    async def run_forever(self) -> None:
        self._stop = asyncio.Event()
        if self._stop_requested:
            self._stop.set()
        LOGGER.info("[Scheduler] Continuous scanning started")
        while not self._stop.is_set():
            session = SessionClock.get_session()
            tickers = await self._scan_targets()
            if not tickers:
                LOGGER.warning("[Scheduler] No scan targets (empty watchlist, discovery empty); waiting")
            else:
                summary = await self.scan_pass(tickers)
                self.last_pass_summary = summary
                LOGGER.info(f"[Scheduler] Session={session.value} | {summary}")
            interval = self.intervals.get(session, SETTINGS.scan_interval_closed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
        LOGGER.info("[Scheduler] Stopped")

    async def scan_pass(self, tickers) -> str:
        hunted = watched = ignored = errors = 0
        for ticker in tickers:
            if self._stop_requested:
                break
            try:
                signal = await self.orchestrator.process_ticker(ticker)
                if signal.decision == HunterDecision.HUNT_NOW:
                    hunted += 1
                elif signal.decision == HunterDecision.WATCH:
                    watched += 1
                else:
                    ignored += 1
            except Exception as e:
                errors += 1
                LOGGER.error(f"[Scheduler] {ticker} failed: {e}")
            await asyncio.sleep(self.ticker_pause)
        return f"scanned={len(tickers)} hunt={hunted} watch={watched} ignore={ignored} errors={errors}"
