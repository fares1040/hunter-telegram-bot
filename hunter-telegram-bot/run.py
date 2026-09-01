"""Hunter Bot — full orchestration pipeline."""
import asyncio
from typing import List, Optional
import pandas as pd
from config.settings import SETTINGS
from core.data_confidence import DataConfidenceReport, DataQuality
from core.exceptions import ProviderError, DataInsufficientError
from core.memory import SignalMemory
from core.session_clock import SessionClock
from core.realtime_manager import create_realtime_manager, RealtimeManager
from utils.logger import LOGGER
from providers.market_data.yfinance_provider import YFinanceProvider
from providers.market_data.polygon_provider import PolygonProvider
from providers.market_data.polygon_realtime_provider import PolygonRealtimeProvider
from providers.market_data.base_provider import MarketDataProvider
from providers.market_data.yfinance_options_provider import YFinanceOptionsProvider
from providers.market_data.polygon_options_provider import PolygonOptionsProvider
from providers.news.finnhub_provider import FinnhubNewsProvider
from providers.news.yfinance_provider import YFinanceNewsProvider
from providers.news.base_provider import NewsProvider
from engines.news_engine import NewsEngine
from engines.catalyst_engine import CatalystEngine
from engines.candidate_gate import CandidateGate
from engines.market_reaction_engine import MarketReactionEngine
from engines.liquidity_proxy import LiquidityProxyEngine
from engines.technical_engine import TechnicalEngine
from engines.intraday_engine import IntradayEngine
from engines.swing_engine import SwingEngine
from engines.target_engine import TargetEngine
from engines.decision_engine import DecisionEngine
from engines.options_engine import OptionsEngine
from engines.risk_engine import RiskEngine
from engines.trap_engine import TrapEngine
from engines.market_context import MarketContextEngine
from engines.supply_demand_engine import SupplyDemandEngine
from engines.options_flow_engine import OptionsFlowEngine
from engines.strategy_engine import StrategyEngine
from ai.analyzer import AIAnalyzer
from bot.telegram_bot import TelegramNotifier
from bot.commands import TelegramCommandBot
from core.watchlist import WatchlistStore
from core.scheduler import ScanScheduler
from engines.discovery import DiscoveryEngine
from providers.universe.watchlist_provider import WatchlistUniverseProvider
from providers.universe.yfinance_screener_provider import YFinanceScreenerUniverseProvider
from models.signal import HunterSignal, HunterDecision
from models.session import SessionSnapshot
from core.session_clock import MarketSession


class HunterOrchestrator:
    def __init__(self, realtime_manager: Optional[RealtimeManager] = None):
        self.realtime_manager = realtime_manager
        if SETTINGS.has_polygon:
            try:
                self.market_provider: MarketDataProvider = PolygonRealtimeProvider(SETTINGS.polygon_api_key)
            except Exception:
                self.market_provider: MarketDataProvider = PolygonProvider(SETTINGS.polygon_api_key)
        else:
            self.market_provider: MarketDataProvider = YFinanceProvider()
        if SETTINGS.has_polygon:
            try:
                from providers.market_data.polygon_options_realtime_provider import PolygonOptionsRealtimeProvider
                self.options_provider = PolygonOptionsRealtimeProvider(SETTINGS.polygon_api_key)
            except Exception:
                from providers.market_data.polygon_options_provider import PolygonOptionsProvider
                self.options_provider = PolygonOptionsProvider(SETTINGS.polygon_api_key)
        else:
            from providers.market_data.yfinance_options_provider import YFinanceOptionsProvider
            self.options_provider = YFinanceOptionsProvider()
        self.news_providers: List[NewsProvider] = ([FinnhubNewsProvider()] if SETTINGS.has_finnhub else []) + [YFinanceNewsProvider()]
        self.news_engine = NewsEngine(self.news_providers)
        self.catalyst_engine = CatalystEngine()
        self.candidate_gate = CandidateGate()
        self.reaction_engine = MarketReactionEngine()
        self.liquidity_engine = LiquidityProxyEngine()
        self.technical_engine = TechnicalEngine()
        self.intraday_engine = IntradayEngine()
        self.swing_engine = SwingEngine()
        self.options_engine = OptionsEngine()
        self.risk_engine = RiskEngine()
        self.trap_engine = TrapEngine()
        self.decision_engine = DecisionEngine()
        self.ai_analyzer = AIAnalyzer()
        self.target_engine = TargetEngine()
        self.supply_demand_engine = SupplyDemandEngine()
        self.options_flow_engine = OptionsFlowEngine()
        self.strategy_engine = StrategyEngine()
        self.memory = SignalMemory(SETTINGS.memory_db_path)
        self.market_context_engine = MarketContextEngine()
        self.notifier = TelegramNotifier(SETTINGS.telegram_bot_token, SETTINGS.telegram_chat_id)

    async def process_ticker(self, ticker: str) -> HunterSignal:
        try:
            data = await self.market_provider.fetch_ticker(ticker)
        except (ProviderError, DataInsufficientError) as e:
            return self._error_signal(ticker, str(e))
        # Populate latency from intraday bars last timestamp if available
        try:
            if data.intraday_bars is not None and hasattr(data.intraday_bars, "index") and len(data.intraday_bars) > 0:
                last_ts = data.intraday_bars.index[-1]
                if hasattr(last_ts, "to_pydatetime"):
                    last_dt = last_ts.to_pydatetime()
                else:
                    last_dt = last_ts
                from datetime import timezone as _tz
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_tz.utc)
                now_utc = data.timestamp if data.timestamp and data.timestamp.tzinfo else __import__("datetime").datetime.now(_tz.utc)
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=_tz.utc)
                data.data_latency_ms = max(0, int((now_utc - last_dt).total_seconds() * 1000))
        except Exception:
            pass
        # Enhance with realtime data if REST data is incomplete and realtime manager is available
        data = await self._enhance_with_realtime(data, ticker)
        if not data.is_data_sufficient:
            return self._error_signal(ticker, "Insufficient market data")

        gate = self.candidate_gate.evaluate(data)
        if not gate.passed:
            return self._ignore_signal(ticker, "Candidate gate rejected: " + ", ".join(gate.reasons))
        # Realtime quotes/trades (opt-in, never fabricate)
        realtime_quotes: list = []
        realtime_trades: list = []
        if SETTINGS.realtime_enabled and getattr(self.market_provider, "supports_realtime_quotes", False):
            try:
                realtime_quotes = await self.market_provider.fetch_quotes(ticker, limit=5)  # type: ignore
            except Exception:
                realtime_quotes = []
            try:
                realtime_trades = await self.market_provider.fetch_trades(ticker, limit=10)  # type: ignore
            except Exception:
                realtime_trades = []
        history = await self._fetch_history(ticker)
        weekly_history = await self._fetch_weekly_history(ticker)
        monthly_history = await self._fetch_monthly_history(ticker)
        raw_news = await self.news_engine.gather_news(ticker, max_age_hours=24)
        if not raw_news:
            return self._ignore_signal(ticker, "No recent news")
        events = self.news_engine.filter_material_events(self.news_engine.cluster_events(raw_news))
        if not events:
            return self._ignore_signal(ticker, "No material events after filtering")
        catalyst_profile = self.catalyst_engine.enrich(events[0])
        event = await self.ai_analyzer.analyze_event(events[0])
        reaction = self.reaction_engine.analyze(event, data, trades=realtime_trades, realtime_max_age_seconds=SETTINGS.realtime_max_age_seconds)
        liquidity = self.liquidity_engine.analyze(data, quotes=realtime_quotes, trades=realtime_trades, realtime_max_age_seconds=SETTINGS.realtime_max_age_seconds)
        technical = self.technical_engine.analyze(data, history)
        options_snapshot = await self.options_provider.fetch_options(ticker, data.current_price) if SETTINGS.options_enabled else None
        # True flow options trades (opt-in)
        true_option_trades: list = []
        if SETTINGS.options_flow_realtime_enabled and getattr(self.options_provider, "supports_options_realtime", False):
            try:
                true_option_trades = await self.options_provider.fetch_option_trades(ticker, limit=50)  # type: ignore
            except Exception:
                true_option_trades = []
        context = await self.market_context_engine.analyze(ticker)
        bullish = event.sentiment in {"POSITIVE", "VERY_POSITIVE"}
        options = self.options_engine.analyze(options_snapshot, data.current_price, bullish=bullish)
        options_flow = self.options_flow_engine.build(options_snapshot, data.current_price, technical.intelligence, true_flow_trades=true_option_trades, true_flow_max_age=SETTINGS.options_flow_realtime_max_age_seconds)
        risk = self.risk_engine.build_plan(data.current_price, technical, SETTINGS.account_size, SETTINGS.risk_per_trade_pct)
        trap_risk, trap_warnings = self.trap_engine.analyze(data, event, reaction, liquidity, technical)
        if catalyst_profile.is_trap_risk:
            trap_warnings = list(trap_warnings) + [f"CATALYST: {reason}" for reason in catalyst_profile.trap_reasons]
        intraday_intelligence = self.intraday_engine.build(
            data,
            technical=technical,
            daily_history=history,
            catalyst_event=event,
            reaction=reaction,
            liquidity=liquidity,
            risk_plan=risk,
            trap_risk=trap_risk,
            trap_warnings=trap_warnings,
        )
        confidence = self._build_confidence(data, event, technical, options)
        swing_intelligence = self.swing_engine.build(
            data,
            daily_history=history,
            technical_intelligence=technical.intelligence,
            catalyst_event=event,
            catalyst_profile=catalyst_profile,
            intraday_intelligence=intraday_intelligence,
            trap_risk=trap_risk,
            trap_warnings=trap_warnings,
        )
        target_result = None
        swing_entry = getattr(swing_intelligence, "entry", None)
        entry_price = swing_entry.entry_zone_low if swing_entry else None
        invalidation = swing_entry.invalidation_price if swing_entry else None
        if entry_price and invalidation:
            try:
                target_result = self.target_engine.build(
                    swing=swing_intelligence,
                    technical=technical.intelligence,
                    intraday=intraday_intelligence,
                    entry_price=entry_price,
                    invalidation=invalidation,
                )
            except Exception as e:
                LOGGER.warning(f"[Pipeline] Target build failed: {e}")
                target_result = None

        supply_demand_result = self.supply_demand_engine.build(
            data,
            daily_history=history,
            weekly_history=weekly_history,
            monthly_history=monthly_history,
            intraday_intelligence=intraday_intelligence,
        )

        strategy_result = self.strategy_engine.analyze(
            data,
            supply_demand_result,
            swing_intelligence=swing_intelligence,
            technical_intelligence=technical.intelligence,
            intraday_intelligence=intraday_intelligence,
            catalyst_event=event,
            catalyst_profile=catalyst_profile,
            options_intelligence=options_flow,
        )

        signal = self.decision_engine.decide(
            data, event, reaction, liquidity, technical, confidence, options, risk,
            trap_risk, trap_warnings,
            market_context=context,
            technical_intelligence=technical.intelligence,
            intraday_intelligence=intraday_intelligence,
            swing_intelligence=swing_intelligence,
            target_result=target_result,
            supply_demand_result=supply_demand_result,
            options_flow_intelligence=options_flow,
            strategy_result=strategy_result,
        )
        key = f"{ticker}:{event.event_id}"
        if signal.decision == HunterDecision.HUNT_NOW and not self.memory.seen(key):
            self.memory.remember(key, ticker, signal.decision.value, signal.hunter_score)
            try: await self.notifier.send_signal(signal)
            except Exception as e: LOGGER.error(f"[Pipeline] Telegram failed: {e}")
        return signal

    async def _enhance_with_realtime(self, data, ticker: str):
        """Supplement incomplete REST session data with realtime quotes/trades.

        Uses WebSocket realtime data (preferred) or REST polling (fallback) to populate
        missing session snapshots when REST data doesn't include current-session bars.
        """
        if not SETTINGS.realtime_enabled:
            return data

        # Check if session data is already complete
        if data.is_data_sufficient:
            return data

        # Try WebSocket realtime manager first
        realtime_quotes = []
        realtime_trades = []

        if self.realtime_manager and SETTINGS.polygon_ws_enabled:
            max_age = SETTINGS.realtime_max_age_seconds
            realtime_quotes = self.realtime_manager.get_fresh_quotes(ticker, SETTINGS.realtime_max_age_seconds)
            realtime_trades = self.realtime_manager.get_fresh_trades(ticker, SETTINGS.realtime_max_age_seconds)

        # Fallback to REST polling if WebSocket not available or no data
        if SETTINGS.realtime_enabled and getattr(self.market_provider, "supports_realtime_quotes", False) and (not realtime_quotes and not realtime_trades):
            try:
                realtime_quotes = await self.market_provider.fetch_quotes(ticker, limit=5)
            except Exception:
                realtime_quotes = []
            try:
                realtime_trades = await self.market_provider.fetch_trades(ticker, limit=10)
            except Exception:
                realtime_trades = []

        if not realtime_quotes and not realtime_trades:
            return data  # No fresh realtime data available

        # Try to build session snapshots from realtime trades
        from models.session import SessionSnapshot
        from core.session_clock import MarketSession

        # Use realtime trades to build session snapshots
        premarket, regular, after_hours = self.realtime_manager.build_session_snapshots_from_realtime(
            ticker,
            data.current_price or 0,
            data.previous_close or 0
        ) if self.realtime_manager else (None, None, None)

        # Only overwrite incomplete session snapshots with realtime data
        if not data.premarket.is_complete and premarket is not None:
            data.premarket = premarket
        if not data.regular.is_complete and regular is not None:
            data.regular = regular
        if not data.after_hours.is_complete and after_hours is not None:
            data.after_hours = after_hours

        # If we now have current_price from realtime but not in data, update it
        if data.current_price is None:
            latest_quote = self.realtime_manager.get_latest_quote(ticker) if self.realtime_manager else None
            if latest_quote and latest_quote.mid:
                data.current_price = latest_quote.mid

        return data

    async def _fetch_history(self, ticker: str) -> Optional[pd.DataFrame]:
        # Route through provider abstraction when available
        try:
            if hasattr(self.market_provider, "fetch_history"):
                df = await self.market_provider.fetch_history(ticker, period="3mo", interval="1d")  # type: ignore
                if df is not None and not df.empty:
                    return df
        except Exception:
            pass
        try:
            import yfinance as yf
            return await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="3mo", interval="1d"))
        except Exception as e:
            LOGGER.warning(f"[History] Failed: {e}"); return None

    async def _fetch_weekly_history(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            if hasattr(self.market_provider, "fetch_history"):
                df = await self.market_provider.fetch_history(ticker, period="2y", interval="1wk")  # type: ignore
                if df is not None and not df.empty:
                    return df
        except Exception:
            pass
        try:
            import yfinance as yf
            return await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="2y", interval="1wk"))
        except Exception as e:
            LOGGER.warning(f"[WeeklyHistory] Failed: {e}"); return None

    async def _fetch_monthly_history(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            if hasattr(self.market_provider, "fetch_history"):
                df = await self.market_provider.fetch_history(ticker, period="max", interval="1mo")  # type: ignore
                if df is not None and not df.empty:
                    return df
        except Exception:
            pass
        try:
            import yfinance as yf
            return await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="max", interval="1mo"))
        except Exception as e:
            LOGGER.warning(f"[MonthlyHistory] Failed: {e}"); return None

    def _build_confidence(self, data, event, technical, options):
        r = DataConfidenceReport(ticker=data.ticker)
        r.add("current_price", DataQuality.REAL, 2.0); r.add("previous_close", DataQuality.REAL if data.previous_close else DataQuality.MISSING, 1.5)
        r.add("premarket_data", DataQuality.REAL if data.premarket.is_complete else DataQuality.MISSING, 1.5)
        r.add("regular_session_data", DataQuality.REAL if data.regular.is_complete else DataQuality.MISSING, 1.5)
        r.add("news_timestamp", DataQuality.REAL if event.primary_source.published_at else DataQuality.MISSING, 1.0)
        r.add("technical_indicators", DataQuality.PROXY if technical.ma20 else DataQuality.MISSING, 1.0)
        r.add("options_chain", DataQuality.PROXY if options.source != "none" else DataQuality.MISSING, 1.0)
        return r

    def _error_signal(self, ticker, reason):
        return HunterSignal(ticker=ticker, decision=HunterDecision.IGNORE, reasoning=reason, data_insufficient_note=reason)
    def _ignore_signal(self, ticker, reason):
        return HunterSignal(ticker=ticker, decision=HunterDecision.IGNORE, data_confidence=50, reasoning=reason)


async def main():
    SETTINGS.validate_production()

    # Get watchlist symbols for realtime subscription
    watchlist = WatchlistStore(SETTINGS.memory_db_path)
    watchlist_symbols = watchlist.list()

    # Add discovered symbols if discovery enabled
    discovery_engine = None
    if SETTINGS.discovery_enabled:
        universe_providers = [
            WatchlistUniverseProvider(watchlist),
            YFinanceScreenerUniverseProvider(),
        ]
        discovery_engine = DiscoveryEngine(universe_providers)
        try:
            pool = await discovery_engine.refresh()
            for symbol in pool.symbols():
                if symbol not in watchlist_symbols:
                    watchlist_symbols.append(symbol)
        except Exception as e:
            LOGGER.warning(f"[Main] Discovery refresh failed: {e}")

    # Create and start RealtimeManager if enabled
    realtime_manager = None
    if SETTINGS.realtime_enabled and SETTINGS.polygon_ws_enabled and SETTINGS.has_polygon:
        try:
            from providers.market_data.polygon_realtime_provider import PolygonRealtimeProvider
            market_provider = PolygonRealtimeProvider(SETTINGS.polygon_api_key)
            realtime_manager = await create_realtime_manager(market_provider, watchlist_symbols)
            if realtime_manager:
                LOGGER.info(f"[Main] RealtimeManager started for {len(watchlist_symbols)} symbols")
        except Exception as e:
            LOGGER.warning(f"[Main] Failed to start RealtimeManager: {e}")

    orchestrator = HunterOrchestrator(realtime_manager=realtime_manager)
    watchlist = WatchlistStore(SETTINGS.memory_db_path)
    discovery_engine = None
    if SETTINGS.discovery_enabled:
        universe_providers = [
            WatchlistUniverseProvider(watchlist),
            YFinanceScreenerUniverseProvider(),
        ]
        discovery_engine = DiscoveryEngine(universe_providers)
    scheduler = ScanScheduler(orchestrator, watchlist, discovery_engine=discovery_engine)
    command_bot = None
    if SETTINGS.telegram_commands_enabled:
        command_bot = TelegramCommandBot(orchestrator, watchlist, orchestrator.memory, scheduler, discovery_engine)
        try:
            await command_bot.start()
        except Exception as e:
            LOGGER.error(f"[Main] Telegram commands failed to start: {e}")
            command_bot = None
    LOGGER.info(
        f"[Main] Hunter running | session={SessionClock.get_session().value} | "
        f"watchlist={watchlist.list()} | commands={'ON' if command_bot else 'OFF'} | "
        f"discovery={'ON' if discovery_engine else 'OFF'}"
    )
    try:
        await scheduler.run_forever()
    finally:
        if command_bot:
            await command_bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("[Main] Interrupted — goodbye")
