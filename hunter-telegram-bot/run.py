"""Hunter Bot — full orchestration pipeline."""
import asyncio
from typing import List, Optional
import pandas as pd
from config.settings import SETTINGS
from core.data_confidence import DataConfidenceReport, DataQuality
from core.exceptions import ProviderError, DataInsufficientError
from core.memory import SignalMemory
from core.session_clock import SessionClock
from utils.logger import LOGGER
from providers.market_data.yfinance_provider import YFinanceProvider
from providers.market_data.polygon_provider import PolygonProvider
from providers.market_data.base_provider import MarketDataProvider
from providers.market_data.yfinance_options_provider import YFinanceOptionsProvider
from providers.market_data.polygon_options_provider import PolygonOptionsProvider
from providers.news.finnhub_provider import FinnhubNewsProvider
from providers.news.base_provider import NewsProvider
from engines.news_engine import NewsEngine
from engines.candidate_gate import CandidateGate
from engines.market_reaction_engine import MarketReactionEngine
from engines.liquidity_proxy import LiquidityProxyEngine
from engines.technical_engine import TechnicalEngine
from engines.decision_engine import DecisionEngine
from engines.options_engine import OptionsEngine
from engines.risk_engine import RiskEngine
from engines.trap_engine import TrapEngine
from engines.market_context import MarketContextEngine
from ai.analyzer import AIAnalyzer
from bot.telegram_bot import TelegramNotifier
from models.signal import HunterSignal, HunterDecision


class HunterOrchestrator:
    def __init__(self):
        self.market_provider: MarketDataProvider = PolygonProvider(SETTINGS.polygon_api_key) if SETTINGS.has_polygon else YFinanceProvider()
        self.options_provider = PolygonOptionsProvider(SETTINGS.polygon_api_key) if SETTINGS.has_polygon else YFinanceOptionsProvider()
        self.news_providers: List[NewsProvider] = [FinnhubNewsProvider()] if SETTINGS.has_finnhub else []
        self.news_engine = NewsEngine(self.news_providers)
        self.candidate_gate = CandidateGate()
        self.reaction_engine = MarketReactionEngine(); self.liquidity_engine = LiquidityProxyEngine(); self.technical_engine = TechnicalEngine()
        self.options_engine = OptionsEngine(); self.risk_engine = RiskEngine(); self.trap_engine = TrapEngine(); self.decision_engine = DecisionEngine(); self.ai_analyzer = AIAnalyzer()
        self.memory = SignalMemory(SETTINGS.memory_db_path)
        self.market_context_engine = MarketContextEngine()
        self.notifier = TelegramNotifier(SETTINGS.telegram_bot_token, SETTINGS.telegram_chat_id)

    async def process_ticker(self, ticker: str) -> HunterSignal:
        try:
            data = await self.market_provider.fetch_ticker(ticker)
        except (ProviderError, DataInsufficientError) as e:
            return self._error_signal(ticker, str(e))
        if not data.is_data_sufficient:
            return self._error_signal(ticker, "Insufficient market data")
        gate = self.candidate_gate.evaluate(data)
        if not gate.passed:
            return self._ignore_signal(ticker, "Candidate gate rejected: " + ", ".join(gate.reasons))
        history = await self._fetch_history(ticker)
        raw_news = await self.news_engine.gather_news(ticker, max_age_hours=24)
        if not raw_news:
            return self._ignore_signal(ticker, "No recent news")
        events = self.news_engine.filter_material_events(self.news_engine.cluster_events(raw_news))
        if not events:
            return self._ignore_signal(ticker, "No material events after filtering")
        event = await self.ai_analyzer.analyze_event(events[0])
        reaction = self.reaction_engine.analyze(event, data)
        liquidity = self.liquidity_engine.analyze(data)
        technical = self.technical_engine.analyze(data, history)
        options_snapshot = await self.options_provider.fetch_options(ticker, data.current_price) if SETTINGS.options_enabled else None
        context = await self.market_context_engine.analyze(ticker)
        bullish = event.sentiment in {"POSITIVE", "VERY_POSITIVE"}
        options = self.options_engine.analyze(options_snapshot, data.current_price, bullish=bullish)
        risk = self.risk_engine.build_plan(data.current_price, technical, SETTINGS.account_size, SETTINGS.risk_per_trade_pct)
        trap_risk, trap_warnings = self.trap_engine.analyze(data, event, reaction, liquidity, technical)
        confidence = self._build_confidence(data, event, technical, options)
        signal = self.decision_engine.decide(data, event, reaction, liquidity, technical, confidence, options, risk, trap_risk, trap_warnings, market_context=context)
        key = f"{ticker}:{event.event_id}"
        if signal.decision == HunterDecision.HUNT_NOW and not self.memory.seen(key):
            self.memory.remember(key, ticker, signal.decision.value, signal.hunter_score)
            try: await self.notifier.send_signal(signal)
            except Exception as e: LOGGER.error(f"[Pipeline] Telegram failed: {e}")
        return signal

    async def _fetch_history(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            return await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period="3mo", interval="1d"))
        except Exception as e:
            LOGGER.warning(f"[History] Failed: {e}"); return None

    def _build_confidence(self, data, event, technical, options):
        r = DataConfidenceReport(ticker=data.ticker)
        r.add("current_price", DataQuality.REAL, 2.0); r.add("previous_close", DataQuality.REAL if data.previous_close else DataQuality.MISSING, 1.5)
        r.add("premarket_data", DataQuality.REAL if data.premarket.is_complete else DataQuality.MISSING, 1.5)
        r.add("regular_session_data", DataQuality.REAL if data.regular.is_complete else DataQuality.MISSING, 1.5)
        r.add("news_timestamp", DataQuality.REAL if event.primary_source.published_at else DataQuality.MISSING, 1.0)
        r.add("technical_indicators", DataQuality.PROXY if technical.ma20 else DataQuality.MISSING, 1.0)
        r.add("options_chain", DataQuality.PROXY if options.available else DataQuality.MISSING, 1.0)
        return r

    def _error_signal(self, ticker, reason):
        return HunterSignal(ticker=ticker, decision=HunterDecision.IGNORE, reasoning=reason, data_insufficient_note=reason)
    def _ignore_signal(self, ticker, reason):
        return HunterSignal(ticker=ticker, decision=HunterDecision.IGNORE, data_confidence=50, reasoning=reason)


async def main():
    SETTINGS.validate_production()
    orchestrator = HunterOrchestrator()
    for ticker in ["AAPL", "NVDA", "TSLA"]:
        signal = await orchestrator.process_ticker(ticker)
        LOGGER.info(f"{ticker}: {signal.decision.value} | {signal.hunter_score}/100")
        await asyncio.sleep(1)

if __name__ == "__main__": asyncio.run(main())
