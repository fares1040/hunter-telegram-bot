"""Stage 2 — Real-time Intelligence regression tests."""
from datetime import datetime, timezone, timedelta
from models.market_quote import Quote, Trade, RealtimeBuffer, parse_polygon_quote, parse_polygon_trade
from models.ticker import TickerData

def _now():
    return datetime.now(timezone.utc)

class TestQuoteTradeModels:
    def test_quote_valid_and_spread(self):
        q = Quote(symbol="AAPL", bid=100, ask=100.5, timestamp=_now(), source="polygon_realtime_quote")
        assert q.is_valid
        assert q.spread == 0.5
        assert q.freshness(30) == "FRESH"

    def test_quote_stale(self):
        q = Quote(symbol="AAPL", bid=100, ask=101, timestamp=_now()-timedelta(seconds=60), ingested_at=_now(), source="polygon_realtime_quote")
        assert q.is_stale(30)
        assert q.freshness(30) == "STALE"

    def test_quote_missing_timestamp_unknown(self):
        q = Quote(symbol="AAPL", bid=100, ask=101, source="unknown")
        assert q.freshness() == "UNKNOWN"
        assert q.latency_ms is None

    def test_trade_valid(self):
        t = Trade(symbol="AAPL", price=100, size=100, timestamp=_now(), source="polygon_realtime_trade")
        assert t.is_valid
        assert t.aggressor_side is None  # never fabricated

    def test_no_fabricated_aggressor(self):
        raw = {"sym": "AAPL", "p": 100, "s": 10, "t": int(_now().timestamp()*1000)}
        t = parse_polygon_trade(raw)
        assert t.aggressor_side is None

    def test_timestamp_normalization(self):
        from datetime import datetime
        naive = datetime.now()
        q = Quote(symbol="AAPL", bid=1, ask=2, timestamp=naive)
        assert q.timestamp.tzinfo is not None

class TestRealtimeBuffer:
    def test_bounded_and_dedup(self):
        buf = RealtimeBuffer(symbol="AAPL", max_events=3)
        for i in range(5):
            q = Quote(symbol="AAPL", bid=100+i, ask=101+i, timestamp=_now()+timedelta(seconds=i), source="polygon_realtime_quote")
            buf.add_quote(q)
        assert len(buf.quotes) == 3
        # dedup
        q = buf.quotes[0]
        assert buf.add_quote(q) is False

    def test_ordering(self):
        buf = RealtimeBuffer(symbol="AAPL")
        t2 = Quote(symbol="AAPL", bid=1, ask=2, timestamp=_now()+timedelta(seconds=10), source="x")
        t1 = Quote(symbol="AAPL", bid=1, ask=2, timestamp=_now(), source="x")
        buf.add_quote(t2)
        buf.add_quote(t1)
        assert buf.quotes[0].timestamp <= buf.quotes[1].timestamp

    def test_fresh_filter(self):
        buf = RealtimeBuffer(symbol="AAPL")
        fresh = Quote(symbol="AAPL", bid=1, ask=2, timestamp=_now(), ingested_at=_now(), source="x")
        stale = Quote(symbol="AAPL", bid=1, ask=2, timestamp=_now()-timedelta(seconds=100), ingested_at=_now(), source="x")
        buf.add_quote(fresh)
        buf.add_quote(stale)
        assert len(buf.fresh_quotes(30)) == 1

class TestPolygonParsing:
    def test_parse_quote_malformed_returns_none(self):
        assert parse_polygon_quote({}) is None
        assert parse_polygon_quote({"sym": "AAPL"}) is not None  # valid without bid/ask still creates Quote

    def test_parse_trade_malformed_returns_none(self):
        assert parse_polygon_trade({}) is None

    def test_parse_quote_rest_style(self):
        raw = {"sym": "AAPL", "bid": 100, "ask": 101, "bid_size": 10, "ask_size": 20, "t": int(_now().timestamp()*1000)}
        q = parse_polygon_quote(raw)
        assert q.bid == 100 and q.ask == 101

class TestProviderAbstraction:
    def test_yfinance_does_not_support_realtime(self):
        from providers.market_data.yfinance_provider import YFinanceProvider
        p = YFinanceProvider()
        assert p.supports_realtime_quotes is False

    def test_polygon_realtime_supports(self):
        from providers.market_data.polygon_realtime_provider import PolygonRealtimeProvider
        p = PolygonRealtimeProvider(api_key="test")
        assert p.supports_realtime_quotes is True
        assert p.supports_realtime_trades is True

class TestHistoryRouting:
    def test_yfinance_has_fetch_history(self):
        from providers.market_data.yfinance_provider import YFinanceProvider
        assert hasattr(YFinanceProvider, "fetch_history")

class TestIntelligenceIntegration:
    def test_liquidity_accepts_realtime(self):
        from engines.liquidity_proxy import LiquidityProxyEngine
        eng = LiquidityProxyEngine()
        data = TickerData(ticker="AAPL", timestamp=_now(), current_price=100, previous_close=99)
        q = Quote(symbol="AAPL", bid=100, ask=100.1, timestamp=_now(), ingested_at=_now(), source="polygon_realtime_quote")
        r = eng.analyze(data, quotes=[q], trades=[])
        assert r is not None

    def test_reaction_accepts_realtime(self):
        from engines.market_reaction_engine import MarketReactionEngine
        eng = MarketReactionEngine()
        assert "trades" in eng.analyze.__code__.co_varnames

class TestLatency:
    def test_latency_ms(self):
        ts = _now() - timedelta(milliseconds=150)
        q = Quote(symbol="AAPL", bid=1, ask=2, timestamp=ts, ingested_at=_now(), source="x")
        assert q.latency_ms is not None and q.latency_ms >= 100

    def test_config_defaults(self):
        from config.settings import SETTINGS
        assert SETTINGS.realtime_enabled is False
        assert SETTINGS.polygon_ws_enabled is False
        assert SETTINGS.realtime_max_age_seconds == 30

class TestDecisionEngineAuthority:
    def test_decision_still_only_via_engine(self):
        from engines.decision_engine import DecisionEngine
        eng = DecisionEngine()
        assert hasattr(eng, "decide")
