"""Stage 3 True Flow tests."""
from datetime import datetime, timezone, timedelta, date
from models.option_realtime import OptionQuote, OptionTrade, OptionRealtimeBuffer, aggregate_true_flow, parse_polygon_option_quote, parse_polygon_option_trade

def _now(): return datetime.now(timezone.utc)

class TestOptionModels:
    def test_quote_valid(self):
        q = OptionQuote(underlying="AAPL", contract="AAPL250117C00150000", bid=1, ask=1.2, timestamp=_now())
        assert q.is_valid
        assert q.spread == 0.2
    def test_trade_premium(self):
        t = OptionTrade(underlying="AAPL", contract="AAPL250117C00150000", price=1.5, size=10, timestamp=_now(), option_type="CALL")
        assert t.premium == 1500.0
        assert t.aggressor_side is None
    def test_freshness(self):
        q = OptionQuote(underlying="AAPL", contract="C", bid=1, ask=2, timestamp=_now()-timedelta(seconds=60), ingested_at=_now())
        assert q.freshness(30)=="STALE"
        assert q.freshness(100)=="FRESH"
    def test_no_aggressor_fabrication(self):
        t = parse_polygon_option_trade({"sym": "AAPL250117C00150000", "p": 1, "s": 10, "t": int(_now().timestamp()*1000)})
        assert t.aggressor_side is None
    def test_dedup(self):
        buf = OptionRealtimeBuffer(symbol="AAPL", max_events=3)
        t = OptionTrade(underlying="AAPL", contract="C1", price=1, size=10, timestamp=_now(), option_type="CALL")
        assert buf.add_trade(t) is True
        assert buf.add_trade(t) is False
    def test_bounded(self):
        buf = OptionRealtimeBuffer(symbol="AAPL", max_events=2)
        for i in range(5):
            buf.add_trade(OptionTrade(underlying="AAPL", contract=f"C{i}", price=1, size=10, timestamp=_now()+timedelta(seconds=i), option_type="CALL"))
        assert len(buf.trades)==2
    def test_aggregation_no_double_count(self):
        trades = [OptionTrade(underlying="AAPL", contract="C1", price=1, size=10, timestamp=_now(), option_type="CALL"), OptionTrade(underlying="AAPL", contract="P1", price=1, size=5, timestamp=_now(), option_type="PUT")]
        m = aggregate_true_flow(trades, max_age=60)
        assert m.call_volume==10 and m.put_volume==5
        assert m.call_trades==1 and m.put_trades==1
        # Ensure snapshot not mixed - metrics separate
    def test_stale_excluded(self):
        t = OptionTrade(underlying="AAPL", contract="C1", price=1, size=10, timestamp=_now()-timedelta(seconds=100), ingested_at=_now(), option_type="CALL")
        m = aggregate_true_flow([t], max_age=30)
        assert m.total_trades==0 and m.stale_excluded==1
    def test_large_prints(self):
        t = OptionTrade(underlying="AAPL", contract="C1", price=5, size=200, timestamp=_now(), option_type="CALL")
        m = aggregate_true_flow([t], large_size=100)
        assert len(m.large_prints)==1
    def test_concentration(self):
        trades = [OptionTrade(underlying="AAPL", contract="C1", price=1, size=10, timestamp=_now(), option_type="CALL") for _ in range(4)]
        m = aggregate_true_flow(trades)
        assert "C1" in m.repeated_contracts
    def test_engine_integration(self):
        from engines.options_flow_engine import OptionsFlowEngine
        from models.options import OptionsSnapshot, OptionContract
        eng = OptionsFlowEngine()
        snap = OptionsSnapshot(ticker="AAPL", underlying_price=100, contracts=[OptionContract(ticker="AAPL", contract_symbol="C", contract_type="CALL", strike=100, expiration=date.today(), volume=100, open_interest=100, bid=1, ask=1.1)], source="test")
        trades = [OptionTrade(underlying="AAPL", contract="C", price=1, size=10, timestamp=_now(), option_type="CALL")]
        res = eng.build(snap, 100, true_flow_trades=trades, true_flow_max_age=60)
        assert any("TRUE_FLOW" in n for n in res.notes)
    def test_snapshot_fallback(self):
        from engines.options_flow_engine import OptionsFlowEngine
        from models.options import OptionsSnapshot
        eng = OptionsFlowEngine()
        res = eng.build(OptionsSnapshot(ticker="AAPL"), 100, true_flow_trades=[])
        assert res is not None
    def test_settings_flags(self):
        from config.settings import SETTINGS
        assert SETTINGS.options_flow_realtime_enabled is False
