"""Tests for watchlist persistence, scan scheduler, memory stats, and Telegram command interface."""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from models.signal import HunterSignal, HunterDecision
from core.watchlist import WatchlistStore, normalize_ticker
from core.scheduler import ScanScheduler
from core.memory import SignalMemory
from core.session_clock import MarketSession
from bot.commands import TelegramCommandBot
from config.settings import Settings


VALID_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_AUTHORIZED_CHAT_ID",
        "SCAN_INTERVAL_REGULAR", "SCAN_INTERVAL_EXTENDED", "SCAN_INTERVAL_CLOSED",
        "OPENAI_API_KEY", "MEMORY_DB_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)


def _run(coro):
    return asyncio.run(coro)


def _make_update(chat_id=111):
    update = MagicMock()
    chat = MagicMock()
    chat.id = chat_id
    chat.send_message = AsyncMock()
    update.effective_chat = chat
    return update, chat


class _FakeOrchestrator:
    def __init__(self, signal=None, exc=None):
        self.signal = signal
        self.exc = exc
        self.news_providers = []
        self.market_provider = MagicMock()
        self.process_ticker = AsyncMock(side_effect=self._call)

    async def _call(self, *a, **k):
        if self.exc:
            raise self.exc
        return self.signal


class TestNormalizeTicker:
    def test_uppercases_and_strips_dollar(self):
        assert normalize_ticker("$aapl") == "AAPL"
        assert normalize_ticker(" nvda ") == "NVDA"

    def test_valid_plain_symbol(self):
        assert normalize_ticker("TSLA") == "TSLA"

    def test_rejects_digits(self):
        with pytest.raises(ValueError):
            normalize_ticker("AA1L")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            normalize_ticker("ABCDEFG")

    def test_rejects_empty_and_none(self):
        with pytest.raises(ValueError):
            normalize_ticker("")
        with pytest.raises(ValueError):
            normalize_ticker(None)

    def test_rejects_special_characters(self):
        with pytest.raises(ValueError):
            normalize_ticker("BRK.B")


class TestWatchlistStore:
    def test_seeds_default_watchlist(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"))
        assert wl.list() == ["AAPL", "NVDA", "TSLA"]

    def test_add_returns_new_symbol_once(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        assert wl.add("amd") == "AMD"
        assert wl.add("AMD") is None

    def test_remove_returns_removed_or_none(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        wl.add("MSFT")
        assert wl.remove("msft") == "MSFT"
        assert wl.remove("MSFT") is None

    def test_contains(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        wl.add("PLTR")
        assert wl.contains("$pltr")
        assert not wl.contains("ZZZZZ")
        assert not wl.contains("BAD!")

    def test_list_sorted_and_persistent(self, tmp_path):
        path = str(tmp_path / "wl.sqlite3")
        wl = WatchlistStore(path, seed_defaults=False)
        wl.add("TSLA"); wl.add("AAPL"); wl.add("NVDA")
        wl2 = WatchlistStore(path, seed_defaults=False)
        assert wl2.list() == ["AAPL", "NVDA", "TSLA"]


class TestScanSchedulerPass:
    def _signal(self, decision):
        return HunterSignal(ticker="X", decision=decision)

    def test_counts_decisions(self, tmp_path):
        signals = [self._signal(HunterDecision.HUNT_NOW), self._signal(HunterDecision.WATCH), self._signal(HunterDecision.IGNORE)]
        orch = _FakeOrchestrator()
        results = iter(signals)

        async def side_effect(*a, **k):
            return next(results)
        orch.process_ticker = AsyncMock(side_effect=side_effect)

        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        sched = ScanScheduler(orch, wl, ticker_pause=0)
        summary = _run(sched.scan_pass(["A", "B", "C"]))
        assert summary == "scanned=3 hunt=1 watch=1 ignore=1 errors=0"

    def test_errors_are_counted_not_raised(self, tmp_path):
        orch = _FakeOrchestrator(exc=RuntimeError("boom"))
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        sched = ScanScheduler(orch, wl, ticker_pause=0)
        summary = _run(sched.scan_pass(["A"]))
        assert summary == "scanned=1 hunt=0 watch=0 ignore=0 errors=1"

    def test_stop_breaks_loop_mid_pass(self, tmp_path):
        orch = _FakeOrchestrator(signal=self._signal(HunterDecision.IGNORE))
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        sched = ScanScheduler(orch, wl, ticker_pause=0)
        sched.stop()

        async def scenario():
            task = asyncio.create_task(sched.scan_pass(["A", "B", "C"]))
            await asyncio.sleep(0)
            await asyncio.wait_for(task, timeout=2)
        _run(scenario())

    def test_run_forever_exits_after_stop(self, tmp_path):
        orch = _FakeOrchestrator(signal=self._signal(HunterDecision.IGNORE))
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        wl.add("AAPL")
        intervals = {s: 0.05 for s in MarketSession}
        sched = ScanScheduler(orch, wl, intervals=intervals, ticker_pause=0)

        async def scenario():
            task = asyncio.create_task(sched.run_forever())
            await asyncio.sleep(0.25)
            sched.stop()
            await asyncio.wait_for(task, timeout=3)
        _run(scenario())
        assert orch.process_ticker.await_count >= 1
        assert sched.last_pass_summary and "hunt=" in sched.last_pass_summary


class TestSignalMemoryStats:
    def test_alert_count_and_recent_ordering(self, tmp_path):
        mem = SignalMemory(str(tmp_path / "m.sqlite3"))
        assert mem.alert_count() == 0
        mem.remember("K1", "AAPL", "HUNT_NOW", 85)
        mem.remember("K2", "NVDA", "WATCH", 60)
        assert mem.alert_count() == 2
        recent = mem.recent_alerts(limit=5)
        assert len(recent) == 2
        assert recent[0]["ticker"] in {"AAPL", "NVDA"}
        assert all(set(a) == {"ticker", "decision", "score", "created_at"} for a in recent)


class TestAuthorizedChatIds:
    def test_defaults_to_alert_chat_only(self):
        s = Settings(_env_file=None, telegram_bot_token=VALID_TOKEN, telegram_chat_id="111")
        assert s.authorized_chat_ids == {"111"}

    def test_extra_ids_parsed(self):
        s = Settings(_env_file=None, telegram_bot_token=VALID_TOKEN, telegram_chat_id="111", telegram_authorized_chat_id="222, 333")
        assert s.authorized_chat_ids == {"111", "222", "333"}


class TestCommandBotAuthorization:
    def _bot(self, tmp_path):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        mem = SignalMemory(str(tmp_path / "m.sqlite3"))
        bot = TelegramCommandBot(_FakeOrchestrator(), wl, mem)
        bot.authorized_ids = {"111"}
        return bot, wl, mem

    def test_unauthorized_chat_gets_no_reply(self, tmp_path):
        bot, _, _ = self._bot(tmp_path)
        update, chat = _make_update(chat_id=999)
        ctx = MagicMock(args=["AAPL"])
        _run(bot.cmd_add(update, ctx))
        chat.send_message.assert_not_awaited()

    def test_authorized_chat_gets_reply(self, tmp_path):
        bot, wl, _ = self._bot(tmp_path)
        update, chat = _make_update(chat_id=111)
        ctx = MagicMock(args=["AAPL"])
        _run(bot.cmd_add(update, ctx))
        chat.send_message.assert_awaited_once()
        assert wl.contains("AAPL")


class TestCommandHandlers:
    def _bot(self, tmp_path, orchestrator=None):
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        mem = SignalMemory(str(tmp_path / "m.sqlite3"))
        orch = orchestrator or _FakeOrchestrator(
            signal=HunterSignal(ticker="AAPL", decision=HunterDecision.WATCH, hunter_score=55, current_price=12.34, catalyst_type="EARNINGS_BEAT")
        )
        bot = TelegramCommandBot(orch, wl, mem)
        bot.authorized_ids = {"111"}
        return bot, wl, mem, orch

    def test_add_without_args_shows_usage(self, tmp_path):
        bot, wl, _, _ = self._bot(tmp_path)
        update, chat = _make_update()
        ctx = MagicMock(args=[])
        _run(bot.cmd_add(update, ctx))
        text = chat.send_message.await_args.args[0]
        assert "/add AAPL" in text
        assert wl.list() == []

    def test_add_invalid_ticker_warns(self, tmp_path):
        bot, wl, _, _ = self._bot(tmp_path)
        update, chat = _make_update()
        ctx = MagicMock(args=["BAD!"])
        _run(bot.cmd_add(update, ctx))
        text = chat.send_message.await_args.args[0]
        assert text.startswith("⚠️")
        assert wl.list() == []

    def test_remove_existing_and_missing(self, tmp_path):
        bot, wl, _, _ = self._bot(tmp_path)
        wl.add("MSFT")
        update, chat = _make_update()
        _run(bot.cmd_remove(update, MagicMock(args=["msft"])))
        assert not wl.contains("MSFT")
        _run(bot.cmd_remove(update, MagicMock(args=["MSFT"])))
        texts = [c.args[0] for c in chat.send_message.await_args_list]
        assert any("removed" in t for t in texts)
        assert any("was not" in t for t in texts)

    def test_watchlist_lists_tickers(self, tmp_path):
        bot, wl, _, _ = self._bot(tmp_path)
        wl.add("AAPL"); wl.add("NVDA")
        update, chat = _make_update()
        _run(bot.cmd_watchlist(update, MagicMock(args=[])))
        text = chat.send_message.await_args.args[0]
        assert "AAPL" in text and "NVDA" in text and "Total: <b>2</b>" in text

    def test_scan_single_ticker_calls_pipeline(self, tmp_path):
        bot, wl, _, orch = self._bot(tmp_path)
        update, chat = _make_update()
        _run(bot.cmd_scan(update, MagicMock(args=["$aapl"])))
        orch.process_ticker.assert_awaited_once_with("AAPL")
        text = chat.send_message.await_args_list[-1].args[0]
        assert "$AAPL" in text and "WATCH" in text and "12.34" in text

    def test_scan_watchlist_uses_scheduler_pass(self, tmp_path):
        orch = _FakeOrchestrator(signal=HunterSignal(ticker="AAPL", decision=HunterDecision.IGNORE))
        wl = WatchlistStore(str(tmp_path / "wl.sqlite3"), seed_defaults=False)
        wl.add("AAPL")
        mem = SignalMemory(str(tmp_path / "m.sqlite3"))
        bot = TelegramCommandBot(orch, wl, mem, scheduler=ScanScheduler(orch, wl, ticker_pause=0))
        bot.authorized_ids = {"111"}
        update, chat = _make_update()
        _run(bot.cmd_scan(update, MagicMock(args=[])))
        orch.process_ticker.assert_awaited_once_with("AAPL")
        text = chat.send_message.await_args_list[-1].args[0]
        assert "Manual pass complete" in text

    def test_status_contains_session_and_providers(self, tmp_path):
        bot, _, _, _ = self._bot(tmp_path)
        update, chat = _make_update()
        _run(bot.cmd_status(update, MagicMock(args=[])))
        text = chat.send_message.await_args.args[0]
        assert "Session:" in text
        assert any(s.value in text for s in MarketSession)
        assert "News providers: 0" in text

    def test_stats_shows_recorded_alerts(self, tmp_path):
        bot, _, mem, _ = self._bot(tmp_path)
        mem.remember("K1", "TSLA", "HUNT_NOW", 88)
        update, chat = _make_update()
        _run(bot.cmd_stats(update, MagicMock(args=[])))
        text = chat.send_message.await_args.args[0]
        assert "Total alerts: <b>1</b>" in text and "TSLA" in text

    def test_format_signal_includes_key_fields(self, tmp_path):
        bot, _, _, _ = self._bot(tmp_path)
        sig = HunterSignal(
            ticker="NVDA", decision=HunterDecision.HUNT_NOW, hunter_score=82,
            current_price=101.5, catalyst_type="FDA_APPROVAL", sentiment="POSITIVE",
            reaction_status="STRONG_POSITIVE_REACTION", liquidity_status="STRONG_LIQUIDITY_PROXY",
            technical_setup="BREAKOUT", options_bias="CALL_SKEW", trap_risk=5,
        )
        text = TelegramCommandBot._format_signal(sig)
        assert "🚀" in text and "HUNT_NOW" in text and "82/100" in text
        assert "FDA_APPROVAL" in text and "BREAKOUT" in text
