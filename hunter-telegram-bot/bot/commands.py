"""Interactive Telegram command interface for Hunter Bot."""
import functools
from typing import Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config.settings import SETTINGS
from core.session_clock import SessionClock, MarketSession
from models.signal import HunterDecision
from utils.logger import LOGGER

MAX_MANUAL_SCAN_TICKERS = 10

HELP_TEXT = (
    "🤖 <b>HUNTER BOT — COMMANDS</b>\n"
    "\n"
    "/discover — Scan the market for fresh candidates\n"
    "/scan [TICKER] — Scan one ticker (or the whole watchlist)\n"
    "/add TICKER — Add ticker to watchlist\n"
    "/remove TICKER — Remove ticker from watchlist\n"
    "/watchlist — Show watchlist\n"
    "/status — Market session, providers, last scan\n"
    "/stats — Alert memory statistics\n"
    "/help — This message"
)


def _authorized(func):
    @functools.wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id) if update and update.effective_chat else ""
        if chat_id not in self.authorized_ids:
            LOGGER.warning(f"[Commands] Rejected command from unauthorized chat {chat_id}")
            return
        try:
            return await func(self, update, context)
        except ValueError as e:
            await self._reply(update, f"⚠️ {e}")
        except Exception as e:
            LOGGER.error(f"[Commands] /{func.__name__} failed: {e}")
            await self._reply(update, "❌ Command failed. Check logs.")
    return wrapper


class TelegramCommandBot:
    """Registers slash commands on a python-telegram-bot Application."""

    def __init__(self, orchestrator, watchlist, memory, scheduler=None, discovery_engine=None):
        self.orchestrator = orchestrator
        self.watchlist = watchlist
        self.memory = memory
        self.scheduler = scheduler
        self.discovery_engine = discovery_engine
        self.authorized_ids = SETTINGS.authorized_chat_ids
        self.application: Optional[Application] = None

    def build_application(self) -> Application:
        app = Application.builder().token(SETTINGS.telegram_bot_token).build()
        routes = {
            "start": self.cmd_start,
            "help": self.cmd_help,
            "scan": self.cmd_scan,
            "add": self.cmd_add,
            "remove": self.cmd_remove,
            "watchlist": self.cmd_watchlist,
            "status": self.cmd_status,
            "stats": self.cmd_stats,
            "discover": self.cmd_discover,
        }
        for name, cb in routes.items():
            app.add_handler(CommandHandler(name, cb))
        self.application = app
        return app

    async def start(self) -> None:
        if self.application is None:
            self.build_application()
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)
        LOGGER.info("[Commands] Telegram polling started")

    async def shutdown(self) -> None:
        if self.application is None:
            return
        try:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        except Exception as e:
            LOGGER.warning(f"[Commands] Shutdown issue: {e}")

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        if update and update.effective_chat:
            await update.effective_chat.send_message(text, parse_mode="HTML", disable_web_page_preview=True)

    @_authorized
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._reply(update, HELP_TEXT)

    @_authorized
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._reply(update, HELP_TEXT)

    @_authorized
    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = [a.upper().lstrip("$") for a in (context.args or [])]
        if args:
            from core.watchlist import normalize_ticker
            ticker = normalize_ticker(args[0])
            await self._reply(update, f"🔍 Scanning <b>${ticker}</b>…")
            signal = await self.orchestrator.process_ticker(ticker)
            await self._reply(update, self._format_signal(signal))
            return
        tickers = self.watchlist.list()[:MAX_MANUAL_SCAN_TICKERS]
        if not tickers:
            await self._reply(update, "Watchlist is empty. Use /add TICKER first.")
            return
        await self._reply(update, f"🔍 Scanning <b>{len(tickers)}</b> tickers…")
        summary = await self.scheduler.scan_pass(tickers) if self.scheduler else "done"
        extra = f"\n{summary}" if self.scheduler else ""
        tail = "" if len(self.watchlist.list()) <= MAX_MANUAL_SCAN_TICKERS else "\n<i>(capped at 10; alerts are sent automatically when found)</i>"
        await self._reply(update, f"✅ Manual pass complete{extra}{tail}")

    @_authorized
    async def cmd_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self._reply(update, "Usage: <code>/add AAPL</code>")
            return
        added = self.watchlist.add(context.args[0])
        if added:
            await self._reply(update, f"✅ <b>${added}</b> added to watchlist ({len(self.watchlist.list())} total)")
        else:
            await self._reply(update, f"ℹ️ ${context.args[0].upper()} is already in the watchlist")

    @_authorized
    async def cmd_remove(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self._reply(update, "Usage: <code>/remove AAPL</code>")
            return
        removed = self.watchlist.remove(context.args[0])
        if removed:
            await self._reply(update, f"🗑 <b>${removed}</b> removed from watchlist")
        else:
            await self._reply(update, "ℹ️ Ticker was not in the watchlist")

    @_authorized
    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tickers = self.watchlist.list()
        if not tickers:
            await self._reply(update, "📋 Watchlist is empty. Add one with /add TICKER")
            return
        lines = ["📋 <b>WATCHLIST</b>", ""] + [f"• <code>{t}</code>" for t in tickers]
        lines.append(f"\nTotal: <b>{len(tickers)}</b>")
        await self._reply(update, "\n".join(lines))

    @_authorized
    async def cmd_discover(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.discovery_engine is None:
            await self._reply(update, "❌ Discovery engine is disabled (DISCOVERY_ENABLED=false)")
            return
        await self._reply(update, "🔭 Scanning the market for candidates…")
        pool = await self.discovery_engine.refresh(force=True)
        if not pool.candidates:
            text = "🔭 <b>DISCOVERY</b>\n\nNo candidates found this pass."
            if pool.warnings:
                text += "\n⚠️ " + "; ".join(pool.warnings[:3])
            await self._reply(update, text)
            return
        lines = [
            f"🔭 <b>HUNTER DISCOVERY</b> — {pool.session.value}",
            "",
        ]
        for i, c in enumerate(pool.candidates[:8], 1):
            price = f"${c.price:.2f}" if c.price is not None else "?"
            chg = f"{c.change_percent:+.1f}%" if c.change_percent is not None else "?"
            reasons = ",".join(c.reasons[:2])
            missing = f" <i>[missing: {','.join(c.missing_fields)}]</i>" if c.missing_fields else ""
            lines.append(f"{i}. <code>{c.symbol}</code> — {price} | {chg} | {c.discovery_score}/100 | {reasons}{missing}")
        lines += [
            "",
            f"raw={pool.raw_count} unique={len(pool.candidates)} dupes={pool.duplicate_count} invalid={pool.invalid_count}",
            "<i>Candidates feed the normal Hunter pipeline; nothing is auto-alerted.</i>",
        ]
        await self._reply(update, "\n".join(lines))

    @_authorized
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = SessionClock.get_session()
        o = self.orchestrator
        realtime = getattr(o.market_provider, "is_realtime", lambda: False)()
        news_count = len(o.news_providers)
        ai_on = bool(SETTINGS.openai_api_key)
        lines = [
            "📊 <b>HUNTER STATUS</b>",
            "",
            f"🕒 Session: <b>{session.value}</b>",
            f"📡 Market data: {type(o.market_provider).__name__} ({'real-time' if realtime else 'delayed'})",
            f"📰 News providers: {news_count}",
            f"🧠 AI analyzer: {'ON' if ai_on else 'OFF (neutral defaults)'}",
            f"📜 Options engine: {'ON' if SETTINGS.options_enabled else 'OFF'}",
            f"📋 Watchlist: {len(self.watchlist.list())} tickers",
            f"🔔 Alerts stored: {self.memory.alert_count()}",
        ]
        if self.scheduler and self.scheduler.last_pass_summary:
            lines.append(f"🔁 Last auto-pass: {self.scheduler.last_pass_summary}")
        await self._reply(update, "\n".join(lines))

    @_authorized
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        recent = self.memory.recent_alerts(limit=5)
        lines = [
            "📈 <b>MEMORY STATS</b>",
            "",
            f"Total alerts: <b>{self.memory.alert_count()}</b>",
        ]
        if recent:
            lines += ["", "<b>Recent alerts:</b>"]
            for a in recent:
                lines.append(f"• ${a['ticker']} | {a['decision']} | {a['score']}/100")
        else:
            lines.append("No alerts recorded yet.")
        await self._reply(update, "\n".join(lines))

    @staticmethod
    def _format_signal(signal) -> str:
        emoji = {HunterDecision.HUNT_NOW: "🚀", HunterDecision.WATCH: "👀", HunterDecision.IGNORE: "❌"}
        lines = [
            f"{emoji[signal.decision]} <b>${signal.ticker}</b> | {signal.session or SessionClock.get_session().value}",
            f"Decision: <b>{signal.decision.value}</b> | Score <b>{signal.hunter_score}/100</b>",
            f"📰 {signal.catalyst_type or 'N/A'} | Impact {signal.news_impact}/100 | {signal.sentiment}",
            f"Reaction: {signal.reaction_status} | Liquidity: {signal.liquidity_status}",
            f"Setup: {signal.technical_setup} | Options: {signal.options_bias} | Trap risk: {signal.trap_risk}",
        ]
        if signal.current_price is not None:
            lines.insert(2, f"💵 ${signal.current_price:.2f}")
        if signal.reasoning:
            lines.append(f"\n🧠 {signal.reasoning[:300]}")
        return "\n".join(lines)
