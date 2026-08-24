"""Telegram presentation layer for Hunter signals."""
from telegram import Bot
from models.signal import HunterSignal, HunterDecision
from utils.logger import LOGGER


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    async def send_signal(self, signal: HunterSignal):
        emoji = {HunterDecision.HUNT_NOW: "🚀", HunterDecision.WATCH: "👀", HunterDecision.IGNORE: "❌"}
        lines = [
            "🚨 <b>HUNTER AI ALERT</b>", "", f"<b>${signal.ticker}</b>  |  {signal.session}",
            f"{emoji[signal.decision]} <b>{signal.decision.value}</b>  |  Score <b>{signal.hunter_score}/100</b>",
            "", f"📰 <b>{signal.catalyst_type}</b> | Impact {signal.news_impact}/100 | {signal.sentiment}",
            f"📈 Reaction: {signal.reaction_status}", f"💧 Liquidity: {signal.liquidity_status} | RVOL: {signal.rvol or 'N/A'}",
            f"📊 Technical: {signal.technical_setup} | Options: {signal.options_bias}",
        ]
        if signal.current_price is not None:
            lines += [f"💵 Reference: ${signal.current_price:.2f}"]
        if signal.entry_trigger is not None:
            lines += ["", "🎯 <b>SCENARIO LEVELS</b>", f"Trigger: ${signal.entry_trigger:.2f}", f"Risk level: ${signal.stop_price:.2f}", f"T1: ${signal.target_1:.2f} | T2: ${signal.target_2:.2f} | T3: ${signal.target_3:.2f}", f"R/R to T1: {signal.reward_to_risk or 0:.1f}x"]
        if signal.target_result is not None:
            t = signal.target_result
            lines += ["", "🧭 <b>TARGET INTELLIGENCE</b>", f"Direction: {t.direction} | Status: {t.status}"]
            if t.tp1:
                lines.append(f"TP1 zone: ${t.tp1.zone.zone_low:.2f}–${t.tp1.zone.zone_high:.2f} ({t.tp1.zone.source_type})")
            if t.tp2:
                lines.append(f"TP2 zone: ${t.tp2.zone.zone_low:.2f}–${t.tp2.zone.zone_high:.2f} ({t.tp2.zone.source_type})")
            if t.tp3:
                lines.append(f"TP3 zone: ${t.tp3.zone.zone_low:.2f}–${t.tp3.zone.zone_high:.2f} ({t.tp3.zone.source_type})")
            if t.risk_reward is not None:
                lines.append(f"Target R/R: {t.risk_reward:.2f}x")
            if t.score is not None:
                lines.append(f"Target score: {t.score.total}/100 | Confidence: {t.confidence.value}/100")
            lines.append("<i>Structural zones, not single prices.</i>")
        if signal.contract_symbol:
            lines += ["", "📜 <b>OPTIONS CANDIDATE</b>", f"{signal.contract_symbol}", f"Strike: {signal.contract_strike} | Exp: {signal.contract_expiration}", f"Mid: ${signal.contract_mid:.2f}" if signal.contract_mid is not None else "Mid: N/A", f"IV: {signal.contract_iv:.1%}" if signal.contract_iv is not None else "IV: N/A", "Chain-derived candidate; not an execution instruction."]
        if signal.warnings:
            lines += ["", "⚠️ <b>WARNINGS</b>"] + [f"• {w}" for w in signal.warnings[:8]]
        lines += ["", f"🧠 {signal.reasoning}", "", "<i>Educational market-research alert. Manage risk independently.</i>"]
        try:
            await self.bot.send_message(chat_id=self.chat_id, text="\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)
        except Exception as e:
            LOGGER.error(f"[Telegram] Failed to send: {e}")
