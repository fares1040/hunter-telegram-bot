"""Hunter Bot — AI Analyzer"""
import json
from typing import Dict
import openai

from models.news import CatalystEvent, CatalystType
from config.settings import SETTINGS
from utils.logger import LOGGER


class AIAnalyzer:
    def __init__(self):
        self.client = None
        if SETTINGS.openai_api_key:
            self.client = openai.AsyncOpenAI(api_key=SETTINGS.openai_api_key)
        self.model = SETTINGS.openai_model

    async def analyze_event(self, event: CatalystEvent) -> CatalystEvent:
        if not self.client or not SETTINGS.openai_api_key:
            LOGGER.warning("[AI] No API key — skipping AI analysis")
            event.impact_score = 50
            event.materiality_score = 50
            event.sentiment = "NEUTRAL"
            return event

        sources_text = "\n\n".join([
            f"Source: {n.source} (Tier: {n.source_tier.name})\nHeadline: {n.headline}\nSummary: {n.summary or 'N/A'}"
            for n in event.all_sources[:3]
        ])

        system_prompt = """You are a ruthless, skeptical financial analyst who specializes in momentum trading.
Your job is to judge whether a news item is ACTUALLY material or just PR fluff.

CRITICAL RULES:
- "Partnership" without dollar amounts or strategic importance = LOW impact
- "Agreement" without terms = LOW impact
- Earnings beats with raised guidance = HIGH impact
- FDA approvals = HIGH impact
- Government contracts with disclosed value = HIGH impact
- Vague AI announcements without product/revenue = MEDIUM at best
- Stock offerings, dilution = NEGATIVE
- Downgrades from major banks = NEGATIVE

You must be conservative. Most news is noise."""

        user_prompt = f"""Analyze this catalyst for ${event.ticker}:

{sources_text}

Respond with valid JSON only:
{{
    "catalyst_type": "EARNINGS|FDA|CONTRACT|PARTNERSHIP|MERGER|ACQUISITION|GOVERNMENT|PRODUCT|AI|UPGRADE|DOWNGRADE|SEC_FILING|OFFERING|OTHER",
    "sentiment": "VERY_POSITIVE|POSITIVE|NEUTRAL|NEGATIVE|VERY_NEGATIVE",
    "impact_score": <0-100>,
    "materiality_score": <0-100>,
    "priced_in_probability": <0.0-1.0>,
    "reasoning": "<1-2 sentences explaining your verdict>"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
            )

            result = json.loads(response.choices[0].message.content)

            event.catalyst_type = self._map_catalyst_type(result.get("catalyst_type", "OTHER"))
            event.sentiment = self._normalize_sentiment(result.get("sentiment", "NEUTRAL"))
            event.impact_score = max(0, min(100, int(result.get("impact_score", 50))))
            event.materiality_score = max(0, min(100, int(result.get("materiality_score", 50))))
            event.priced_in_probability = max(0.0, min(1.0, float(result.get("priced_in_probability", 0.5))))

            LOGGER.info(
                f"[AI] {event.ticker} | {event.catalyst_type.value} | "
                f"Impact: {event.impact_score} | Materiality: {event.materiality_score} | "
                f"Priced-in: {event.priced_in_probability:.0%}"
            )

        except Exception as e:
            LOGGER.error(f"[AI] Analysis failed for {event.ticker}: {e}")
            event.impact_score = 50
            event.materiality_score = 40
            event.sentiment = "NEUTRAL"

        return event

    def _map_catalyst_type(self, raw: str) -> CatalystType:
        mapping = {
            "EARNINGS": CatalystType.EARNINGS,
            "FDA": CatalystType.FDA,
            "CONTRACT": CatalystType.CONTRACT,
            "PARTNERSHIP": CatalystType.PARTNERSHIP,
            "MERGER": CatalystType.MERGER,
            "ACQUISITION": CatalystType.ACQUISITION,
            "GOVERNMENT": CatalystType.GOVERNMENT,
            "PRODUCT": CatalystType.PRODUCT,
            "AI": CatalystType.AI,
            "UPGRADE": CatalystType.UPGRADE,
            "DOWNGRADE": CatalystType.DOWNGRADE,
            "SEC_FILING": CatalystType.SEC_FILING,
            "OFFERING": CatalystType.OFFERING,
        }
        return mapping.get(raw.upper(), CatalystType.OTHER)

    def _normalize_sentiment(self, raw: str) -> str:
        allowed = {"VERY_POSITIVE", "POSITIVE", "NEUTRAL", "NEGATIVE", "VERY_NEGATIVE"}
        return raw.upper() if raw.upper() in allowed else "NEUTRAL"
