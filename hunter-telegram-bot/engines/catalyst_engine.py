"""Catalyst Engine — deterministic classification, materiality, freshness, trap flags.

Rule-based and fully explainable. The LLM layer (ai/analyzer.py) may refine an
event AFTER this engine has spoken, but never replaces these computed values.
No network access happens here: it consumes what NewsEngine already gathered.
"""
import re
from typing import Dict, List, Optional, Tuple

from models.catalyst import (
    CatalystProfile,
    FreshnessBucket,
    Recommendation,
    SentimentLabel,
    TRAP_CATEGORIES,
)
from models.news import CatalystEvent, CatalystType, SourceTier

# ---------------------------------------------------------------------------
# Deterministic rule table. Each rule: (rule_name, keywords, category,
# sentiment, weight). First matching rule wins for category; all matching
# sentiments vote so conflicting signals produce MIXED.
# ---------------------------------------------------------------------------
RULES: List[Tuple[str, Tuple[str, ...], str, SentimentLabel]] = [
    ("offering",       ("offering", "public offering", "shares outstanding", "registered direct"), "FINANCING_OFFERING", SentimentLabel.NEGATIVE),
    ("dilution",       ("dilution", "dilutive", "warrant inducement", "at-the-market", "atm agreement"), "DILUTION", SentimentLabel.NEGATIVE),
    ("bankruptcy",     ("bankruptcy", "chapter 11", "chapter 7", "receivership", "delisting notice"), "BANKRUPTCY_RESTRUCTURING", SentimentLabel.NEGATIVE),
    ("reverse_split",  ("reverse split", "reverse stock split"), "DILUTION", SentimentLabel.NEGATIVE),
    ("sec_filing",     ("sec filing", "10-k", "10-q", "8-k", "s-1 filing", "s-3 filing", "form 4"), "SEC_FILINGS", SentimentLabel.UNKNOWN),
    ("fda",            ("fda", "approval", "cleared", "phase 3", "phase 2 trial", "clinical trial", "orphan drug"), "FDA_REGULATORY", SentimentLabel.POSITIVE),
    ("earnings_beat",  ("beats estimates", "tops estimates", "record revenue", "raises guidance", "raised guidance", "boosts outlook", "strong quarter"), "GUIDANCE", SentimentLabel.POSITIVE),
    ("earnings_miss",  ("misses estimates", "lowers guidance", "lowered guidance", "cuts outlook", "weak quarter", "withdraws guidance"), "GUIDANCE", SentimentLabel.NEGATIVE),
    ("earnings",       ("earnings", "quarterly results", "reports q", "annual results"), "EARNINGS", SentimentLabel.UNKNOWN),
    ("gov_contract",   ("government contract", "defense contract", "department of defense", "nasa", "federal contract"), "CONTRACTS", SentimentLabel.POSITIVE),
    ("contract",       ("contract", "awarded", "purchase order", "agreement with"), "CONTRACTS", SentimentLabel.POSITIVE),
    ("partnership",    ("partnership", "collaboration", "strategic alliance", "joint venture", "mou"), "PARTNERSHIPS", SentimentLabel.POSITIVE),
    ("acquisition",    ("acquisition", "acquire", "to be acquired", "merger", "buyout", "takeover"), "MA_ACQUISITION", SentimentLabel.POSITIVE),
    ("product",        ("launches", "unveils", "introduces", "expands product"), "PRODUCT_LAUNCH", SentimentLabel.POSITIVE),
    ("ai_announcement",("artificial intelligence", "ai-powered", "machine learning platform", "generative ai"), "AI_ANNOUNCEMENT", SentimentLabel.POSITIVE),
    ("upgrade",        ("upgraded", "raises price target", "initiates at buy", "overweight", "outperform"), "ANALYST_ACTION", SentimentLabel.POSITIVE),
    ("downgrade",      ("downgraded", "cuts price target", "initiates at sell", "underweight", "underperform"), "ANALYST_ACTION", SentimentLabel.NEGATIVE),
    ("compliance",     ("nasdaq compliance", "deficiency letter", "non-compliance", "regulatory probe", "investigation"), "COMPLIANCE_EXCHANGE", SentimentLabel.NEGATIVE),
]

CATEGORY_WEIGHTS: Dict[str, int] = {
    "MA_ACQUISITION": 35,
    "FDA_REGULATORY": 32,
    "GUIDANCE": 30,
    "EARNINGS": 28,
    "CONTRACTS": 26,
    "BANKRUPTCY_RESTRUCTURING": 30,
    "DILUTION": 24,
    "FINANCING_OFFERING": 22,
    "ANALYST_ACTION": 18,
    "SEC_FILINGS": 16,
    "PARTNERSHIPS": 20,
    "AI_ANNOUNCEMENT": 14,
    "PRODUCT_LAUNCH": 14,
    "COMPLIANCE_EXCHANGE": 20,
}

FIGURES_RE = re.compile(r"(\$\s?\d|\d+(\.\d+)?\s?%|\b\d+(\.\d+)?[mb]\b)", re.IGNORECASE)

CATEGORY_TO_TYPE: Dict[str, CatalystType] = {
    "EARNINGS": CatalystType.EARNINGS,
    "GUIDANCE": CatalystType.GUIDANCE,
    "FDA_REGULATORY": CatalystType.FDA,
    "CONTRACTS": CatalystType.CONTRACT,
    "PARTNERSHIPS": CatalystType.PARTNERSHIP,
    "MA_ACQUISITION": CatalystType.ACQUISITION,
    "PRODUCT_LAUNCH": CatalystType.PRODUCT,
    "AI_ANNOUNCEMENT": CatalystType.AI,
    "ANALYST_ACTION": CatalystType.UPGRADE,
    "SEC_FILINGS": CatalystType.SEC_FILING,
    "FINANCING_OFFERING": CatalystType.OFFERING,
    "DILUTION": CatalystType.DILUTION,
    "BANKRUPTCY_RESTRUCTURING": CatalystType.BANKRUPTCY,
    "COMPLIANCE_EXCHANGE": CatalystType.REGULATORY,
}


class CatalystEngine:
    """Consumes clustered CatalystEvents from NewsEngine and produces normalized,
    deterministic CatalystProfiles."""

    # Mirrors NewsEngine._score_event_basics; used when an event arrives
    # without a precomputed source_tier_score (standalone usage).
    TIER_SCORES = {
        SourceTier.TIER_1_OFFICIAL: 100,
        SourceTier.TIER_2_MAJOR: 85,
        SourceTier.TIER_3_FINANCIAL: 60,
        SourceTier.TIER_4_UNVERIFIED: 25,
    }

    def classify(self, headline: str) -> Tuple[str, SentimentLabel, int, List[str]]:
        text = (headline or "").lower()
        matched_rules: List[str] = []
        category_votes: Dict[str, int] = {}
        pos = neg = 0
        for name, keywords, category, sentiment in RULES:
            hits = [k for k in keywords if k in text]
            if not hits:
                continue
            matched_rules.append(name)
            category_votes[category] = category_votes.get(category, 0) + len(hits)
            if sentiment is SentimentLabel.POSITIVE:
                pos += len(hits)
            elif sentiment is SentimentLabel.NEGATIVE:
                neg += len(hits)

        if not matched_rules:
            return "UNKNOWN", SentimentLabel.UNKNOWN, 0, []

        category = max(category_votes.items(), key=lambda kv: kv[1])[0]
        if pos and neg:
            label = SentimentLabel.MIXED
            confidence = 55
        elif pos:
            label = SentimentLabel.POSITIVE
            confidence = min(90, 60 + 8 * pos)
        elif neg:
            label = SentimentLabel.NEGATIVE
            confidence = min(90, 60 + 8 * neg)
        else:
            label = SentimentLabel.UNKNOWN
            confidence = 35
        return category, label, confidence, matched_rules

    @staticmethod
    def freshness_bucket(age_minutes: Optional[float]) -> FreshnessBucket:
        if age_minutes is None:
            return FreshnessBucket.STALE
        if age_minutes <= 30:
            return FreshnessBucket.BREAKING
        if age_minutes <= 120:
            return FreshnessBucket.RECENT
        if age_minutes <= 360:
            return FreshnessBucket.AGING
        return FreshnessBucket.STALE

    def materiality_breakdown(
        self,
        category: str,
        source_tier_score: int,
        freshness: FreshnessBucket,
        has_figures: bool,
        cluster_size: int,
    ) -> Dict[str, int]:
        freshness_points = {
            FreshnessBucket.BREAKING: 15,
            FreshnessBucket.RECENT: 12,
            FreshnessBucket.AGING: 6,
            FreshnessBucket.STALE: 0,
        }
        corroboration = min(max(cluster_size - 1, 0), 3) * 5
        return {
            "category": CATEGORY_WEIGHTS.get(category, 5),
            "source_quality": round(source_tier_score / 100 * 20),
            "freshness": freshness_points[freshness],
            "figures": 15 if has_figures else 0,
            "corroboration": corroboration,
        }

    def assess(self, event: CatalystEvent) -> CatalystProfile:
        src = event.primary_source
        age = src.age_minutes
        bucket = self.freshness_bucket(age)
        category, label, confidence, rules = self.classify(src.headline)
        has_figures = bool(FIGURES_RE.search(src.headline or ""))
        cluster_size = len(event.all_sources)
        tier_score = event.source_tier_score or self.TIER_SCORES.get(event.best_tier, 30)

        breakdown = self.materiality_breakdown(category, tier_score, bucket, has_figures, cluster_size)
        materiality = int(sum(breakdown.values()))

        missing: List[str] = []
        if src.published_at is None:
            missing.append("published_at")
        if age is None:
            missing.append("age")
        if not has_figures:
            missing.append("figures")

        evidence = [f"{tier_score}/100 tier score"]
        if cluster_size > 1:
            evidence.append(f"{cluster_size} independent sources")
        if rules:
            evidence.append("rules: " + ",".join(rules))
        if has_figures:
            evidence.append("headline contains financial figures")

        profile = CatalystProfile(
            symbol=event.ticker,
            headline=src.headline,
            source=src.source,
            source_tier_score=tier_score,
            published_at=src.published_at,
            url=src.url,
            age_minutes=round(age, 1) if age is not None else None,
            freshness=bucket,
            category=category,
            sentiment=label,
            classification_confidence=confidence,
            materiality=min(materiality, 100),
            materiality_breakdown=breakdown,
            matched_rules=rules,
            evidence=evidence,
            missing_fields=missing,
            cluster_size=cluster_size,
        )
        self._mark_trap_risk(profile)
        return profile

    def _mark_trap_risk(self, profile: CatalystProfile) -> None:
        if profile.category in TRAP_CATEGORIES:
            profile.is_trap_risk = True
            profile.trap_reasons.append(f"Dilution/financing risk: {profile.category}")
        if profile.freshness is FreshnessBucket.STALE and profile.materiality >= 50:
            profile.is_trap_risk = True
            profile.trap_reasons.append("Stale news may already be priced in")
        if profile.sentiment is SentimentLabel.MIXED and profile.materiality >= 60:
            profile.trap_reasons.append("Conflicting signals inside one catalyst")

        if profile.is_trap_risk:
            profile.recommendation = Recommendation.TRAP_RISK
        elif profile.sentiment is SentimentLabel.POSITIVE and profile.materiality >= 60 and profile.freshness in (FreshnessBucket.BREAKING, FreshnessBucket.RECENT):
            profile.recommendation = Recommendation.OPPORTUNITY
        elif profile.materiality >= 40:
            profile.recommendation = Recommendation.WATCH
        else:
            profile.recommendation = Recommendation.NEUTRAL

    SENTIMENT_TO_EVENT = {
        SentimentLabel.POSITIVE: "POSITIVE",
        SentimentLabel.NEGATIVE: "NEGATIVE",
        SentimentLabel.MIXED: "NEUTRAL",
        SentimentLabel.UNKNOWN: "NEUTRAL",
    }

    def enrich(self, event: CatalystEvent, profile: Optional[CatalystProfile] = None) -> CatalystProfile:
        """Apply deterministic values onto the existing CatalystEvent BEFORE the
        optional AI layer runs. Existing non-default values are preserved."""
        profile = profile or self.assess(event)
        if event.catalyst_type is CatalystType.OTHER and profile.category in CATEGORY_TO_TYPE:
            mapped = CATEGORY_TO_TYPE[profile.category]
            event.catalyst_type = CatalystType.DOWNGRADE if (
                mapped is CatalystType.UPGRADE and profile.sentiment is SentimentLabel.NEGATIVE
            ) else mapped
        if event.sentiment == "NEUTRAL":
            event.sentiment = self.SENTIMENT_TO_EVENT[profile.sentiment]
        if event.materiality_score == 0:
            event.materiality_score = profile.materiality
        return profile
