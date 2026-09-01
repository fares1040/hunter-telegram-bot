"""Hunter Bot — News Engine"""
import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from difflib import SequenceMatcher

from models.news import NewsItem, CatalystEvent, CatalystType, SourceTier, ensure_utc
from core.exceptions import NewsValidationError
from core.data_confidence import DataQuality
from utils.logger import LOGGER


class NewsEngine:
    # Cache configuration: max entries and TTL (seconds)
    _CACHE_MAX_SIZE = 1000
    _CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, providers: List):
        self.providers = providers
        self._event_cache: Dict[str, tuple[CatalystEvent, float]] = {}  # event_id -> (event, timestamp)

    async def gather_news(self, ticker: str, max_age_hours: float = 24.0) -> List[NewsItem]:
        since = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        all_items: List[NewsItem] = []

        for provider in self.providers:
            try:
                items = await provider.fetch_news(ticker, since)
                all_items.extend(items)
            except Exception as e:
                LOGGER.warning(f"[NewsEngine] Provider {provider.name} failed: {e}")

        unique = {}
        for item in all_items:
            if item.id not in unique:
                unique[item.id] = item

        return list(unique.values())

    def cluster_events(self, items: List[NewsItem]) -> List[CatalystEvent]:
        if not items:
            return []

        clusters: List[List[NewsItem]] = []
        sorted_items = sorted(items, key=lambda x: ensure_utc(x.published_at) or datetime.min.replace(tzinfo=timezone.utc))

        for item in sorted_items:
            placed = False
            for cluster in clusters:
                representative = cluster[0]
                if item.published_at and representative.published_at:
                    item_ts = ensure_utc(item.published_at)
                    rep_ts = ensure_utc(representative.published_at)
                    time_diff = abs((item_ts - rep_ts).total_seconds())
                    if time_diff > 4 * 3600:
                        continue
                sim = SequenceMatcher(None, item.headline.lower(), representative.headline.lower()).ratio()
                if sim > 0.65:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        events = []
        for cluster in clusters:
            primary = min(cluster, key=lambda x: x.source_tier.value)
            others = [i for i in cluster if i.id != primary.id]

            event_id = self._generate_event_id(primary)
            now = time.time()
            if event_id in self._event_cache:
                existing, _ = self._event_cache[event_id]
                existing.additional_sources.extend(others)
                self._event_cache[event_id] = (existing, now)
                events.append(existing)
                continue

            event = CatalystEvent(
                event_id=event_id,
                ticker=primary.ticker,
                catalyst_type=CatalystType.OTHER,
                headline_summary=primary.headline,
                primary_source=primary,
                additional_sources=others,
            )
            self._score_event_basics(event)
            self._event_cache[event_id] = (event, now)
            self._evict_expired_cache(now)
            events.append(event)

        return events

    def _generate_event_id(self, item: NewsItem) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]", "", item.headline.lower())[:40]
        return hashlib.sha256(f"{item.ticker}:{normalized}".encode()).hexdigest()[:16]

    def _score_event_basics(self, event: CatalystEvent):
        best_tier = event.best_tier
        tier_scores = {
            SourceTier.TIER_1_OFFICIAL: 100,
            SourceTier.TIER_2_MAJOR: 85,
            SourceTier.TIER_3_FINANCIAL: 60,
            SourceTier.TIER_4_UNVERIFIED: 25,
        }
        event.source_tier_score = tier_scores.get(best_tier, 30)

        age = event.primary_source.age_minutes
        if age is None:
            event.freshness_score = 50
        elif age <= 30:
            event.freshness_score = 100
        elif age >= 240:
            event.freshness_score = 0
        else:
            event.freshness_score = int(100 - ((age - 30) / 210) * 100)

    def _evict_expired_cache(self, now: float) -> None:
        """Remove expired entries and enforce max cache size."""
        # Remove expired entries (older than TTL)
        expired_keys = [
            k for k, (_, ts) in self._event_cache.items()
            if now - ts > self._CACHE_TTL_SECONDS
        ]
        for k in expired_keys:
            del self._event_cache[k]

        # If still over max size, remove oldest entries
        if len(self._event_cache) > self._CACHE_MAX_SIZE:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(self._event_cache.items(), key=lambda x: x[1][1])
            excess = len(self._event_cache) - self._CACHE_MAX_SIZE
            for k, _ in sorted_items[:excess]:
                del self._event_cache[k]

    def filter_material_events(self, events: List[CatalystEvent], min_tier_score: int = 40) -> List[CatalystEvent]:
        filtered = []
        for event in events:
            if event.source_tier_score < min_tier_score:
                LOGGER.info(f"[NewsEngine] Rejected {event.event_id}: low tier score ({event.source_tier_score})")
                continue
            if not event.is_fresh(max_age_minutes=180):
                # Tier-1 contextual window up to 240m: keep as contextual evidence for WATCH, not fresh for HUNT
                if event.best_tier == SourceTier.TIER_1_OFFICIAL and event.is_fresh(max_age_minutes=240):
                    # keep tier-1 stale as contextual (freshness will remain STALE, WhyNow stays UNKNOWN/PARTIAL)
                    filtered.append(event)
                    continue
                LOGGER.info(f"[NewsEngine] Rejected {event.event_id}: stale")
                continue
            filtered.append(event)
        return filtered