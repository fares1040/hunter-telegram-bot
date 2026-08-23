# HUNTER TELEGRAM BOT — FINAL INTELLIGENCE BUILD

A Telegram-first, news-driven stock opportunity engine. The project is intentionally **logic-first**: the bot is the product; a web UI is not required.

## What this build contains

### 1. Candidate Gate
- Cheap first-pass filter before expensive news/AI/options work.
- Price, dollar-liquidity and extreme-gap checks.

### 2. Catalyst Intelligence
- Multi-source news collection.
- Source-quality ranking.
- Duplicate/cluster handling.
- Freshness and materiality gates.
- AI skepticism rules for PR fluff, vague partnerships and dilution.

### 3. Market Reaction
- Before/after-news price and volume windows.
- No invented reaction data: missing windows are `DATA_INSUFFICIENT`.
- Negative reaction can veto a bullish catalyst.

### 4. Technical Discovery
- MA20 / MA50 / MA200.
- RSI, VWAP, ATR, Bollinger Bands.
- Premarket and regular-session separation.
- Breakout/reclaim/setup classification.

### 5. Liquidity Proxy
- RVOL.
- Dollar volume.
- Volume spike.
- Explicitly **not** labeled as money inflow, dark-pool flow, or smart-money flow unless a future provider supplies evidence for those claims.

### 6. Options / Contracts
- Observable option-chain analysis.
- Call/put volume, OI and premium comparisons.
- Put/call ratios.
- Liquid contract candidate selection.
- Strike, expiration, mid and IV are carried into Telegram alerts.
- Free yfinance option chain fallback.
- Optional Polygon options snapshot provider when `POLYGON_API_KEY` is configured.
- Chain-derived bias is clearly labeled as inferred; it is not proof of institutional activity.

### 7. Trap Detector
Detects and penalizes:
- Negative post-news reaction.
- Weak liquidity.
- Extreme gaps.
- High priced-in probability.
- Extreme RSI.

### 8. Composite Score
One transparent 0–100 score combining:
`News + Impact + Reaction + Liquidity + Technical + Options + Risk - Trap`

### 9. Risk Gate
Generates scenario levels from price/ATR/structure:
- Breakout trigger.
- Risk level.
- T1 / T2 / T3.
- Reward-to-risk.
- Optional account-based position sizing.

These are **scenario calculations for research**, not automatic orders or personalized investment advice.

### 10. Memory
SQLite stores alert keys and outcomes so the bot can avoid duplicate alerts and later learn from observed results.

### 11. Backtesting Foundation
`backtest.py` provides a clean evaluation layer for historical returns without look-ahead. A larger historical replay can be built on top of the same scoring interfaces.

### 12. Telegram Output
The alert is designed around decision quality, not UI:
- Decision.
- Score.
- Catalyst.
- Reaction.
- Liquidity.
- Technical setup.
- Options bias.
- Scenario risk levels.
- Contract candidate when available.
- Trap warnings.

## Data-provider reality
- **yfinance:** free fallback and option-chain fallback; not guaranteed real-time.
- **Finnhub:** news when configured.
- **Polygon:** optional higher-quality market/options source when configured.
- **OpenAI:** optional catalyst reasoning layer.
- **Telegram:** alert delivery.

The system never pretends a free provider is real-time when it is not.

## Local verification

```bash
python -m compileall .
pytest -q
```

Current local suite: **91 passed, 0 failed**.

## Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add keys to .env
python run.py
```

Never commit `.env`, API keys, `data/`, `venv/`, or `__pycache__/`.

## Production direction

The architecture is ready for the next layer: real-time Polygon market data, richer options trade/quote events, stronger institutional-data providers, historical backtesting, outcome labeling, adaptive weights, and scheduled Telegram scanning.


## Version 2.1.0 — Critical Production Upgrade

This release addresses four critical production areas:

1. **Real-time-capable market data:** PolygonProvider is implemented and selected automatically when `POLYGON_API_KEY` is configured; yfinance remains the explicit fallback.
2. **Correct market reaction:** before/after windows now use symmetric 5-minute volume comparisons and explicit +5m/+15m/+30m price checkpoints with `DATA_INSUFFICIENT` when evidence is missing.
3. **Options intelligence:** contract scoring considers direction, DTE, volume, open interest, spread, moneyness, IV and Greeks when supplied by the provider.
4. **No-lookahead backtesting:** replay utilities model entry/stop/targets bar-by-bar and report hit rate, average R and drawdown without peeking into future bars.

Additional context added: market regime, sector strength, trap-risk multiplier scoring, persistent memory, and clear separation between chain-derived inference and true order-flow evidence.

## Version 2.2.0 — Continuous Operation & Interactive Commands

The bot is now a long-running service instead of a one-shot scan:

1. **Session-aware scheduler** (`core/scheduler.py`): full watchlist passes on a loop with per-session pacing — `SCAN_INTERVAL_REGULAR`, `SCAN_INTERVAL_EXTENDED` (premarket/after-hours), `SCAN_INTERVAL_CLOSED`.
2. **Persistent watchlist** (`core/watchlist.py`): SQLite-backed; survives restarts; seeded with AAPL/NVDA/TSLA.
3. **Interactive Telegram commands** (`bot/commands.py`): `/scan [TICKER]`, `/add TICKER`, `/remove TICKER`, `/watchlist`, `/status`, `/stats`, `/help`. Commands are restricted to authorized chat IDs (`TELEGRAM_CHAT_ID` plus optional `TELEGRAM_AUTHORIZED_CHAT_ID`).
4. **Deterministic test clock**: fixtures anchor to the most recent regular-session minute so the suite no longer fails when run on weekends/holidays (120 tests passing).

## Version 2.3.0 — Phase 2.5: Market Discovery Engine

Hunter now searches the market instead of only scanning a fixed watchlist:

1. **Universe provider interface** (`providers/universe/base_provider.py`): pluggable sources answering "which symbols are worth looking at right now?" with whatever real metrics they supply — nothing invented.
2. **YFinance screener provider** (`providers/universe/yfinance_screener_provider.py`): real live discovery via the Yahoo screener API — session-aware query plans (custom low-threshold movers + predefined gainers/losers/most-actives), NASDAQ/NYSE only, OTC deliberately excluded until a reliable source exists.
3. **Watchlist universe provider**: user symbols always included as a source.
4. **DiscoveryEngine** (`engines/discovery.py`): normalization → deduplication (cross-source merge) → filters → transparent 0–100 score with per-component breakdown (`score_breakdown`) and explicit `missing_fields`; bounded concurrency; TTL cache; provider failures isolated.
5. **Integration**: scheduler merges the ranked candidate pool into each scan pass (`max_scan_batch` capped); `/discover` command shows the pool without auto-alerting.

Known limits (documented, by design): delayed data, extended-hours screener coverage not guaranteed, Polygon real-time snapshot/gainer endpoints are paid-tier — a `PolygonUniverseProvider` can slot in behind the same interface later.

## Version 2.4.0 — Phase 2.6: Catalyst Intelligence Engine

Every news event now gets a deterministic, explainable intelligence profile before the optional AI layer runs:

1. **Normalized catalyst model** (`models/catalyst.py`): `CatalystProfile` strictly separates REAL data (headline/source/tier/url/published_at), INFERENCE (category, sentiment, confidence) and SCORES (materiality + transparent breakdown). The LLM may refine an event afterwards but never overwrites these computed values.
2. **CatalystEngine** (`engines/catalyst_engine.py`): rule-based classification (earnings/guidance, FDA, contracts, partnerships, M&A, analyst actions, offerings/dilution/bankruptcy, SEC filings, compliance), sentiment voting with MIXED detection, freshness buckets (BREAKING ≤30m / RECENT ≤120m / AGING ≤360m / STALE), and a 0–100 materiality score assembled from five visible components: category weight + source quality + freshness + financial figures in headline + multi-source corroboration.
3. **Trap-risk flags**: offering/dilution/bankruptcy headlines are auto-flagged `TRAP_RISK`; stale high-materiality events get an explicit "may already be priced in" warning; these flow into the decision engine's trap warnings.
4. **Real no-key news source** (`providers/news/yfinance_provider.py`): Yahoo Finance news via yfinance — always registered alongside Finnhub; publisher→tier mapping mirrors existing conventions; malformed items skipped, never fabricated. Documented limit: Yahoo is an aggregator with retail skew, so its TIER_4 items are routinely rejected by the material gate — by design.
5. **`/news TICKER` command**: shows top catalysts with recommendation (OPPORTUNITY/WATCH/TRAP_RISK/NEUTRAL), age bucket, per-component materiality math and source links — deterministic output only.
6. **Fix**: `cluster_events` crashed when any article lacked a timestamp (naive-vs-aware datetime sort); now handled. `CatalystType` extended additively (GUIDANCE, DILUTION, BANKRUPTCY, REGULATORY, CLINICAL_TRIAL).

203 tests passing.

## Version 2.5.0 — Phase 2.7: Technical Intelligence Engine

Raw market data is now turned into structured, explainable technical intelligence — without touching the legacy pipeline consumed by Risk/Trap/Decision:

1. **Structured model** (`models/technical.py`): `TechnicalIntelligence` with strict REAL DATA / INFERENCE / SCORE separation and a `timeframe` field ("1d" today) so multi-timeframe analysis can slot in later without structural changes.
2. **Trend**: MA20/50/200 (MA200 only from real 200-bar history — the old MA50 fallback was a silent fake), price-vs-MA distances, alignment string, 5-day MA slopes, HH/HL/LH/LL pivot structure with evidence → BULLISH / BEARISH / NEUTRAL / TRANSITION; UNKNOWN when history is insufficient.
3. **Momentum**: RSI(14), ROC-5/ROC-10, MACD(12/26/9) when ≥35 bars, BUILDING/FADING acceleration via RSI delta, conservative two-pivot RSI divergence only when clearly detectable → STRONG/POSITIVE/NEUTRAL/NEGATIVE/WEAK.
4. **Volatility**: ATR + ATR%, Bollinger bands/width, width percentile vs trailing widths → SQUEEZE / EXPANSION / NORMAL / EXTREME (ATR% > 8 forces EXTREME).
5. **VWAP**: computed from real intraday bars when present (ΣTP·V/ΣV), else provider session snapshot, else explicitly UNAVAILABLE — never fabricated. Reclaim/rejection flags require bar data.
6. **Volume/RVOL**: reuses `TickerData.relative_volume`/`dollar_volume`; spike ratio, volume acceleration, dollar volume → LOW/NORMAL/ELEVATED/HIGH/EXTREME.
7. **Support/Resistance**: deterministic levels from swing pivots (touch-counted), previous-day high/low, premarket high/low, 10-day extremes; each level carries price, type, strength 0–100, distance %, and source evidence; near-duplicates merge with combined evidence.
8. **Setups**: FAILED_BREAKOUT, BREAKOUT/HIGHER_HIGH_BREAKOUT, LOWER_LOW_BREAKDOWN, VWAP_RECLAIM/REJECTION, PULLBACK, RESISTANCE/SUPPORT_TEST, CONSOLIDATION — every detection carries explicit evidence.
9. **Explainable Technical Score** (0–100): six components with documented weights — Trend 25%, Momentum 25%, Volume 15%, Volatility 15%, Structure/S-R 15%, VWAP 5% — renormalized over available components; unavailable components are listed with reasons in the breakdown.
10. **Integration**: intelligence is built inside the existing `TechnicalEngine.analyze()` from already-loaded data (zero new network calls, ~7 ms/ticker CPU); legacy `setup_score` semantics preserved for Risk/Trap/Decision gates; DecisionEngine optionally receives the intelligence object and surfaces its summary on signals.

Known limitations: RVOL depends on provider-supplied session volume/avg-volume fields (honestly None when absent); divergence detection intentionally conservative; multi-timeframe execution deferred until intraday resampling lands.

252 tests passing (208 preserved + 44 new).
