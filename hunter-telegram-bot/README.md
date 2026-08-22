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
