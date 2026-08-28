# Hunter AI — Production Deployment Guide

## Overview

Hunter AI is a persistent Python service that scans the stock market on a schedule and sends Telegram alerts. It is **not** a web application and does not require a web server.

- **Version**: 2.10.0
- **Entrypoint**: `python3 run.py`
- **Runtime**: asyncio (long-running process)
- **Architecture**: `HunterOrchestrator` → `ScanScheduler.run_forever()`

## Requirements

- Python 3.9+
- pip
- Telegram bot token + chat ID (required)
- Optional: Polygon API key (real-time data + options)
- Optional: Finnhub API key (news)
- Optional: OpenAI API key (catalyst reasoning)

## Installation

```bash
cd hunter-telegram-bot
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your real credentials
```

## Environment Variables (names only)

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Yes | — | Alert chat ID |
| `OPENAI_API_KEY` | No | | Catalyst reasoning |
| `OPENAI_MODEL` | No | gpt-4o-mini | OpenAI model |
| `POLYGON_API_KEY` | No | | Real-time data + options |
| `FINNHUB_API_KEY` | No | | News gathering |
| `HUNTER_MIN_SCORE` | No | 70 | Minimum score to act |
| `HUNTER_MIN_DATA_CONFIDENCE` | No | 60 | Minimum data confidence |
| `ACCOUNT_SIZE` | No | | Account size for risk |
| `RISK_PER_TRADE_PCT` | No | 0.5 | Risk per trade (%) |
| `OPTIONS_ENABLED` | No | true | Options analysis |
| `MARKET_TIMEZONE` | No | America/New_York | Market timezone |
| `LOG_LEVEL` | No | INFO | Logging level |
| `MEMORY_DB_PATH` | No | data/hunter_memory.sqlite3 | SQLite path |
| `SCAN_INTERVAL_REGULAR` | No | 300 | Scan interval (s) regular |
| `SCAN_INTERVAL_EXTENDED` | No | 900 | Scan interval (s) extended |
| `SCAN_INTERVAL_CLOSED` | No | 1800 | Scan interval (s) closed |
| `TELEGRAM_COMMANDS_ENABLED` | No | true | Enable /commands |
| `TELEGRAM_AUTHORIZED_CHAT_ID` | No | | Extra authorized chat IDs |
| `DISCOVERY_ENABLED` | No | true | Market discovery |
| `DISCOVERY_POOL_SIZE` | No | 10 | Discovery pool size |
| `REALTIME_ENABLED` | No | false | Stage 2 real-time (disabled) |
| `POLYGON_WS_ENABLED` | No | false | Polygon WebSocket (disabled) |
| `OPTIONS_FLOW_REALTIME_ENABLED` | No | false | Stage 3 true flow (disabled) |

## Startup

```bash
python3 run.py
```

The service starts `ScanScheduler.run_forever()` — a continuous loop that scans the watchlist at configured intervals per market session. Press Ctrl+C to stop gracefully.

## Docker

```bash
docker build -t hunter-ai .
docker run -d --name hunter -v $(pwd)/data:/app/data --env-file .env hunter-ai
```

Or with docker-compose:

```bash
docker-compose up -d
docker-compose logs -f hunter
docker-compose down
```

## Systemd Service

Create `/etc/systemd/system/hunter.service`:

```ini
[Unit]
Description=Hunter AI Bot
After=network.target

[Service]
Type=simple
User=hunter
WorkingDirectory=/opt/hunter-ai
ExecStart=/usr/bin/python3 run.py
Restart=always
RestartSec=10
EnvironmentFile=/opt/hunter-ai/.env
StandardOutput=append:/var/log/hunter.log
StandardError=append:/var/log/hunter.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hunter
sudo systemctl start hunter
sudo systemctl status hunter
journalctl -u hunter -f
```

## Restart Behavior

- **Docker**: `restart: unless-stopped` — automatic restart on crash.
- **Systemd**: `Restart=always, RestartSec=10` — automatic restart with 10s delay.
- **Keyboard interrupt**: Graceful shutdown — scheduler stops, command bot shuts down, SQLite connections close.
- **Provider failure**: Graceful — errors logged, next ticker/cycle continues.
- **SQLite**: File-based, recreated if missing. Persistent volume required for production.

## Log Inspection

```bash
# Docker
docker-compose logs -f hunter

# Systemd
journalctl -u hunter -f
tail -f /var/log/hunter.log

# Console (dev)
python3 run.py 2>&1 | tee hunter.log
```

Logs are written to stdout with format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Configuration errors at startup` | Missing/placeholder Telegram credentials | Set valid `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| `Insufficient market data` | yfinance/Polygon unavailable | Check network, verify API keys |
| `No recent news` | Finnhub key missing or no news | Set `FINNHUB_API_KEY` or use yfinance news fallback |
| `Options chain unavailable` | No options provider configured | Set `POLYGON_API_KEY` for options data |
| `Telegram failed` | Bot token invalid or chat blocked | Verify token, ensure bot is admin in chat |
| High error count in scheduler | Provider timeouts | Check network latency, reduce `scan_interval_*` |
| SQLite locked | Multiple processes accessing same DB | Use single process instance, or separate DB path per instance |

## Security Notes

- Never commit `.env` with real secrets.
- `.env.example` contains only variable names and placeholders.
- `validate_production()` rejects placeholder tokens at startup.
- Provider keys are never logged.
- Real-time and True Flow features are **disabled by default**.

## Architecture

```
python3 run.py
  → main()
    → SETTINGS.validate_production()
    → HunterOrchestrator()
      → YFinanceProvider / PolygonProvider
      → NewsEngine, CatalystEngine, DecisionEngine, etc.
      → TelegramNotifier
      → ScanScheduler.run_forever()
        → loop: scan_pass(tickers) → process_ticker(ticker)
          → full pipeline → DecisionEngine.decide()
          → HUNT_NOW → memory.remember() → TelegramNotifier.send_signal()
```

DecisionEngine is the sole authority for HUNT_NOW/WATCH/IGNORE. Decision 2.0 (WhyNow, Conviction, OpportunityQuality) is additive and never overrides gates.