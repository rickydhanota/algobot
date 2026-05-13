# AlgoBot

An automated paper-trading bot for stocks and options using the Alpaca API. Reads real-time tape (order flow), evaluates multi-factor signals, and executes bracketed trades with strict risk management.

## Features

- **Real-time tape reading** — order-flow analysis using quote-rule and tick-rule classification to gauge buy/sell pressure
- **Three stock strategies** — VWAP momentum, opening-range breakout (ORB), and volume-surge follow-through
- **Options scanning** — unusual-activity detection (V/OI ratio), strike/DTE selection, IV-aware contract scoring
- **Risk management** — 2% max risk per trade, daily-loss circuit breaker, 5-position concurrency cap, 1.5:1 minimum R:R
- **Bracketed orders** — entry, stop-loss, and take-profit submitted in a single request
- **Live HTML dashboard** — equity, day P&L, interactive P&L bar chart with adjustable time range (1D–12M), open positions, closed trades, CSV export
- **Auto-launch** — cron job + macOS `pmset` wake schedule for hands-free weekday operation

## Project structure

```
trading_bot/
├── config.py              # all tunable parameters
├── main.py                # bot orchestrator + terminal dashboard
├── dashboard.html         # browser dashboard (auto-refreshing)
├── launch.sh              # cron entry-point
├── data/
│   ├── alpaca_client.py   # singleton API/stream clients
│   └── market_data.py     # bars, VWAP, ORB, options chain
├── signals/
│   ├── tape_reader.py     # real-time order-flow analysis
│   └── technical.py       # VWAP, RSI, ATR, RVOL, trend
├── strategy/
│   ├── risk_manager.py    # sizing + circuit breakers
│   ├── stock_strategy.py  # 3 stock setups, 0-100 scoring
│   └── options_strategy.py# unusual activity + contract scoring
├── execution/
│   └── order_manager.py   # bracketed orders, position tracking
└── performance/
    └── tracker.py         # SQLite trade log + stats
```

## Setup

1. Create an Alpaca paper-trading account: https://app.alpaca.markets

2. Clone and install:
   ```bash
   git clone <repo-url>
   cd trading_bot
   ./setup.sh
   ```

3. Add your API keys to `.env` (copied from `.env.example`):
   ```
   ALPACA_API_KEY=PK...
   ALPACA_SECRET_KEY=...
   PAPER_TRADING=true
   ```

4. Run:
   ```bash
   source .venv/bin/activate
   python main.py
   ```

5. Open the dashboard in a browser:
   ```bash
   open dashboard.html
   ```

## Auto-launch on macOS

Schedule the bot to start every weekday at 6:25 AM PT (5 min before market open):

```bash
# Install cron entry
(crontab -l 2>/dev/null; echo "25 6 * * 1-5 $HOME/trading_bot/launch.sh") | crontab -

# Wake Mac from sleep at 6:20 AM
sudo pmset repeat wakeorpoweron MTWRF 06:20:00
```

## Strategy parameters

All thresholds live in `config.py`. Key defaults:

| Parameter | Value | Meaning |
|---|---|---|
| `MIN_SIGNAL_SCORE` | 68 | Reject setups scoring under 68/100 |
| `MAX_RISK_PER_TRADE_PCT` | 0.02 | Risk 2% of equity per trade |
| `MAX_DAILY_LOSS_PCT` | 0.05 | Stop trading after 5% daily loss |
| `RVOL_MIN` | 1.5 | Minimum relative volume |
| `ATR_STOP_MULT` | 1.5 | Stop distance = 1.5× ATR |
| `ATR_TARGET_MULT` | 2.5 | Target distance = 2.5× ATR (1.67:1 R:R) |
| `OPT_DTE_MIN/MAX` | 7 / 21 | Options expiry window |
| `OPT_DELTA_MIN/MAX` | 0.25 / 0.50 | Options delta range |

## Disclaimer

This is software for paper-trading and educational purposes. Past performance — even in paper accounts — is not indicative of future results. Automated trading carries significant risk; never trade with money you cannot afford to lose.
