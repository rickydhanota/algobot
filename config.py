import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'

# ── Capital & Risk ────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 3000.0
MAX_RISK_PER_TRADE_PCT = 0.02     # 2% per trade = $60 on $3k
MAX_DAILY_LOSS_PCT = 0.05         # stop trading after 5% down = $150
MAX_CONCURRENT_POSITIONS = 5
MIN_SIGNAL_SCORE = 68             # 0-100, only enter A/B setups

# ── Session timing (all Eastern) ─────────────────────────────────────────────
ORB_CAPTURE_MINUTES = 15          # capture open range first 15 min
ACTIVE_TRADING_START = (10, 0)    # (hour, minute)
NO_NEW_TRADES_AFTER = (15, 30)    # close-only after 3:30 PM
MARKET_CLOSE = (16, 0)

# ── Tape / order-flow thresholds ─────────────────────────────────────────────
RVOL_MIN = 1.5                    # relative-volume floor
IMBALANCE_STRONG = 0.40           # buy/sell imbalance → strong signal
IMBALANCE_MODERATE = 0.20         # moderate signal
LARGE_PRINT_MULTIPLIER = 3.0      # print > 3× avg size = institutional

# ── Technical parameters ──────────────────────────────────────────────────────
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5               # stop = entry ± 1.5 ATR
ATR_TARGET_MULT = 2.5             # target = entry ± 2.5 ATR (1.67 R:R min)
VWAP_BUFFER_PCT = 0.001           # 0.1% band around VWAP
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
VOLUME_MA_PERIOD = 20

# ── Options parameters ────────────────────────────────────────────────────────
OPT_DTE_MIN = 7
OPT_DTE_MAX = 21
OPT_DELTA_MIN = 0.25
OPT_DELTA_MAX = 0.50
OPT_VOI_THRESHOLD = 3.0           # unusual = volume/OI > 3×
OPT_MAX_SPREAD_PCT = 0.10         # skip wide spreads > 10% of mid
OPT_IV_RANK_BUY_MAX = 40          # buy options when IVR < 40
OPT_IV_RANK_SELL_MIN = 60         # sell premium when IVR > 60

# ── News & earnings filters ───────────────────────────────────────────────────
NEWS_FETCH_INTERVAL_MIN = 15       # how often to refresh news (minutes)
NEWS_LOOKBACK_HOURS = 24           # how far back to fetch articles
SKIP_EARNINGS_DAYS = 1             # skip trades if earnings within N days
NEWS_NEGATIVE_THRESHOLD = -0.30    # sentiment ≤ this rejects the trade
NEWS_POSITIVE_BOOST = 0.30         # sentiment ≥ this adds bonus to signal score
NEWS_SCORE_BONUS = 5               # points added/subtracted from signal score

# ── Intel: 13F filings + congressional trades ────────────────────────────────
INTEL_REFRESH_HOURS = 6            # how often to refresh 13F + political feeds
INTEL_SCORE_BOOST_MAX = 5          # cap on score boost from smart-money alignment

# ── Macro / Fed / White House monitoring ──────────────────────────────────────
MACRO_FETCH_INTERVAL_MIN = 30      # how often to refresh macro feeds
MACRO_LOOKBACK_HOURS = 48          # how far back to read Fed/WH articles
HALT_ON_FOMC_DAY = True            # no new trades on FOMC announcement days
HALT_ON_GEOPOLITICAL = True        # no new trades on major war/crisis news
REDUCE_SIZE_ON_HIGH_RISK = True    # use macro risk multiplier for position size

# ── Default watchlist ─────────────────────────────────────────────────────────
WATCHLIST = [
    'SPY', 'QQQ', 'IWM',
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL',
    'AMD', 'PLTR', 'COIN', 'MSTR', 'SOFI',
]
