"""
Session-window logic for time-of-day trading priority.

Intraday volume is U-shaped: heaviest at the open and close, thinnest at
midday. Trade quality follows the same curve — so we:

  • Prioritize POWER OPEN (9:30–11:00 ET) — first 90 minutes
  • Prioritize POWER HOUR (15:00–15:50 ET) — last hour before day-trade cutoff
  • Penalize MIDDAY (11:00–15:00 ET) — chop / low volume
  • Skip OFF-HOURS — outside session

Effects:
  • Score adjustment is added to every signal before threshold check
  • Adaptive learner buckets stats per (symbol, strategy, session_window)
    so the bot can learn that "ORB on TSLA at power_open" performs
    differently from "ORB on TSLA at midday".
"""
from __future__ import annotations
from datetime import datetime

# All times Eastern
POWER_OPEN_START  = (9, 30)
POWER_OPEN_END    = (11, 0)        # first 90 minutes
MIDDAY_START      = (11, 0)
MIDDAY_END        = (15, 0)
POWER_CLOSE_START = (15, 0)        # power hour
POWER_CLOSE_END   = (15, 55)       # day-trade-only cutoff


def current_window(now_et: datetime) -> str:
    """Return current session window label."""
    t = (now_et.hour, now_et.minute)
    if POWER_OPEN_START  <= t < POWER_OPEN_END:   return 'power_open'
    if POWER_CLOSE_START <= t < POWER_CLOSE_END:  return 'power_close'
    if MIDDAY_START      <= t < MIDDAY_END:       return 'midday'
    return 'off_hours'


def score_adjustment(window: str) -> int:
    """Points added/subtracted from signal score based on session window."""
    return {
        'power_open':  +5,
        'power_close': +5,
        'midday':      -7,    # require markedly better setup at midday
        'off_hours':   0,
    }.get(window, 0)


def size_multiplier(window: str) -> float:
    """Position-size multiplier per window (compounds with macro multiplier)."""
    return {
        'power_open':  1.20,   # larger in high-volume window
        'power_close': 1.20,
        'midday':      0.70,   # smaller when liquidity is thin
        'off_hours':   0.0,    # no trading
    }.get(window, 1.0)


def is_priority(window: str) -> bool:
    return window in ('power_open', 'power_close')


def label(window: str) -> str:
    return {
        'power_open':  'POWER OPEN',
        'power_close': 'POWER HOUR',
        'midday':      'MIDDAY',
        'off_hours':   'OFF-HOURS',
    }.get(window, window.upper())


def to_dict(now_et: datetime) -> dict:
    """Dashboard-friendly snapshot."""
    w = current_window(now_et)
    return {
        'window':       w,
        'label':        label(w),
        'is_priority':  is_priority(w),
        'score_adj':    score_adjustment(w),
        'size_mult':    size_multiplier(w),
    }
