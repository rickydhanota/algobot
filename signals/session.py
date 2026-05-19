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
        'midday':      -5,    # loosened from -10 — let solid setups (score 80+) fire midday
        'off_hours':   0,
    }.get(window, 0)


def size_multiplier(window: str) -> float:
    """Position-size multiplier per window (compounds with macro multiplier)."""
    return {
        'power_open':  1.20,   # larger in high-volume window
        'power_close': 1.20,
        'midday':      0.50,   # tightened from 0.7 — midday allowed but half-size
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


def dte_preference(
    session_window: str,
    tape_strength: float = None,
    tape_confluence: int = None,
    vol_rate: float = None,
) -> tuple:
    """
    Pick the DTE band the chain filter should use, based on current
    market conditions.

    Logic:
      • POWER window + strong tape + elevated volume
          → 0–7 DTE  (max gamma for fast moves, low theta exposure)
      • POWER window + moderate signals
          → 1–14 DTE (same-week weeklies, some gamma cushion)
      • MIDDAY or weaker conditions
          → 7–30 DTE (further-out for theta protection on slow setups)

    Returns (dte_min, dte_max) — both inclusive.
    """
    is_power = session_window in ('power_open', 'power_close')

    strong_tape = (
        (tape_strength is not None and tape_strength >= 0.5)
        or (tape_confluence is not None and tape_confluence >= 70)
    )
    moderate_tape = (
        (tape_strength is not None and tape_strength >= 0.3)
        or (tape_confluence is not None and tape_confluence >= 50)
    )
    elevated_vol = vol_rate is not None and vol_rate >= 1.0
    healthy_vol  = vol_rate is None or vol_rate >= 0.7

    if is_power and strong_tape and elevated_vol:
        return (0, 7)        # max-gamma window
    if is_power and (moderate_tape or elevated_vol) and healthy_vol:
        return (1, 14)       # same-week + next-week
    return (7, 30)            # safer further-out for slow conditions


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
