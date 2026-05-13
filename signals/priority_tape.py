"""
Enhanced tape reader for high-priority options underlyings (SPY, SPX, TSLA).

These 3 names are the bread-and-butter of options day-trading. They warrant
deeper tape analysis than the regular reader provides:

  • Multi-window imbalance — last 100 / 500 / 1000 trades
  • Cumulative Volume Delta (CVD) — running buy-vs-sell aggregate
  • Block-print detection — institutional-size trades
  • Sweep detection — multiple aggressive prints within a short window
  • Per-symbol tuned thresholds — TSLA needs bigger swings than SPY to matter

SPX has no tape (it's an index), so we proxy from SPY's tape and scale references.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

# Symbols treated as SPY-proxied
SPX_PROXY_SYMBOL = 'SPY'
SPX_PROXY_SCALE  = 10.0   # SPX ≈ SPY × 10


# Per-symbol tape profiles
PROFILES = {
    'SPY': {
        'block_size':         5000,    # >5k-share prints = institutional
        'sweep_window_ms':    500,
        'sweep_min_prints':   4,
        'imbalance_thresh':   0.20,    # SPY moves on thinner imbalances
        'cvd_window_seconds': 300,     # 5-min CVD memory
        'tick_size':          0.01,
    },
    'TSLA': {
        'block_size':         1000,
        'sweep_window_ms':    1000,
        'sweep_min_prints':   5,
        'imbalance_thresh':   0.30,    # TSLA is noisy, need stronger signal
        'cvd_window_seconds': 180,     # faster mover, shorter memory
        'tick_size':          0.01,
    },
    'SPX': {                            # used only for dashboard labeling
        'block_size':         5000,
        'sweep_window_ms':    500,
        'sweep_min_prints':   4,
        'imbalance_thresh':   0.20,
        'cvd_window_seconds': 300,
        'tick_size':          0.05,
    },
}

DEFAULT_PROFILE = {
    'block_size':         2000,
    'sweep_window_ms':    1000,
    'sweep_min_prints':   4,
    'imbalance_thresh':   0.25,
    'cvd_window_seconds': 300,
    'tick_size':          0.01,
}

PRIORITY_SYMBOLS = ('SPY', 'SPX', 'TSLA')


@dataclass
class TapeWindow:
    """Aggregated metrics over a fixed number of recent trades."""
    n: int
    buy_vol: int = 0
    sell_vol: int = 0
    block_count: int = 0
    avg_size: float = 0.0

    @property
    def imbalance(self) -> float:
        tot = self.buy_vol + self.sell_vol
        return 0.0 if tot == 0 else (self.buy_vol - self.sell_vol) / tot

    @property
    def total_volume(self) -> int:
        return self.buy_vol + self.sell_vol


@dataclass
class PriorityTapeSignal:
    symbol: str
    # Multi-window imbalances
    imbalance_100: float
    imbalance_500: float
    imbalance_1000: float
    # Cumulative volume delta over recent window
    cvd: int                       # net signed share volume
    cvd_trend: str                 # 'up' | 'down' | 'flat'
    # Block / sweep stats
    block_count_recent: int        # blocks in last 100 trades
    block_volume_pct: float        # blocks as % of recent volume
    sweep_detected: bool
    sweep_direction: Optional[str] # 'buy' | 'sell' | None
    # Aggregate
    direction: str                 # 'buy' | 'sell' | 'neutral'
    strength: float                # 0 - 1
    confluence_score: int          # 0 - 100 — combined confidence
    sample_count: int

    @property
    def is_strong(self) -> bool:
        return self.confluence_score >= 70

    @property
    def is_moderate(self) -> bool:
        return self.confluence_score >= 50

    def to_dict(self) -> dict:
        return {
            'symbol':             self.symbol,
            'imbalance_100':      round(self.imbalance_100, 3),
            'imbalance_500':      round(self.imbalance_500, 3),
            'imbalance_1000':     round(self.imbalance_1000, 3),
            'cvd':                self.cvd,
            'cvd_trend':          self.cvd_trend,
            'block_count':        self.block_count_recent,
            'block_volume_pct':   round(self.block_volume_pct, 3),
            'sweep_detected':     self.sweep_detected,
            'sweep_direction':    self.sweep_direction,
            'direction':          self.direction,
            'strength':           round(self.strength, 3),
            'confluence_score':   self.confluence_score,
            'sample_count':       self.sample_count,
        }


class PriorityTapeReader:
    """
    Stateful enhanced tape reader for SPY, SPX, TSLA.
    Feed trades via record_trade; query via analyze.
    """

    def __init__(self):
        # symbol → deque of (ts, price, size, direction)
        self._buffer: Dict[str, deque] = {}
        # symbol → list of CVD samples [(ts, cvd_value), ...]
        self._cvd_history: Dict[str, deque] = {}
        # Last classified direction for tick rule
        self._last_price: Dict[str, float] = {}
        self._last_direction: Dict[str, str] = {}

    def _profile(self, symbol: str) -> dict:
        return PROFILES.get(symbol, DEFAULT_PROFILE)

    def _get_buffer(self, symbol: str) -> deque:
        if symbol not in self._buffer:
            self._buffer[symbol] = deque(maxlen=2000)
            self._cvd_history[symbol] = deque(maxlen=600)  # ~10 min @ 1/s
        return self._buffer[symbol]

    # ── Trade ingestion ──────────────────────────────────────────────────────

    def record_trade(self, symbol: str, price: float, size: int,
                     bid: float = 0.0, ask: float = 0.0):
        # Map SPX trades onto SPY's buffer for proxy reading
        if symbol == 'SPX':
            return  # SPX has no trades; we read SPY directly
        # Also record SPY trades onto SPX as a proxy
        for sym in (symbol, 'SPX') if symbol == 'SPY' else (symbol,):
            direction = self._classify(sym, price, bid, ask)
            buf = self._get_buffer(sym)
            buf.append({
                'ts':        datetime.now(timezone.utc),
                'price':     price,
                'size':      size,
                'direction': direction,
            })

    def _classify(self, symbol: str, price: float, bid: float, ask: float) -> str:
        if ask and bid:
            mid = (bid + ask) / 2
            if price >= ask:     d = 'buy'
            elif price <= bid:   d = 'sell'
            elif price > mid:    d = 'buy'
            elif price < mid:    d = 'sell'
            else:                d = self._tick_rule(symbol, price)
        else:
            d = self._tick_rule(symbol, price)
        self._last_price[symbol] = price
        self._last_direction[symbol] = d
        return d

    def _tick_rule(self, symbol: str, price: float) -> str:
        prev = self._last_price.get(symbol)
        if prev is None:    return 'neutral'
        if price > prev:    return 'buy'
        if price < prev:    return 'sell'
        return self._last_direction.get(symbol, 'neutral')

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _window(self, trades: List[dict], n: int) -> TapeWindow:
        slice_ = trades[-n:] if len(trades) >= n else trades
        if not slice_:
            return TapeWindow(n=n)

        sizes = [t['size'] for t in slice_]
        avg = float(np.mean(sizes))
        buy  = sum(t['size'] for t in slice_ if t['direction'] == 'buy')
        sell = sum(t['size'] for t in slice_ if t['direction'] == 'sell')
        return TapeWindow(n=n, buy_vol=buy, sell_vol=sell, avg_size=avg)

    def _detect_sweep(self, trades: List[dict], profile: dict) -> tuple[bool, Optional[str]]:
        """Look for ≥N same-direction prints within sweep_window_ms."""
        if len(trades) < profile['sweep_min_prints']:
            return False, None
        window_ms = profile['sweep_window_ms']
        last = trades[-15:]   # only check recent
        # Walk forward
        for i in range(len(last) - profile['sweep_min_prints'] + 1):
            seg = last[i:i + profile['sweep_min_prints']]
            dt_ms = (seg[-1]['ts'] - seg[0]['ts']).total_seconds() * 1000
            if dt_ms > window_ms:
                continue
            dirs = {t['direction'] for t in seg if t['direction'] in ('buy', 'sell')}
            if len(dirs) == 1:
                return True, dirs.pop()
        return False, None

    def _compute_cvd(self, trades: List[dict], window_seconds: int) -> tuple[int, str]:
        if not trades:
            return 0, 'flat'
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        recent = [t for t in trades if t['ts'] >= cutoff]
        cvd = sum(t['size'] if t['direction'] == 'buy' else -t['size'] for t in recent)
        # Trend: compare first half vs second half of window
        if len(recent) >= 20:
            mid = len(recent) // 2
            first_cvd  = sum(t['size'] if t['direction'] == 'buy' else -t['size'] for t in recent[:mid])
            second_cvd = sum(t['size'] if t['direction'] == 'buy' else -t['size'] for t in recent[mid:])
            if second_cvd > first_cvd + 100:    trend = 'up'
            elif second_cvd < first_cvd - 100:  trend = 'down'
            else:                               trend = 'flat'
        else:
            trend = 'flat'
        return cvd, trend

    def analyze(self, symbol: str) -> Optional[PriorityTapeSignal]:
        trades = list(self._get_buffer(symbol))
        if len(trades) < 20:
            return None

        profile = self._profile(symbol)
        w100  = self._window(trades, 100)
        w500  = self._window(trades, 500)
        w1000 = self._window(trades, 1000)

        # Block detection on last 100 trades
        block_size = profile['block_size']
        recent_100 = trades[-100:]
        block_count = sum(1 for t in recent_100 if t['size'] >= block_size)
        recent_vol = sum(t['size'] for t in recent_100) or 1
        block_vol_pct = sum(t['size'] for t in recent_100 if t['size'] >= block_size) / recent_vol

        sweep_detected, sweep_dir = self._detect_sweep(trades, profile)
        cvd, cvd_trend = self._compute_cvd(trades, profile['cvd_window_seconds'])

        # Aggregate direction — multi-window agreement matters
        directions = []
        for w in (w100, w500, w1000):
            if w.total_volume == 0:
                continue
            if w.imbalance >  profile['imbalance_thresh']:  directions.append('buy')
            elif w.imbalance < -profile['imbalance_thresh']: directions.append('sell')
            else:                                             directions.append('neutral')

        if directions.count('buy') >= 2:
            direction = 'buy'
        elif directions.count('sell') >= 2:
            direction = 'sell'
        else:
            direction = 'neutral'

        # Confluence score
        score = 0
        # Multi-window agreement (0-40)
        if directions.count(direction) == 3 and direction != 'neutral':
            score += 40
        elif directions.count(direction) == 2 and direction != 'neutral':
            score += 25

        # Magnitude of recent imbalance (0-20)
        s = abs(w100.imbalance)
        if   s > 0.5: score += 20
        elif s > 0.3: score += 14
        elif s > 0.2: score += 8

        # CVD trend confirmation (0-15)
        if (direction == 'buy' and cvd_trend == 'up') or (direction == 'sell' and cvd_trend == 'down'):
            score += 15
        elif cvd_trend != 'flat':
            score += 5

        # Block participation (0-15)
        if block_vol_pct > 0.20:   score += 15
        elif block_vol_pct > 0.10: score += 8

        # Sweep alignment (0-10)
        if sweep_detected and sweep_dir == direction:
            score += 10

        strength = min(1.0, abs(w100.imbalance) / 0.6)

        return PriorityTapeSignal(
            symbol=symbol,
            imbalance_100=w100.imbalance,
            imbalance_500=w500.imbalance,
            imbalance_1000=w1000.imbalance,
            cvd=cvd,
            cvd_trend=cvd_trend,
            block_count_recent=block_count,
            block_volume_pct=block_vol_pct,
            sweep_detected=sweep_detected,
            sweep_direction=sweep_dir,
            direction=direction,
            strength=strength,
            confluence_score=min(100, score),
            sample_count=len(trades),
        )

    def snapshot_all(self) -> Dict[str, dict]:
        """Return dashboard-friendly dict for all priority symbols."""
        out = {}
        for sym in PRIORITY_SYMBOLS:
            sig = self.analyze(sym)
            if sig:
                out[sym] = sig.to_dict()
        return out
