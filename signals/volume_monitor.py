"""
Real-time volume monitoring for priority symbols (SPY / SPX / TSLA).

Tracks the *rate* of trade prints over rolling windows. Low rate or stale
market = unreliable signals, so we skip trading.

NOTE: Alpaca's free tier streams only IEX-exchange trades (~2-3% of total
NYSE/Nasdaq volume). We therefore track RELATIVE rates rather than absolute
share counts — is activity higher or lower than this symbol's recent norm?
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


# Per-symbol thresholds
THRESHOLDS = {
    'SPY':  {'min_ratio': 0.5, 'stale_seconds': 15},
    'SPX':  {'min_ratio': 0.5, 'stale_seconds': 15},
    'TSLA': {'min_ratio': 0.5, 'stale_seconds': 20},
}
DEFAULT_THRESHOLD = {'min_ratio': 0.5, 'stale_seconds': 30}


@dataclass
class VolumeState:
    symbol: str
    trades_last_60s:     int
    trades_last_5min:    int
    rolling_avg_per_min: float
    rate_ratio:          float       # current rate ÷ rolling avg (1.0 = normal)
    is_healthy:          bool
    is_stale:            bool        # no prints in N seconds
    last_trade_age_s:    float
    share_volume_60s:    int

    @property
    def health_label(self) -> str:
        if self.is_stale:        return 'stale'
        if not self.is_healthy:  return 'low'
        if self.rate_ratio >= 1.5: return 'elevated'
        return 'normal'

    def to_dict(self) -> dict:
        return {
            'symbol':              self.symbol,
            'trades_last_60s':     self.trades_last_60s,
            'trades_last_5min':    self.trades_last_5min,
            'rolling_avg_per_min': round(self.rolling_avg_per_min, 1),
            'rate_ratio':          round(self.rate_ratio, 2),
            'is_healthy':          self.is_healthy,
            'is_stale':            self.is_stale,
            'last_trade_age_s':    round(self.last_trade_age_s, 1),
            'share_volume_60s':    self.share_volume_60s,
            'health_label':        self.health_label,
        }


class VolumeMonitor:
    """Tracks trade-print rate and share volume per priority symbol."""

    def __init__(self):
        # symbol -> deque of (timestamp, size)
        self._trades: Dict[str, deque] = {}

    def _buf(self, sym: str) -> deque:
        if sym not in self._trades:
            self._trades[sym] = deque(maxlen=20000)
        return self._trades[sym]

    def record_trade(self, symbol: str, size: int = 0):
        ts = datetime.now(timezone.utc)
        self._buf(symbol).append((ts, size))
        # SPY trades double-count for SPX proxy
        if symbol == 'SPY':
            self._buf('SPX').append((ts, size))

    def analyze(self, symbol: str) -> Optional[VolumeState]:
        buf = self._buf(symbol)
        if len(buf) < 10:
            return None

        now = datetime.now(timezone.utc)
        t_60s   = now - timedelta(seconds=60)
        t_5min  = now - timedelta(minutes=5)
        t_30min = now - timedelta(minutes=30)

        last_60s_trades = [t for t in buf if t[0] >= t_60s]
        last_5m_count   = sum(1 for t in buf if t[0] >= t_5min)
        last_30m_count  = sum(1 for t in buf if t[0] >= t_30min)

        rolling_avg = last_30m_count / 30.0 if last_30m_count else 0
        rate_ratio  = (len(last_60s_trades) / rolling_avg) if rolling_avg > 0 else 1.0

        cfg = THRESHOLDS.get(symbol, DEFAULT_THRESHOLD)
        last_age = (now - buf[-1][0]).total_seconds()
        is_stale = last_age > cfg['stale_seconds']

        if rolling_avg <= 0:
            is_healthy = not is_stale          # Bootstrapping: trust if we have any prints
        else:
            is_healthy = (rate_ratio >= cfg['min_ratio']) and (not is_stale)

        return VolumeState(
            symbol=symbol,
            trades_last_60s=len(last_60s_trades),
            trades_last_5min=last_5m_count,
            rolling_avg_per_min=rolling_avg,
            rate_ratio=rate_ratio,
            is_healthy=is_healthy,
            is_stale=is_stale,
            last_trade_age_s=last_age,
            share_volume_60s=sum(t[1] for t in last_60s_trades),
        )

    def is_tradeable(self, symbol: str) -> tuple[bool, str]:
        """Returns (ok, reason) — used by the bot to gate trades."""
        v = self.analyze(symbol)
        if v is None:
            return True, ''         # not enough data yet → don't block
        if v.is_stale:
            return False, f'tape stale ({v.last_trade_age_s:.0f}s since last print)'
        if not v.is_healthy:
            return False, f'low volume (rate {v.rate_ratio:.2f}× normal)'
        return True, ''

    def snapshot_all(self) -> Dict[str, dict]:
        out = {}
        for sym in ('SPY', 'SPX', 'TSLA'):
            v = self.analyze(sym)
            if v:
                out[sym] = v.to_dict()
        return out
