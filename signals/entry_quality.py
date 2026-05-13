"""
Pre-trade entry-quality gate.

Before sending an order, validate that the entry price is *good* — not the
top of a 30-second spike, not into a wide spread, not when the tape has
stopped printing.

Returns a verdict + a 0-100 quality score plus a human-readable reason.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


# Tunables
CHASE_WINDOW_SECONDS = 30
CHASE_MAX_PCT = 0.003           # 0.3% adverse move in last 30s = chasing
MAX_SPREAD_PCT_DEFAULT = 0.0020 # 0.2% spread ceiling
PRIORITY_MAX_SPREAD = {
    'SPY':  0.0005,             # SPY trades 1 cent wide — be strict
    'TSLA': 0.0015,
    'SPX':  0.0005,             # using SPY-derived
}
VWAP_FAR_PCT = 0.005            # >0.5% from VWAP is "far"
MIN_QUALITY_TO_TRADE = 55       # score floor


@dataclass
class EntryVerdict:
    allowed: bool
    score: int                  # 0-100
    reason: str                 # human-readable label
    spike_pct: float = 0.0      # adverse move in last window
    vwap_distance_pct: float = 0.0
    spread_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            'allowed':          self.allowed,
            'score':            self.score,
            'reason':           self.reason,
            'spike_pct':        round(self.spike_pct, 4),
            'vwap_distance_pct': round(self.vwap_distance_pct, 4),
            'spread_pct':       round(self.spread_pct, 4),
        }


class EntryQualityChecker:
    """
    Maintains a short price history per symbol so we can evaluate whether the
    *current* price is a reasonable entry vs. a momentary spike.
    """

    def __init__(self):
        # symbol → deque[(ts, price)]
        self._prices: Dict[str, deque] = {}

    def record_price(self, symbol: str, price: float):
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=600)   # ~10 min @ 1 trade/s
        self._prices[symbol].append((datetime.now(timezone.utc), price))

    def check(
        self,
        symbol: str,
        direction: str,           # 'long' | 'short'
        entry_price: float,
        vwap: float = 0.0,
        bid: float = 0.0,
        ask: float = 0.0,
        imbalance: float = 0.0,
        volume_healthy: bool = True,
    ) -> EntryVerdict:
        history = self._prices.get(symbol)

        # ── Check 1: spread ───────────────────────────────────────────────────
        spread_pct = 0.0
        if bid > 0 and ask > 0 and entry_price > 0:
            spread_pct = (ask - bid) / entry_price
        max_spread = PRIORITY_MAX_SPREAD.get(symbol, MAX_SPREAD_PCT_DEFAULT)
        if spread_pct > max_spread:
            return EntryVerdict(
                False, 0, f'spread {spread_pct*100:.2f}% > {max_spread*100:.2f}% cap',
                spread_pct=spread_pct,
            )

        # ── Check 2: don't chase a spike ──────────────────────────────────────
        spike_pct = 0.0
        if history and len(history) >= 5:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=CHASE_WINDOW_SECONDS)
            recent = [p for ts, p in history if ts >= cutoff]
            if recent:
                if direction == 'long':
                    spike_pct = (entry_price - min(recent)) / min(recent)
                else:
                    spike_pct = (max(recent) - entry_price) / max(recent)
                if spike_pct > CHASE_MAX_PCT:
                    return EntryVerdict(
                        False, 0,
                        f'chasing {spike_pct*100:.2f}% spike in last {CHASE_WINDOW_SECONDS}s',
                        spike_pct=spike_pct, spread_pct=spread_pct,
                    )

        # ── Check 3: volume must be healthy ───────────────────────────────────
        if not volume_healthy:
            return EntryVerdict(
                False, 0, 'volume unhealthy (low/stale tape)',
                spike_pct=spike_pct, spread_pct=spread_pct,
            )

        # ── Check 4: VWAP distance (informational; only blocks if far AND wrong side) ─
        vwap_dist = 0.0
        if vwap > 0 and entry_price > 0:
            vwap_dist = (entry_price - vwap) / vwap
            if direction == 'long' and vwap_dist > VWAP_FAR_PCT:
                return EntryVerdict(
                    False, 0,
                    f'price {vwap_dist*100:.1f}% above VWAP — extended long entry',
                    vwap_distance_pct=vwap_dist, spread_pct=spread_pct,
                )
            if direction == 'short' and vwap_dist < -VWAP_FAR_PCT:
                return EntryVerdict(
                    False, 0,
                    f'price {abs(vwap_dist)*100:.1f}% below VWAP — extended short entry',
                    vwap_distance_pct=vwap_dist, spread_pct=spread_pct,
                )

        # ── Quality score ─────────────────────────────────────────────────────
        score = 50
        if abs(imbalance) >= 0.4:        score += 20
        elif abs(imbalance) >= 0.2:      score += 10
        if spread_pct < 0.0005:          score += 15
        elif spread_pct < 0.001:         score += 8
        if spike_pct < 0.001:            score += 10
        if abs(vwap_dist) < 0.0015:      score += 5
        score = min(100, score)

        if score < MIN_QUALITY_TO_TRADE:
            return EntryVerdict(
                False, score, f'quality score {score} < {MIN_QUALITY_TO_TRADE} floor',
                spike_pct=spike_pct, vwap_distance_pct=vwap_dist, spread_pct=spread_pct,
            )

        return EntryVerdict(
            True, score, f'ok ({score}/100)',
            spike_pct=spike_pct, vwap_distance_pct=vwap_dist, spread_pct=spread_pct,
        )
