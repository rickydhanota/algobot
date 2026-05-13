"""
Tape reader: interprets real-time trade prints to gauge order-flow pressure.

Key metrics produced per symbol:
  imbalance   -1.0 → +1.0  (negative = seller aggression, positive = buyer)
  direction   'buy' | 'sell' | 'neutral'
  strength    0.0 → 1.0
  large_print_ratio  fraction of volume in large prints
  uptick_ratio       fraction of ticks that are upticks
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import config


@dataclass
class TapeSignal:
    symbol: str
    direction: str          # 'buy' | 'sell' | 'neutral'
    imbalance: float        # -1 → +1
    strength: float         # 0 → 1
    large_print_ratio: float
    uptick_ratio: float
    avg_trade_size: float
    total_volume: int
    sample_count: int

    @property
    def score_component(self) -> int:
        """0-30 contribution to overall signal score."""
        s = abs(self.imbalance)
        if s > 0.6:
            base = 30
        elif s > 0.40:
            base = 22
        elif s > 0.20:
            base = 14
        else:
            base = 5
        # Boost for large-print confirmation
        if self.large_print_ratio > 0.15:
            base = min(30, base + 5)
        return base

    @property
    def is_strong(self) -> bool:
        return abs(self.imbalance) >= config.IMBALANCE_STRONG

    @property
    def is_moderate(self) -> bool:
        return abs(self.imbalance) >= config.IMBALANCE_MODERATE


class TapeReader:
    """
    Stateful per-symbol tape analyzer.
    Feed it trades via `record_trade`; call `analyze` to get a TapeSignal.
    """

    def __init__(self):
        # symbol → deque of trade dicts
        self._buffer: Dict[str, deque] = {}
        # symbol → last trade price (for tick-rule classification)
        self._last_price: Dict[str, float] = {}
        # symbol → last classified direction (for zero-tick rule)
        self._last_direction: Dict[str, str] = {}

    def _get_buffer(self, symbol: str) -> deque:
        if symbol not in self._buffer:
            self._buffer[symbol] = deque(maxlen=500)
        return self._buffer[symbol]

    def record_trade(
        self,
        symbol: str,
        price: float,
        size: int,
        bid: float = 0.0,
        ask: float = 0.0,
        conditions: list = None,
    ):
        direction = self._classify(symbol, price, bid, ask)
        self._get_buffer(symbol).append({
            'price': price,
            'size': size,
            'direction': direction,
            'conditions': conditions or [],
        })

    def _classify(self, symbol: str, price: float, bid: float, ask: float) -> str:
        """Tick-rule + quote-rule classification."""
        # Quote rule: trade at ask = buy, at bid = sell
        if ask and bid:
            mid = (bid + ask) / 2
            if price >= ask:
                direction = 'buy'
            elif price <= bid:
                direction = 'sell'
            elif price > mid:
                direction = 'buy'
            elif price < mid:
                direction = 'sell'
            else:
                direction = self._tick_rule(symbol, price)
        else:
            direction = self._tick_rule(symbol, price)

        self._last_price[symbol] = price
        self._last_direction[symbol] = direction
        return direction

    def _tick_rule(self, symbol: str, price: float) -> str:
        prev = self._last_price.get(symbol)
        if prev is None:
            return 'neutral'
        if price > prev:
            return 'buy'
        if price < prev:
            return 'sell'
        # Zero tick — inherit last direction
        return self._last_direction.get(symbol, 'neutral')

    def analyze(self, symbol: str, min_trades: int = 20) -> Optional[TapeSignal]:
        buf = list(self._get_buffer(symbol))
        if len(buf) < min_trades:
            return None

        buy_vol = sum(t['size'] for t in buf if t['direction'] == 'buy')
        sell_vol = sum(t['size'] for t in buf if t['direction'] == 'sell')
        total_vol = buy_vol + sell_vol or 1

        sizes = [t['size'] for t in buf]
        avg_size = np.mean(sizes) if sizes else 1
        large_threshold = avg_size * config.LARGE_PRINT_MULTIPLIER
        large_vol = sum(t['size'] for t in buf if t['size'] >= large_threshold)

        upticks = sum(1 for t in buf if t['direction'] == 'buy')

        imbalance = (buy_vol - sell_vol) / total_vol
        direction = 'buy' if imbalance > 0.05 else ('sell' if imbalance < -0.05 else 'neutral')
        strength = min(1.0, abs(imbalance) / 0.6)

        return TapeSignal(
            symbol=symbol,
            direction=direction,
            imbalance=imbalance,
            strength=strength,
            large_print_ratio=large_vol / total_vol,
            uptick_ratio=upticks / len(buf),
            avg_trade_size=avg_size,
            total_volume=total_vol,
            sample_count=len(buf),
        )

    def clear(self, symbol: str):
        if symbol in self._buffer:
            self._buffer[symbol].clear()

    def clear_all(self):
        for buf in self._buffer.values():
            buf.clear()
        self._last_price.clear()
        self._last_direction.clear()
