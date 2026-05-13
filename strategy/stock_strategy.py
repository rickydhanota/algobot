"""
Stock strategy engine.

Implements three setups, each contributing sub-scores to a composite 0-100:
  1. VWAP Momentum  — stock reclaims / bounces off VWAP with tape confirmation
  2. ORB Breakout   — price breaks opening-range high/low with volume expansion
  3. Volume Surge   — RVOL > 2× with strong directional tape (catch-up move)

Only setups scoring >= MIN_SIGNAL_SCORE are returned as actionable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import config
from signals.tape_reader import TapeSignal
from signals.technical import TechnicalSignal


@dataclass
class StockSetup:
    symbol: str
    direction: str        # 'long' | 'short'
    strategy: str         # 'vwap_momentum' | 'orb_breakout' | 'volume_surge'
    score: int            # 0-100
    entry: float
    stop: float
    target: float
    atr: float
    rvol: float
    tape_imbalance: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: List[str] = field(default_factory=list)

    @property
    def reward_risk(self) -> float:
        if abs(self.entry - self.stop) == 0:
            return 0
        return abs(self.target - self.entry) / abs(self.entry - self.stop)

    @property
    def is_long(self) -> bool:
        return self.direction == 'long'


class StockStrategy:
    """Evaluates a symbol and returns the best setup if one meets the threshold."""

    def evaluate(
        self,
        tech: TechnicalSignal,
        tape: Optional[TapeSignal],
    ) -> Optional[StockSetup]:
        best: Optional[StockSetup] = None

        for setup in [
            self._vwap_momentum(tech, tape),
            self._orb_breakout(tech, tape),
            self._volume_surge(tech, tape),
        ]:
            if setup and setup.score >= config.MIN_SIGNAL_SCORE:
                if best is None or setup.score > best.score:
                    best = setup

        return best

    # ── VWAP Momentum ─────────────────────────────────────────────────────────

    def _vwap_momentum(
        self, tech: TechnicalSignal, tape: Optional[TapeSignal]
    ) -> Optional[StockSetup]:
        price = tech.last_price
        vwap = tech.vwap
        if not vwap:
            return None

        score = 0
        notes = []

        # Tape (0-30)
        if tape:
            tape_score = tape.score_component
            direction = tape.direction
            score += tape_score
            notes.append(f'tape={tape.imbalance:+.2f}')
        else:
            direction = 'buy' if tech.above_vwap else 'sell'

        # VWAP position (0-20)
        if (direction == 'buy' and tech.above_vwap) or (direction == 'sell' and not tech.above_vwap):
            score += 20
            notes.append('VWAP aligned')
        elif tech.vwap_cross:
            score += 12
            notes.append('VWAP cross')
        else:
            return None  # VWAP momentum requires price-VWAP alignment

        # RVOL (0-25)
        score += tech.rvol_score
        if tech.rvol >= config.RVOL_MIN:
            notes.append(f'RVOL={tech.rvol:.1f}×')
        else:
            score -= 10  # penalise weak volume

        # RSI (0-15)
        score += tech.rsi_score
        notes.append(f'RSI={tech.rsi:.0f}')

        # Trend alignment (0-10)
        if direction == 'buy' and tech.trend_up:
            score += 10
            notes.append('trend↑')
        elif direction == 'sell' and not tech.trend_up:
            score += 10
            notes.append('trend↓')

        score = max(0, min(100, score))
        if score < config.MIN_SIGNAL_SCORE:
            return None

        trade_dir = 'long' if direction == 'buy' else 'short'
        entry = price
        if trade_dir == 'long':
            stop = tech.stop_distance and (entry - tech.stop_distance) or (vwap * 0.998)
            target = entry + tech.target_distance
        else:
            stop = entry + (tech.stop_distance or entry * 0.015)
            target = entry - (tech.target_distance or entry * 0.025)

        return StockSetup(
            symbol=tech.symbol,
            direction=trade_dir,
            strategy='vwap_momentum',
            score=score,
            entry=entry,
            stop=round(stop, 2),
            target=round(target, 2),
            atr=tech.atr,
            rvol=tech.rvol,
            tape_imbalance=tape.imbalance if tape else 0.0,
            notes=notes,
        )

    # ── ORB Breakout ──────────────────────────────────────────────────────────

    def _orb_breakout(
        self, tech: TechnicalSignal, tape: Optional[TapeSignal]
    ) -> Optional[StockSetup]:
        if not tech.orb_breakout:
            return None

        score = 0
        notes = [f'ORB {tech.orb_breakout}']

        # ORB base bonus (0-15)
        score += 15

        # Tape must confirm breakout direction
        if tape:
            expected_tape = 'buy' if tech.orb_breakout == 'long' else 'sell'
            if tape.direction == expected_tape:
                score += tape.score_component
                notes.append(f'tape confirms: {tape.imbalance:+.2f}')
            else:
                return None  # ORB with opposing tape = fake breakout, skip

        # RVOL (0-25) — breakouts require volume expansion
        score += tech.rvol_score
        if tech.rvol < config.RVOL_MIN:
            return None  # no volume = no breakout validity
        notes.append(f'RVOL={tech.rvol:.1f}×')

        # RSI not overextended
        score += tech.rsi_score

        score = max(0, min(100, score))
        if score < config.MIN_SIGNAL_SCORE:
            return None

        price = tech.last_price
        is_long = tech.orb_breakout == 'long'
        orb_range = (tech.orb_high - tech.orb_low) if tech.orb_high and tech.orb_low else tech.atr

        if is_long:
            stop = tech.orb_high - orb_range * 0.5   # stop inside ORB
            target = price + orb_range * 1.5
        else:
            stop = tech.orb_low + orb_range * 0.5
            target = price - orb_range * 1.5

        return StockSetup(
            symbol=tech.symbol,
            direction='long' if is_long else 'short',
            strategy='orb_breakout',
            score=score,
            entry=price,
            stop=round(stop, 2),
            target=round(target, 2),
            atr=tech.atr,
            rvol=tech.rvol,
            tape_imbalance=tape.imbalance if tape else 0.0,
            notes=notes,
        )

    # ── Volume Surge ──────────────────────────────────────────────────────────

    def _volume_surge(
        self, tech: TechnicalSignal, tape: Optional[TapeSignal]
    ) -> Optional[StockSetup]:
        if tech.rvol < 2.0:
            return None
        if tape is None or not tape.is_moderate:
            return None

        score = 0
        notes = [f'VolSurge RVOL={tech.rvol:.1f}×']

        # Volume (25 max, needs 2× minimum)
        score += tech.rvol_score

        # Tape direction (0-30)
        score += tape.score_component
        notes.append(f'tape={tape.imbalance:+.2f}')

        # RSI momentum
        score += tech.rsi_score

        # Trend alignment
        direction = tape.direction
        if direction == 'buy' and tech.trend_up:
            score += 10
        elif direction == 'sell' and not tech.trend_up:
            score += 10

        score = max(0, min(100, score))
        if score < config.MIN_SIGNAL_SCORE:
            return None

        price = tech.last_price
        trade_dir = 'long' if direction == 'buy' else 'short'

        if trade_dir == 'long':
            stop = price - tech.stop_distance
            target = price + tech.target_distance
        else:
            stop = price + tech.stop_distance
            target = price - tech.target_distance

        return StockSetup(
            symbol=tech.symbol,
            direction=trade_dir,
            strategy='volume_surge',
            score=score,
            entry=price,
            stop=round(stop, 2),
            target=round(target, 2),
            atr=tech.atr,
            rvol=tech.rvol,
            tape_imbalance=tape.imbalance,
            notes=notes,
        )
