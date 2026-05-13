"""
Technical signal generator.

Evaluates a symbol's bar data and real-time VWAP to produce a TechnicalSignal
with sub-scores used by the strategy layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config


@dataclass
class TechnicalSignal:
    symbol: str
    last_price: float
    vwap: float
    rsi: float
    atr: float
    rvol: float
    above_vwap: bool
    vwap_cross: bool          # price just crossed VWAP in last 3 bars
    trend_up: bool
    orb_high: Optional[float]
    orb_low: Optional[float]
    orb_breakout: Optional[str]  # 'long' | 'short' | None
    support: float
    resistance: float

    # ── Scoring helpers ───────────────────────────────────────────────────────

    @property
    def rvol_score(self) -> int:
        """0-25 based on relative volume."""
        if self.rvol >= 3.0:   return 25
        if self.rvol >= 2.0:   return 18
        if self.rvol >= 1.5:   return 12
        if self.rvol >= 1.2:   return 6
        return 0

    @property
    def vwap_score(self) -> int:
        """0-20: reward position relative to VWAP for the expected direction."""
        return 20 if (self.above_vwap or self.vwap_cross) else 0

    @property
    def rsi_score(self) -> int:
        """0-15: RSI in a healthy momentum zone (not overextended)."""
        if 40 <= self.rsi <= 60:   return 15
        if 30 <= self.rsi <= 70:   return 8
        return 0

    @property
    def trend_score(self) -> int:
        """0-10."""
        return 10 if self.trend_up else 0

    @property
    def orb_score(self) -> int:
        """0-15 bonus when price is breaking the opening range."""
        return 15 if self.orb_breakout else 0

    def total_technical_score(self, direction: str) -> int:
        """Aggregate technical sub-scores for a proposed direction."""
        score = self.rvol_score + self.rsi_score
        if direction == 'buy':
            score += self.vwap_score if self.above_vwap else 0
            score += self.trend_score if self.trend_up else 0
            score += self.orb_score if self.orb_breakout == 'long' else 0
        else:
            score += self.vwap_score if not self.above_vwap else 0
            score += self.trend_score if not self.trend_up else 0
            score += self.orb_score if self.orb_breakout == 'short' else 0
        return score

    @property
    def stop_distance(self) -> float:
        return self.atr * config.ATR_STOP_MULT if self.atr else self.last_price * 0.015

    @property
    def target_distance(self) -> float:
        return self.atr * config.ATR_TARGET_MULT if self.atr else self.last_price * 0.025


class TechnicalAnalyzer:
    """Stateless: produces a TechnicalSignal from bar DataFrame + live price."""

    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        live_price: float,
        live_vwap: float,
        orb: Optional[dict] = None,
    ) -> Optional[TechnicalSignal]:
        if bars.empty or len(bars) < config.ATR_PERIOD + 5:
            return None

        last_bar = bars.iloc[-1]
        prev_bar = bars.iloc[-2] if len(bars) > 1 else last_bar

        vwap = live_vwap if live_vwap else last_bar.get('vwap', live_price)
        rsi = float(last_bar.get('rsi', 50))
        atr = float(last_bar.get('atr', live_price * 0.01))
        rvol = float(last_bar.get('rvol', 1.0)) if not np.isnan(last_bar.get('rvol', np.nan)) else 1.0

        above_vwap = live_price > vwap * (1 + config.VWAP_BUFFER_PCT)
        was_above = prev_bar.get('close', live_price) > prev_bar.get('vwap', vwap) * (1 + config.VWAP_BUFFER_PCT)
        vwap_cross = above_vwap != was_above

        # Trend: price above 20-bar simple MA
        if len(bars) >= 20:
            ma20 = bars['close'].iloc[-20:].mean()
            trend_up = live_price > ma20
        else:
            trend_up = live_price > bars['close'].mean()

        # Support / resistance from recent swing highs/lows
        recent = bars.iloc[-20:]
        resistance = float(recent['high'].max())
        support = float(recent['low'].min())

        # ORB breakout check
        orb_breakout = None
        if orb:
            if live_price > orb['high'] * 1.001:
                orb_breakout = 'long'
            elif live_price < orb['low'] * 0.999:
                orb_breakout = 'short'

        return TechnicalSignal(
            symbol=symbol,
            last_price=live_price,
            vwap=vwap,
            rsi=rsi,
            atr=atr,
            rvol=rvol,
            above_vwap=above_vwap,
            vwap_cross=vwap_cross,
            trend_up=trend_up,
            orb_high=orb['high'] if orb else None,
            orb_low=orb['low'] if orb else None,
            orb_breakout=orb_breakout,
            support=support,
            resistance=resistance,
        )
