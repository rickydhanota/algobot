"""
Options strategy engine.

Identifies unusual options activity (large sweeps, high V/OI) and scores
candidate contracts for directional plays.

Produces OptionsSetup objects compatible with the order manager.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import config
from signals.tape_reader import TapeSignal
from signals.technical import TechnicalSignal


@dataclass
class OptionsSetup:
    underlying: str
    option_symbol: str
    option_type: str          # 'call' | 'put'
    strike: float
    expiry: date
    dte: int
    direction: str            # 'buy' | 'sell_to_open' (we mostly buy for defined risk)
    premium: float            # per-share mid price
    bid: float
    ask: float
    delta: float
    iv: float
    volume: int
    open_interest: int
    voi_ratio: float          # volume / OI — unusual activity indicator
    score: int
    underlying_score: int     # score from underlying tape + technicals
    contracts: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: List[str] = field(default_factory=list)

    @property
    def cost_per_contract(self) -> float:
        return self.premium * 100

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.premium if self.premium else 1.0

    @property
    def is_valid(self) -> bool:
        return (
            self.score >= config.MIN_SIGNAL_SCORE
            and self.spread_pct <= config.OPT_MAX_SPREAD_PCT
            and config.OPT_DTE_MIN <= self.dte <= config.OPT_DTE_MAX
            and config.OPT_DELTA_MIN <= abs(self.delta) <= config.OPT_DELTA_MAX
        )


class OptionsStrategy:
    """
    Scans option chains for unusual activity and evaluates whether to trade.
    Requires the underlying to also have a bullish/bearish technical + tape signal.
    """

    def evaluate(
        self,
        underlying: str,
        chain: Optional[dict],
        tech: TechnicalSignal,
        tape: Optional[TapeSignal],
    ) -> Optional[OptionsSetup]:
        if chain is None:
            return None

        # Determine directional bias from underlying signals
        direction, underlying_score = self._underlying_bias(tech, tape)
        if underlying_score < 50:
            return None  # underlying signal too weak to trade options

        # Find best contract
        opt_type = 'call' if direction == 'long' else 'put'
        best = self._select_contract(underlying, chain, opt_type, direction, underlying_score)
        return best

    def _underlying_bias(
        self, tech: TechnicalSignal, tape: Optional[TapeSignal]
    ) -> tuple[str, int]:
        score = 0
        direction = 'long'

        # Tape contribution (0-40)
        if tape:
            if tape.direction == 'buy':
                score += int(tape.score_component * 1.3)
                direction = 'long'
            elif tape.direction == 'sell':
                score += int(tape.score_component * 1.3)
                direction = 'short'

        # Technical contribution (0-60)
        if tech:
            tech_score = tech.total_technical_score(
                'buy' if direction == 'long' else 'sell'
            )
            score += tech_score

        return direction, min(score, 100)

    def _select_contract(
        self,
        underlying: str,
        chain: dict,
        opt_type: str,
        direction: str,
        underlying_score: int,
    ) -> Optional[OptionsSetup]:
        today = date.today()
        candidates = []

        for symbol, snap in chain.items():
            try:
                details = snap.greeks if hasattr(snap, 'greeks') else None
                if details is None:
                    continue

                delta = abs(details.delta or 0)
                iv = details.implied_volatility or 0
                expiry = snap.details.expiry_date if hasattr(snap, 'details') else None

                if expiry is None:
                    continue

                dte = (expiry - today).days
                if not (config.OPT_DTE_MIN <= dte <= config.OPT_DTE_MAX):
                    continue
                if not (config.OPT_DELTA_MIN <= delta <= config.OPT_DELTA_MAX):
                    continue

                # Check call vs put
                contract_type = 'call' if 'C' in symbol else 'put'
                if contract_type != opt_type:
                    continue

                quote = snap.latest_quote if hasattr(snap, 'latest_quote') else None
                if quote is None:
                    continue

                bid = quote.bid_price or 0
                ask = quote.ask_price or 0
                premium = (bid + ask) / 2 if bid and ask else 0
                if premium <= 0:
                    continue

                spread_pct = (ask - bid) / premium if premium else 1.0
                if spread_pct > config.OPT_MAX_SPREAD_PCT:
                    continue

                volume = snap.daily_bar.volume if hasattr(snap, 'daily_bar') and snap.daily_bar else 0
                oi = snap.greeks.open_interest if hasattr(snap.greeks, 'open_interest') else 1
                voi = volume / oi if oi else 0

                strike = snap.details.strike_price if hasattr(snap, 'details') else 0

                score = self._score_contract(
                    delta=delta,
                    dte=dte,
                    iv=iv,
                    voi=voi,
                    spread_pct=spread_pct,
                    underlying_score=underlying_score,
                )

                candidates.append({
                    'symbol': symbol,
                    'strike': strike,
                    'expiry': expiry,
                    'dte': dte,
                    'premium': premium,
                    'bid': bid,
                    'ask': ask,
                    'delta': delta,
                    'iv': iv,
                    'volume': volume,
                    'oi': oi,
                    'voi': voi,
                    'score': score,
                })

            except Exception:
                continue

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x['score'])
        if best['score'] < config.MIN_SIGNAL_SCORE:
            return None

        notes = [
            f'V/OI={best["voi"]:.1f}',
            f'delta={best["delta"]:.2f}',
            f'DTE={best["dte"]}',
            f'spread={best["bid"]:.2f}/{best["ask"]:.2f}',
        ]

        return OptionsSetup(
            underlying=underlying,
            option_symbol=best['symbol'],
            option_type=opt_type,
            strike=best['strike'],
            expiry=best['expiry'],
            dte=best['dte'],
            direction='buy',
            premium=best['premium'],
            bid=best['bid'],
            ask=best['ask'],
            delta=best['delta'],
            iv=best['iv'],
            volume=best['volume'],
            open_interest=best['oi'],
            voi_ratio=best['voi'],
            score=best['score'],
            underlying_score=underlying_score,
            notes=notes,
        )

    def _score_contract(
        self,
        delta: float,
        dte: int,
        iv: float,
        voi: float,
        spread_pct: float,
        underlying_score: int,
    ) -> int:
        score = 0

        # Underlying direction quality (0-35)
        score += int(underlying_score * 0.35)

        # V/OI unusual activity (0-25)
        if voi >= config.OPT_VOI_THRESHOLD * 2:
            score += 25
        elif voi >= config.OPT_VOI_THRESHOLD:
            score += 18
        elif voi >= 1.0:
            score += 8

        # Delta sweet spot 0.30-0.45 (0-20)
        if 0.30 <= delta <= 0.45:
            score += 20
        elif 0.25 <= delta <= 0.50:
            score += 12

        # Spread tightness (0-10)
        if spread_pct < 0.03:
            score += 10
        elif spread_pct < 0.06:
            score += 6
        elif spread_pct < 0.10:
            score += 2

        # IV rank: buy low IV (0-10)
        # IV as decimal; rough IVR approximation using current IV vs 0.20 baseline
        if iv < 0.30:
            score += 10  # low IV, cheap options
        elif iv < 0.50:
            score += 5

        return min(score, 100)
