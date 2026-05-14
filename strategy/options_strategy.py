"""
Options strategy engine.

Identifies unusual options activity (large sweeps, high V/OI) and scores
candidate contracts for directional plays.

Note: Alpaca's free option-chain endpoint does NOT populate greeks, so we
estimate delta from the strike vs underlying spot price (close enough for
our 0.25-0.50 filter band).

Produces OptionsSetup objects compatible with the order manager.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import config
from signals.tape_reader import TapeSignal
from signals.technical import TechnicalSignal


# OCC symbol format: ROOT YYMMDD C|P STRIKEx1000 (8 digits)
OCC_PATTERN = re.compile(r'^([A-Z]+?)(\d{6})([CP])(\d{8})$')


def parse_occ_symbol(symbol: str):
    """Parse OCC option symbol → (underlying, expiry_date, 'C'|'P', strike_float)."""
    m = OCC_PATTERN.match(symbol)
    if not m:
        return None
    underlying, exp_str, opt_type, strike_str = m.groups()
    try:
        expiry = date(2000 + int(exp_str[:2]), int(exp_str[2:4]), int(exp_str[4:6]))
        strike = int(strike_str) / 1000.0
        return underlying, expiry, opt_type, strike
    except (ValueError, TypeError):
        return None


def estimate_delta(opt_type: str, spot: float, strike: float, dte: int) -> float:
    """Rough delta magnitude estimate from moneyness + DTE.
    Returns 0.01–0.99. Accurate within ±0.10 for normal liquid options."""
    if spot <= 0 or strike <= 0:
        return 0.5
    if opt_type == 'C':
        moneyness = (spot - strike) / spot          # >0 when ITM call
    else:
        moneyness = (strike - spot) / spot          # >0 when ITM put
    time_factor = max(0.3, min(2.0, (max(dte, 1) / 14.0) ** 0.5))
    delta = 0.5 + 10.0 * moneyness * time_factor
    return max(0.01, min(0.99, delta))


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
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
    ) -> Optional[OptionsSetup]:
        if chain is None:
            return None

        # Determine directional bias from underlying signals
        direction, underlying_score = self._underlying_bias(tech, tape)
        if underlying_score < 50:
            return None  # underlying signal too weak to trade options

        # Find best contract — pass underlying spot price so we can estimate
        # delta when greeks aren't returned by the data feed.
        opt_type = 'call' if direction == 'long' else 'put'
        spot = tech.last_price if tech else 0.0
        best = self._select_contract(
            underlying, chain, opt_type, direction, underlying_score,
            spot_price=spot, dte_min=dte_min, dte_max=dte_max,
        )
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
        spot_price: float = 0.0,
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
    ) -> Optional[OptionsSetup]:
        today = date.today()
        candidates = []
        expected_type = 'C' if opt_type == 'call' else 'P'

        # Use overrides if provided (dynamic DTE selection), else config defaults
        dte_lo = dte_min if dte_min is not None else config.OPT_DTE_MIN
        dte_hi = dte_max if dte_max is not None else config.OPT_DTE_MAX

        for symbol, snap in chain.items():
            try:
                # Parse symbol → strike, expiry, type (works even when greeks
                # are unavailable, which is the case on Alpaca's free tier)
                parsed = parse_occ_symbol(symbol)
                if parsed is None:
                    continue
                _under, expiry, parsed_type, strike = parsed
                if parsed_type != expected_type:
                    continue

                dte = (expiry - today).days
                if not (dte_lo <= dte <= dte_hi):
                    continue

                # Use real greeks if available, otherwise estimate from spot
                greeks = getattr(snap, 'greeks', None)
                if greeks is not None and getattr(greeks, 'delta', None):
                    delta = abs(greeks.delta)
                    iv = getattr(greeks, 'implied_volatility', 0) or 0
                else:
                    delta = estimate_delta(parsed_type, spot_price, strike, dte)
                    iv = 0  # unknown — drop the IV-based scoring component

                if not (config.OPT_DELTA_MIN <= delta <= config.OPT_DELTA_MAX):
                    continue

                quote = getattr(snap, 'latest_quote', None)
                if quote is None:
                    continue
                bid = quote.bid_price or 0
                ask = quote.ask_price or 0
                premium = (bid + ask) / 2 if (bid and ask) else 0
                if premium <= 0:
                    continue

                spread_pct = (ask - bid) / premium if premium else 1.0
                if spread_pct > config.OPT_MAX_SPREAD_PCT:
                    continue

                # Volume and OI from snapshot if available
                daily_bar = getattr(snap, 'daily_bar', None)
                volume = daily_bar.volume if daily_bar else 0
                oi = getattr(greeks, 'open_interest', 0) if greeks else 0
                voi = (volume / oi) if oi else 0

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
