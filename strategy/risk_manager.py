"""
Risk manager: position sizing, daily loss circuit breaker, trade validation.

Rules:
  • Never risk more than MAX_RISK_PER_TRADE_PCT of account equity per trade.
  • Stop all new trades if daily P&L reaches -MAX_DAILY_LOSS_PCT.
  • Cap concurrent positions at MAX_CONCURRENT_POSITIONS.
  • Require minimum 1.5:1 reward-to-risk.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class SizeResult:
    shares: int
    dollar_risk: float
    stop_price: float
    target_price: float
    reward_risk: float
    valid: bool
    reason: str = ''


class RiskManager:
    def __init__(self, account_size: float = config.ACCOUNT_SIZE):
        self.account_size = account_size
        self.session_peak_equity: float = account_size  # tracks high-water mark
        self.daily_pnl: float = 0.0
        self.open_positions: int = 0
        self.macro_risk_multiplier: float = 1.0   # set by macro analysis
        self.macro_halt_reason: str = ''           # non-empty halts all trades

    def update_account(self, equity: float):
        self.account_size = equity
        # Track session high-water mark for drawdown circuit breaker
        if equity > self.session_peak_equity:
            self.session_peak_equity = equity

    def record_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def reset_daily(self):
        self.daily_pnl = 0.0

    @property
    def daily_loss_limit(self) -> float:
        return self.account_size * config.MAX_DAILY_LOSS_PCT

    @property
    def max_risk_dollars(self) -> float:
        """Base risk × macro multiplier (smaller on FOMC days, tariff news, etc.)."""
        base = self.account_size * config.MAX_RISK_PER_TRADE_PCT
        if config.REDUCE_SIZE_ON_HIGH_RISK:
            return base * self.macro_risk_multiplier
        return base

    def is_trading_allowed(self) -> tuple[bool, str]:
        if self.macro_halt_reason:
            return False, f'Macro halt: {self.macro_halt_reason}'
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, f'Daily loss limit hit ({self.daily_pnl:.2f})'
        # Drawdown-from-peak circuit breaker: never let losses run away
        max_dd = getattr(config, 'MAX_DRAWDOWN_FROM_PEAK_PCT', 0.03)
        dd = (self.session_peak_equity - self.account_size) / self.session_peak_equity if self.session_peak_equity else 0
        if dd >= max_dd:
            return False, f'Drawdown {dd*100:.1f}% from peak (cap {max_dd*100:.1f}%)'
        if self.open_positions >= config.MAX_CONCURRENT_POSITIONS:
            return False, f'Max positions ({config.MAX_CONCURRENT_POSITIONS}) reached'
        return True, ''

    def size_stock_trade(
        self,
        entry: float,
        stop: float,
        target: float,
        direction: str,
    ) -> SizeResult:
        allowed, reason = self.is_trading_allowed()
        if not allowed:
            return SizeResult(0, 0, stop, target, 0, False, reason)

        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return SizeResult(0, 0, stop, target, 0, False, 'Zero stop distance')

        target_distance = abs(target - entry)
        rr = target_distance / stop_distance
        if rr < 1.5:
            return SizeResult(0, 0, stop, target, rr, False, f'R:R too low ({rr:.2f})')

        max_risk = self.max_risk_dollars
        raw_shares = int(max_risk / stop_distance)

        # Don't allocate more than 30% of account in one position
        max_position_value = self.account_size * 0.30
        max_shares_by_size = int(max_position_value / entry) if entry > 0 else 0
        shares = min(raw_shares, max_shares_by_size)

        if shares <= 0:
            return SizeResult(0, 0, stop, target, rr, False, 'Position too small')

        dollar_risk = shares * stop_distance
        return SizeResult(
            shares=shares,
            dollar_risk=dollar_risk,
            stop_price=stop,
            target_price=target,
            reward_risk=rr,
            valid=True,
        )

    def size_options_trade(
        self,
        premium: float,
        contracts: int = 1,
    ) -> tuple[int, bool, str]:
        """
        For options we risk the entire premium paid.
        Returns (contracts, valid, reason).
        """
        allowed, reason = self.is_trading_allowed()
        if not allowed:
            return 0, False, reason

        cost_per_contract = premium * 100
        max_risk = self.max_risk_dollars
        max_contracts = int(max_risk / cost_per_contract) if cost_per_contract > 0 else 0

        if max_contracts <= 0:
            return 0, False, f'Premium ${premium:.2f} exceeds risk budget'

        actual = min(contracts, max_contracts)
        return actual, True, ''

    def stop_for_long(self, entry: float, atr: float) -> float:
        return entry - atr * config.ATR_STOP_MULT

    def target_for_long(self, entry: float, atr: float) -> float:
        return entry + atr * config.ATR_TARGET_MULT

    def stop_for_short(self, entry: float, atr: float) -> float:
        return entry + atr * config.ATR_STOP_MULT

    def target_for_short(self, entry: float, atr: float) -> float:
        return entry - atr * config.ATR_TARGET_MULT
