"""
Order manager: places, tracks, and exits orders via Alpaca.

Supports:
  - Bracketed stock orders (entry + stop-loss + take-profit in one request)
  - Options market orders with manual stop tracking
  - Position monitoring and trailing logic
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

from data.alpaca_client import AlpacaClients
from strategy.stock_strategy import StockSetup
from strategy.options_strategy import OptionsSetup
from strategy.risk_manager import RiskManager

log = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    trade_id: str
    symbol: str
    asset_type: str           # 'stock' | 'option'
    direction: str
    strategy: str
    entry_price: float
    stop_price: float
    target_price: float
    shares: int               # shares for stocks, contracts×100 for options
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None
    status: str = 'open'      # 'open' | 'closed' | 'stopped' | 'target' | 'eod_close'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    # Options-specific
    option_type: Optional[str] = None    # 'call' | 'put'
    strike: Optional[float] = None
    contracts: Optional[int] = None      # number of contracts

    @property
    def unrealized_pnl(self) -> float:
        if not self.entry_price:
            return 0.0
        return 0.0  # populated externally from live price

    def close(self, exit_price: float, reason: str = 'manual'):
        self.exit_price = exit_price
        self.exit_time = datetime.now(timezone.utc)
        self.status = reason
        mult = 1 if self.direction == 'long' else -1
        self.realized_pnl = (exit_price - self.entry_price) * self.shares * mult


class OrderManager:
    def __init__(self, risk_manager: RiskManager):
        self.risk = risk_manager
        self._client = AlpacaClients.trading()
        self.active: Dict[str, ActiveTrade] = {}
        self._trade_counter = 0

    def _next_id(self) -> str:
        self._trade_counter += 1
        return f'T{self._trade_counter:04d}'

    # ── Stock orders ─────────────────────────────────────────────────────────

    def place_stock_trade(self, setup: StockSetup) -> Optional[ActiveTrade]:
        size = self.risk.size_stock_trade(
            entry=setup.entry,
            stop=setup.stop,
            target=setup.target,
            direction=setup.direction,
        )
        if not size.valid:
            log.warning(f'[{setup.symbol}] Sizing rejected: {size.reason}')
            return None

        side = OrderSide.BUY if setup.is_long else OrderSide.SELL
        stop_price = round(size.stop_price, 2)
        target_price = round(size.target_price, 2)

        try:
            req = MarketOrderRequest(
                symbol=setup.symbol,
                qty=size.shares,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=target_price),
            )
            order = self._client.submit_order(req)
            trade_id = self._next_id()
            trade = ActiveTrade(
                trade_id=trade_id,
                symbol=setup.symbol,
                asset_type='stock',
                direction=setup.direction,
                strategy=setup.strategy,
                entry_price=setup.entry,
                stop_price=stop_price,
                target_price=target_price,
                shares=size.shares,
                order_id=str(order.id),
            )
            self.active[trade_id] = trade
            self.risk.open_positions += 1
            log.info(
                f'[{setup.symbol}] {setup.direction.upper()} {size.shares}sh '
                f'entry={setup.entry:.2f} stop={stop_price:.2f} '
                f'target={target_price:.2f} RR={size.reward_risk:.1f}'
            )
            return trade

        except Exception as e:
            log.error(f'[{setup.symbol}] Order failed: {e}')
            return None

    # ── Options orders ────────────────────────────────────────────────────────

    def place_options_trade(self, setup: OptionsSetup) -> Optional[ActiveTrade]:
        contracts, valid, reason = self.risk.size_options_trade(setup.premium)
        if not valid:
            log.warning(f'[{setup.underlying}] Options sizing rejected: {reason}')
            return None

        setup.contracts = contracts

        try:
            req = MarketOrderRequest(
                symbol=setup.option_symbol,
                qty=contracts,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            trade_id = self._next_id()

            # For options: stop at 50% of premium, target at 100% gain
            stop_price = round(setup.premium * 0.50, 2)
            target_price = round(setup.premium * 2.00, 2)

            trade = ActiveTrade(
                trade_id=trade_id,
                symbol=setup.option_symbol,
                asset_type='option',
                direction=setup.direction,
                strategy=f'options_{setup.option_type}',
                entry_price=setup.premium,
                stop_price=stop_price,
                target_price=target_price,
                shares=contracts * 100,
                order_id=str(order.id),
                option_type=setup.option_type,
                strike=setup.strike,
                contracts=contracts,
            )
            self.active[trade_id] = trade
            self.risk.open_positions += 1
            log.info(
                f'[{setup.underlying}] OPTIONS {setup.option_type.upper()} '
                f'{contracts}× {setup.option_symbol} @ {setup.premium:.2f} '
                f'stop={stop_price:.2f} target={target_price:.2f}'
            )
            return trade

        except Exception as e:
            log.error(f'[{setup.underlying}] Options order failed: {e}')
            return None

    # ── Position management ───────────────────────────────────────────────────

    def check_exits(self, live_prices: Dict[str, float]):
        """Check open positions against stop/target and close if hit."""
        for trade_id, trade in list(self.active.items()):
            if trade.status != 'open':
                continue
            price = live_prices.get(trade.symbol)
            if not price:
                continue

            hit_stop = (
                (trade.direction in ('long', 'buy') and price <= trade.stop_price) or
                (trade.direction in ('short', 'sell') and price >= trade.stop_price)
            )
            hit_target = (
                (trade.direction in ('long', 'buy') and price >= trade.target_price) or
                (trade.direction in ('short', 'sell') and price <= trade.target_price)
            )

            if hit_target:
                self._close_trade(trade, price, 'target')
            elif hit_stop:
                self._close_trade(trade, price, 'stopped')

    def _close_trade(self, trade: ActiveTrade, price: float, reason: str):
        try:
            self._client.close_position(trade.symbol)
            trade.close(price, reason)
            self.risk.open_positions = max(0, self.risk.open_positions - 1)
            self.risk.record_daily_pnl(trade.realized_pnl or 0)
            result = '✓ WIN' if (trade.realized_pnl or 0) > 0 else '✗ LOSS'
            log.info(
                f'[{trade.symbol}] CLOSE {reason} @ {price:.2f} '
                f'P&L={trade.realized_pnl:.2f} {result}'
            )
        except Exception as e:
            log.error(f'[{trade.symbol}] Close failed: {e}')

    def close_all(self):
        """EOD: close all open positions."""
        for trade in self.active.values():
            if trade.status == 'open':
                try:
                    self._client.close_position(trade.symbol)
                    trade.status = 'eod_close'
                    self.risk.open_positions = max(0, self.risk.open_positions - 1)
                except Exception as e:
                    log.error(f'close_all [{trade.symbol}]: {e}')

    def get_open_trades(self) -> List[ActiveTrade]:
        return [t for t in self.active.values() if t.status == 'open']

    def get_closed_trades(self) -> List[ActiveTrade]:
        return [t for t in self.active.values() if t.status != 'open']
