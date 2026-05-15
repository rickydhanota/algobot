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
    contracts: Optional[int] = None      # CURRENT remaining contracts
    # Trimming state (populated by ExitManager)
    original_contracts: Optional[int] = None    # at entry
    tier_hits: List[int] = field(default_factory=list)
    partial_exits: List[dict] = field(default_factory=list)
    realized_partial_pnl: float = 0.0

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
        contracts, valid, reason = self.risk.size_options_trade(
            setup.premium, score=getattr(setup, 'score', 75),
        )
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
                original_contracts=contracts,
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

    def _partial_close_option(self, trade: ActiveTrade, qty: int, reason: str):
        """Partially close an options position (sell `qty` contracts)."""
        if trade.asset_type != 'option' or trade.contracts is None:
            return
        qty = min(qty, trade.contracts)
        if qty <= 0:
            return
        try:
            req = MarketOrderRequest(
                symbol=trade.symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            self._client.submit_order(req)
        except Exception as e:
            log.error(f'[{trade.symbol}] Partial close failed: {e}')
            return

        # Update trade state
        # Estimate exit price from latest price tracked externally; if absent
        # we approximate via the entry+target midpoint. Best-effort book-keeping.
        approx_exit = trade.target_price if trade.target_price else trade.entry_price * 1.20
        partial_pnl = (approx_exit - trade.entry_price) * qty * 100
        trade.realized_partial_pnl += partial_pnl
        trade.partial_exits.append({
            'qty':       qty,
            'price':     approx_exit,
            'reason':    reason,
            'pnl':       round(partial_pnl, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        trade.contracts -= qty
        trade.shares = trade.contracts * 100
        log.info(
            f'[{trade.symbol}] TRIM {qty} contracts ({reason}) → '
            f'{trade.contracts} remaining; partial P&L ≈ ${partial_pnl:+.2f}'
        )

        # Bank the partial P&L day-level
        self.risk.record_daily_pnl(partial_pnl)

    def close_by_id(
        self,
        trade_id: str,
        qty: Optional[int] = None,
        live_prices: dict = None,
        reason: str = 'manual',
    ) -> tuple[bool, str]:
        """
        Close (partially or fully) an open position by trade_id.

          qty=None or qty >= available → full close
          qty < available              → partial close
          For options, qty is CONTRACTS. For stocks, qty is SHARES.
        """
        trade = self.active.get(trade_id)
        if trade is None:
            return False, f'trade {trade_id} not found'
        if trade.status != 'open':
            return False, f'trade {trade_id} is not open (status={trade.status})'

        if trade.asset_type == 'option':
            available = trade.contracts or 0
            unit = 'contracts'
        else:
            available = trade.shares or 0
            unit = 'shares'

        if available <= 0:
            return False, 'no quantity available'

        price = (live_prices or {}).get(trade.symbol) or trade.entry_price

        # Full close (qty omitted or >= available)
        if qty is None or qty >= available:
            try:
                self._close_trade(trade, price, reason)
                return True, f'closed {available} {unit} of {trade.symbol} @ {price:.2f}'
            except Exception as e:
                return False, f'close failed: {e}'

        # Partial close
        if qty <= 0:
            return False, f'invalid quantity {qty}'

        try:
            if trade.asset_type == 'option':
                self._partial_close_option(trade, qty, f'manual_partial_{reason}')
            else:
                self._partial_close_stock(trade, qty, price, f'manual_partial_{reason}')
            remaining = trade.contracts if trade.asset_type == 'option' else trade.shares
            return True, f'partial close: {qty}/{available} {unit} of {trade.symbol}; {remaining} remaining'
        except Exception as e:
            return False, f'partial close failed: {e}'

    def _partial_close_stock(self, trade: ActiveTrade, qty: int, price: float, reason: str):
        """Sell `qty` shares of an open stock position (or buy-to-cover for shorts)."""
        if trade.asset_type != 'stock':
            return
        qty = min(qty, trade.shares)
        if qty <= 0:
            return
        # Close direction is opposite of open
        side = OrderSide.SELL if trade.direction in ('long', 'buy') else OrderSide.BUY
        try:
            req = MarketOrderRequest(
                symbol=trade.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            self._client.submit_order(req)
        except Exception as e:
            log.error(f'[{trade.symbol}] Partial stock close failed: {e}')
            raise

        mult = 1 if trade.direction in ('long', 'buy') else -1
        partial_pnl = (price - trade.entry_price) * qty * mult
        trade.realized_partial_pnl += partial_pnl
        trade.partial_exits.append({
            'qty':       qty,
            'price':     price,
            'reason':    reason,
            'pnl':       round(partial_pnl, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        trade.shares -= qty
        log.info(
            f'[{trade.symbol}] PARTIAL {qty}sh ({reason}) → '
            f'{trade.shares} remaining; partial P&L ${partial_pnl:+.2f}'
        )
        self.risk.record_daily_pnl(partial_pnl)

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
