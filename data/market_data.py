"""Real-time and historical market data fetching and caching."""
from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from alpaca.data.requests import (
    StockBarsRequest,
    StockSnapshotRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data.alpaca_client import AlpacaClients, OPTIONS_AVAILABLE

if OPTIONS_AVAILABLE:
    from alpaca.data.requests import OptionChainRequest


@dataclass
class BarCache:
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_updated: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))


@dataclass
class RealtimeQuote:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last

    @property
    def spread_pct(self) -> float:
        if not self.mid:
            return 1.0
        return (self.ask - self.bid) / self.mid


class MarketDataManager:
    """Manages historical bar fetching and intraday VWAP/tape state."""

    def __init__(self):
        self._bar_cache: Dict[str, BarCache] = defaultdict(BarCache)
        self._quotes: Dict[str, RealtimeQuote] = {}
        # Intraday VWAP accumulators: symbol → (cum_tp_vol, cum_vol)
        self._vwap_state: Dict[str, list] = defaultdict(lambda: [0.0, 0.0])
        # Recent trades buffer for tape analysis: symbol → deque of (price, size)
        self.trade_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._orb: Dict[str, dict] = {}  # opening range high/low per symbol

    # ── Historical bars ───────────────────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute),
        lookback_days: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        cache = self._bar_cache[symbol]
        age = (datetime.now(timezone.utc) - cache.last_updated).total_seconds()
        if not force_refresh and age < 60 and not cache.bars.empty:
            return cache.bars

        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            feed="iex",
        )
        bars_resp = AlpacaClients.stock_hist().get_stock_bars(req)
        df = bars_resp.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol") if symbol in df.index.get_level_values("symbol") else pd.DataFrame()

        if df.empty:
            return df

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = self._add_indicators(df)

        cache.bars = df
        cache.last_updated = datetime.now(timezone.utc)
        return df

    def get_daily_bars(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        return self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), lookback_days)

    # ── Snapshot / quote ─────────────────────────────────────────────────────

    def get_snapshot(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        req = StockSnapshotRequest(symbol_or_symbols=symbols, feed="iex")
        snaps = AlpacaClients.stock_hist().get_stock_snapshot(req)
        quotes = {}
        for sym, snap in snaps.items():
            q = RealtimeQuote(symbol=sym)
            if snap.latest_quote:
                q.bid = snap.latest_quote.bid_price or 0.0
                q.ask = snap.latest_quote.ask_price or 0.0
            if snap.latest_trade:
                q.last = snap.latest_trade.price or 0.0
                q.volume = snap.latest_trade.size or 0
            self._quotes[sym] = q
            quotes[sym] = q
        return quotes

    # ── VWAP ─────────────────────────────────────────────────────────────────

    def update_vwap(self, symbol: str, price: float, size: int) -> float:
        state = self._vwap_state[symbol]
        state[0] += price * size
        state[1] += size
        return state[0] / state[1] if state[1] else price

    def get_vwap(self, symbol: str) -> float:
        state = self._vwap_state[symbol]
        return state[0] / state[1] if state[1] else 0.0

    def reset_vwap(self, symbol: str):
        self._vwap_state[symbol] = [0.0, 0.0]

    # ── Opening range ────────────────────────────────────────────────────────

    def set_orb(self, symbol: str, high: float, low: float):
        self._orb[symbol] = {'high': high, 'low': low, 'range': high - low}

    def get_orb(self, symbol: str) -> Optional[dict]:
        return self._orb.get(symbol)

    def compute_orb_from_bars(self, symbol: str, bars: pd.DataFrame) -> Optional[dict]:
        today = datetime.now(timezone.utc).date()
        open_time = datetime(today.year, today.month, today.day, 9, 30, tzinfo=timezone.utc)
        orb_end = open_time + timedelta(minutes=config.ORB_CAPTURE_MINUTES)

        today_bars = bars[(bars.index >= open_time) & (bars.index < orb_end)]
        if today_bars.empty:
            return None

        high = today_bars['high'].max()
        low = today_bars['low'].min()
        self.set_orb(symbol, high, low)
        return self._orb[symbol]

    # ── Tape feed ────────────────────────────────────────────────────────────

    def record_trade(self, symbol: str, price: float, size: int, conditions: list = None):
        self.trade_buffer[symbol].append({
            'price': price,
            'size': size,
            'ts': datetime.now(timezone.utc),
            'conditions': conditions or [],
        })
        self.update_vwap(symbol, price, size)

    # ── Indicators ───────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 2:
            return df

        # VWAP from bars (cumulative within dataset — approximation)
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - 100 / (1 + rs)

        # ATR
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(config.ATR_PERIOD).mean()

        # Relative volume
        df['vol_ma'] = df['volume'].rolling(config.VOLUME_MA_PERIOD).mean()
        df['rvol'] = df['volume'] / df['vol_ma'].replace(0, np.nan)

        return df

    # ── Options chain ────────────────────────────────────────────────────────

    def get_option_chain(self, symbol: str) -> Optional[dict]:
        client = AlpacaClients.option_hist()
        if client is None:
            return None
        try:
            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=date.today() + timedelta(days=config.OPT_DTE_MIN),
                expiration_date_lte=date.today() + timedelta(days=config.OPT_DTE_MAX),
            )
            return client.get_option_chain(req)
        except Exception:
            return None
