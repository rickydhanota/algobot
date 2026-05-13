"""
Unified intel aggregator: combines 13F holdings and congressional trades
into a per-symbol view with directional bias for the strategy layer.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from intel.political_trades import PoliticalTradesFeed
from intel.sec_edgar import Form13FFetcher

log = logging.getLogger(__name__)


@dataclass
class SymbolIntel:
    symbol: str
    fund_count: int = 0          # 13F funds holding it
    fund_total_value: int = 0    # combined position value (USD)
    top_funds: List[str] = None  # top fund names
    political_buys: int = 0
    political_sells: int = 0
    political_bias: int = 0      # -1 / 0 / +1
    political_label: str = ''

    @property
    def has_smart_money(self) -> bool:
        return self.fund_count > 0 or (self.political_buys + self.political_sells) > 0

    @property
    def score_boost(self) -> int:
        """Small directional bonus when smart money aligns with trade direction."""
        boost = 0
        if self.fund_count >= 3:    boost += 3
        elif self.fund_count >= 1:  boost += 1
        if self.political_bias > 0: boost += 2
        return boost

    def to_dict(self) -> dict:
        return {
            'symbol':            self.symbol,
            'fund_count':        self.fund_count,
            'fund_total_value':  self.fund_total_value,
            'top_funds':         self.top_funds or [],
            'political_buys':    self.political_buys,
            'political_sells':   self.political_sells,
            'political_bias':    self.political_bias,
            'political_label':   self.political_label,
        }


class IntelAggregator:
    def __init__(self):
        self.political = PoliticalTradesFeed()
        self.thirteenf = Form13FFetcher()
        self._symbol_cache: Dict[str, SymbolIntel] = {}
        self._last_refresh: Optional[datetime] = None

    def refresh(self, watchlist: List[str]):
        log.info('Refreshing intel sources (13F + congressional)...')
        self.political.refresh(lookback_days=60)
        self.thirteenf.refresh(max_funds=None)
        self._rebuild_cache(watchlist)
        self._last_refresh = datetime.now()

    def _rebuild_cache(self, watchlist: List[str]):
        self._symbol_cache.clear()
        for sym in watchlist:
            funds = self.thirteenf.funds_holding(sym)
            buys = sum(1 for t in self.political.for_symbol(sym, 999) if t['side'] == 'buy')
            sells = sum(1 for t in self.political.for_symbol(sym, 999) if t['side'] == 'sell')
            bias, label = self.political.directional_bias(sym)

            self._symbol_cache[sym] = SymbolIntel(
                symbol=sym,
                fund_count=len(funds),
                fund_total_value=sum(f['value_usd'] for f in funds),
                top_funds=[f['fund'] for f in funds[:3]],
                political_buys=buys,
                political_sells=sells,
                political_bias=bias,
                political_label=label,
            )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def for_symbol(self, symbol: str) -> Optional[SymbolIntel]:
        return self._symbol_cache.get(symbol)

    def score_boost(self, symbol: str, direction: str) -> int:
        """Returns 0–5 point bonus when smart money aligns with trade direction."""
        intel = self.for_symbol(symbol)
        if not intel:
            return 0
        boost = intel.score_boost

        # Reverse if politicians are selling but we're going long
        if direction == 'long' and intel.political_bias < 0:
            boost = max(0, boost - 2)
        if direction == 'short' and intel.political_bias > 0:
            boost = max(0, boost - 2)
        return boost

    def dashboard_data(self) -> dict:
        """Compact summary for HTML dashboard."""
        return {
            'last_refresh':    self._last_refresh.isoformat() if self._last_refresh else None,
            'tracked_funds':   list(self.thirteenf.all_holdings().keys()),
            'recent_political': self.political.recent(limit=20),
            'symbol_intel':    [s.to_dict() for s in self._symbol_cache.values() if s.has_smart_money],
            'top_fund_picks':  self.thirteenf.watchlist_summary(list(self._symbol_cache.keys())),
        }
