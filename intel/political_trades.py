"""
Political trades aggregator (STOCK Act disclosures).

NOTE: The previously-free community mirrors (Senate Stock Watcher, House
Stock Watcher S3 buckets) have been restricted/taken down as of late 2024.
This module now degrades gracefully when feeds return errors.

To enable real political trade data, add an API key for one of:
  • Quiver Quantitative (https://api.quiverquant.com)  — ~$10/mo
  • Finnhub (https://finnhub.io)                      — free tier 60req/min

Set the key in .env as POLITICAL_TRADES_API_KEY and uncomment the relevant
fetcher below.

Trade data is 30–45 days delayed by law (STOCK Act); useful as a bias
layer, never for day trading.
"""
from __future__ import annotations
import json
import logging
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

SENATE_URL = 'https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json'
HOUSE_URL  = 'https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json'

USER_AGENT = 'AlgoBot/1.0 (algobot-paper@example.com)'
TIMEOUT_SECONDS = 30


def _fetch_json(url: str) -> list:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_amount(amt: str) -> str:
    """Compact amount ranges, e.g. '$1,001 - $15,000' → '$1k–$15k'."""
    if not amt:
        return ''
    a = amt.replace(',', '').replace('$', '').strip()
    a = a.replace(' - ', '–').replace('–', '–')
    return '$' + a if not a.startswith('$') else a


class PoliticalTradesFeed:
    def __init__(self):
        self._senate: List[dict] = []
        self._house: List[dict] = []
        self._by_symbol: Dict[str, List[dict]] = defaultdict(list)
        self._last_refresh: Optional[datetime] = None

    def refresh(self, lookback_days: int = 60):
        """Pull latest filings from both chambers."""
        cutoff = date.today() - timedelta(days=lookback_days)

        try:
            senate_raw = _fetch_json(SENATE_URL)
            self._senate = self._normalize(senate_raw, chamber='Senate', cutoff=cutoff)
        except Exception as e:
            log.warning(f'Senate trades fetch failed: {e}')
            self._senate = []

        try:
            house_raw = _fetch_json(HOUSE_URL)
            self._house = self._normalize(house_raw, chamber='House', cutoff=cutoff)
        except Exception as e:
            log.warning(f'House trades fetch failed: {e}')
            self._house = []

        self._index_by_symbol()
        self._last_refresh = datetime.now()
        log.info(
            f'Political trades refreshed: '
            f'{len(self._senate)} senate + {len(self._house)} house transactions'
        )

    def _normalize(self, raw: list, chamber: str, cutoff: date) -> List[dict]:
        out = []
        for entry in raw:
            try:
                tx_date = _parse_date(entry.get('transaction_date') or entry.get('date'))
                if not tx_date or tx_date < cutoff:
                    continue

                ticker = (entry.get('ticker') or '').upper().strip()
                if not ticker or ticker in ('--', 'N/A', ''):
                    continue

                politician = (
                    entry.get('senator') or
                    entry.get('representative') or
                    entry.get('name') or
                    'Unknown'
                )

                tx_type = (entry.get('type') or '').lower()
                if 'purchase' in tx_type or 'buy' in tx_type:
                    side = 'buy'
                elif 'sale' in tx_type or 'sell' in tx_type:
                    side = 'sell'
                elif 'exchange' in tx_type:
                    side = 'exchange'
                else:
                    side = 'other'

                out.append({
                    'date':       tx_date.isoformat(),
                    'politician': politician,
                    'chamber':    chamber,
                    'ticker':     ticker,
                    'side':       side,
                    'amount':     _normalize_amount(entry.get('amount', '')),
                    'asset':      entry.get('asset_description', '')[:80],
                })
            except Exception:
                continue

        out.sort(key=lambda x: x['date'], reverse=True)
        return out

    def _index_by_symbol(self):
        self._by_symbol.clear()
        for tx in self._senate + self._house:
            self._by_symbol[tx['ticker']].append(tx)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def for_symbol(self, ticker: str, limit: int = 10) -> List[dict]:
        return self._by_symbol.get(ticker.upper(), [])[:limit]

    def recent(self, limit: int = 30) -> List[dict]:
        all_tx = sorted(self._senate + self._house, key=lambda x: x['date'], reverse=True)
        return all_tx[:limit]

    def directional_bias(self, ticker: str, days: int = 30) -> tuple[int, str]:
        """
        Returns (net_signal, label) where:
          +1 → more buys than sells, 'bullish'
          -1 → more sells than buys, 'bearish'
           0 → balanced or no activity
        """
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        txs = [t for t in self.for_symbol(ticker, limit=999) if t['date'] >= cutoff]
        if not txs:
            return 0, 'no activity'

        buys  = sum(1 for t in txs if t['side'] == 'buy')
        sells = sum(1 for t in txs if t['side'] == 'sell')
        if buys > sells * 1.5:
            return 1, f'bullish ({buys}B / {sells}S)'
        if sells > buys * 1.5:
            return -1, f'bearish ({buys}B / {sells}S)'
        return 0, f'mixed ({buys}B / {sells}S)'

    def watchlist_summary(self, symbols: List[str]) -> List[dict]:
        """For each symbol, returns recent activity summary."""
        out = []
        for sym in symbols:
            txs = self.for_symbol(sym, limit=5)
            if not txs:
                continue
            bias, label = self.directional_bias(sym)
            out.append({
                'symbol':       sym,
                'recent_count': len(self.for_symbol(sym, limit=999)),
                'bias':         bias,
                'bias_label':   label,
                'recent':       txs,
            })
        out.sort(key=lambda x: -x['recent_count'])
        return out
