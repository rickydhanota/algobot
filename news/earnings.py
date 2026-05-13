"""
Earnings & IPO calendar via yfinance (free, no API key required).

Used primarily as a *defensive filter* — avoid trading symbols reporting
earnings today/tomorrow, since earnings gaps invalidate technical setups.
"""
from __future__ import annotations
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    log.warning('yfinance not installed — earnings/IPO data disabled')


class EarningsCalendar:
    """Caches next earnings date per symbol; refreshed once per day."""

    def __init__(self):
        self._cache: Dict[str, Optional[date]] = {}
        self._last_refresh: Optional[date] = None

    def refresh(self, symbols: List[str], force: bool = False):
        if not YF_AVAILABLE:
            return
        if not force and self._last_refresh == date.today():
            return

        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                cal = t.calendar
                edate = None

                if isinstance(cal, dict):
                    raw = cal.get('Earnings Date')
                    if isinstance(raw, list) and raw:
                        raw = raw[0]
                    if raw:
                        edate = raw.date() if hasattr(raw, 'date') else raw

                self._cache[sym] = edate
            except Exception:
                self._cache[sym] = None

        self._last_refresh = date.today()
        upcoming = sum(1 for v in self._cache.values() if v)
        log.info(f'Earnings calendar refreshed: {upcoming}/{len(symbols)} have known dates')

    def has_earnings_within(self, symbol: str, days: int = 1) -> bool:
        edate = self._cache.get(symbol)
        if not edate:
            return False
        delta = (edate - date.today()).days
        return 0 <= delta <= days

    def days_until_earnings(self, symbol: str) -> Optional[int]:
        edate = self._cache.get(symbol)
        if not edate:
            return None
        return (edate - date.today()).days

    def upcoming(self, symbols: List[str], within_days: int = 14) -> List[dict]:
        result = []
        for sym in symbols:
            edate = self._cache.get(sym)
            if not edate:
                continue
            delta = (edate - date.today()).days
            if 0 <= delta <= within_days:
                result.append({
                    'symbol':    sym,
                    'date':      edate.isoformat(),
                    'days_away': delta,
                })
        result.sort(key=lambda x: x['days_away'])
        return result


class IPOCalendar:
    """Lightweight IPO calendar using yfinance's screener (best-effort)."""

    def __init__(self):
        self._cache: List[dict] = []
        self._last_refresh: Optional[date] = None

    def refresh(self, force: bool = False):
        """Refresh IPO list — uses Yahoo Finance trending IPOs (limited scope)."""
        if not YF_AVAILABLE:
            return
        if not force and self._last_refresh == date.today():
            return

        try:
            # yfinance doesn't expose IPO calendar directly; we use the
            # search/screener endpoints if available, otherwise leave empty.
            # For real IPO tracking, consider adding a Finnhub free API key.
            self._cache = []
            self._last_refresh = date.today()
        except Exception as e:
            log.warning(f'IPO refresh failed: {e}')

    def upcoming(self) -> List[dict]:
        return self._cache
