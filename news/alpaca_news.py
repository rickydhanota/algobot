"""
Alpaca News API client.

Pulls news via Alpaca's news endpoint (powered by Benzinga). Covers WSJ,
Reuters, Bloomberg, CNBC, MarketWatch, and other major outlets.

Free with your existing Alpaca account — no extra signup.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import config

try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False

log = logging.getLogger(__name__)


class AlpacaNewsFeed:
    """Fetches and caches news articles for watchlist symbols."""

    def __init__(self):
        self._client = None
        self._cache: Dict[str, List[dict]] = defaultdict(list)
        self._all_articles: List[dict] = []
        self._last_fetch: Optional[datetime] = None

        if NEWS_API_AVAILABLE:
            try:
                self._client = NewsClient(
                    api_key=config.ALPACA_API_KEY,
                    secret_key=config.ALPACA_SECRET_KEY,
                )
            except Exception as e:
                log.warning(f'NewsClient init failed: {e}')

    def fetch(self, symbols: List[str], hours: int = 24):
        if self._client is None:
            return
        try:
            req = NewsRequest(
                symbols=symbols,
                start=datetime.now(timezone.utc) - timedelta(hours=hours),
                limit=50,
                include_content=False,
                exclude_contentless=True,
            )
            resp = self._client.get_news(req)

            self._cache.clear()
            self._all_articles.clear()
            seen_ids = set()

            articles = resp.news if hasattr(resp, 'news') else resp.data.get('news', [])
            for art in articles:
                art_id = getattr(art, 'id', None) or getattr(art, 'url', None)
                if art_id in seen_ids:
                    continue
                seen_ids.add(art_id)

                created = getattr(art, 'created_at', None)
                article = {
                    'id':         art_id,
                    'headline':   getattr(art, 'headline', ''),
                    'summary':    getattr(art, 'summary', '') or '',
                    'author':     getattr(art, 'author', '') or '',
                    'source':     getattr(art, 'source', '') or '',
                    'url':        getattr(art, 'url', '') or '',
                    'symbols':    list(getattr(art, 'symbols', []) or []),
                    'created_at': created.isoformat() if created else None,
                }
                self._all_articles.append(article)
                for sym in article['symbols']:
                    if sym in symbols:
                        self._cache[sym].append(article)

            self._all_articles.sort(key=lambda a: a.get('created_at') or '', reverse=True)
            self._last_fetch = datetime.now(timezone.utc)
            log.info(f'News fetched: {len(self._all_articles)} articles across {len(self._cache)} symbols')

        except Exception as e:
            log.warning(f'News fetch failed: {e}')

    def for_symbol(self, symbol: str) -> List[dict]:
        return self._cache.get(symbol, [])

    def all_recent(self, limit: int = 30) -> List[dict]:
        return self._all_articles[:limit]
