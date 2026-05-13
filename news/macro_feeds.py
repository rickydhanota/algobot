"""
Macro news aggregator: pulls official RSS feeds from the Fed, White House,
and Treasury — the original sources that move markets.

These are public, free, and reliable. No scraping required.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.error import URLError

log = logging.getLogger(__name__)

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    log.warning('feedparser not installed — macro feeds disabled')


# Official RSS sources — public, free, no auth required
FEEDS = {
    'fed_press':    'https://www.federalreserve.gov/feeds/press_all.xml',
    'fed_speeches': 'https://www.federalreserve.gov/feeds/speeches.xml',
    'fed_monetary': 'https://www.federalreserve.gov/feeds/press_monetary.xml',
    'whitehouse':   'https://www.whitehouse.gov/feed/',
    'treasury':     'https://home.treasury.gov/rss/press-releases',
}


@dataclass
class MacroArticle:
    source:   str                          # 'fed_press' | 'whitehouse' | etc.
    title:    str
    summary:  str
    link:     str
    published: Optional[datetime] = None
    category: str = 'general'              # 'fed' | 'president' | 'tariff' | 'general'

    def to_dict(self) -> dict:
        return {
            'source':    self.source,
            'title':     self.title,
            'summary':   self.summary[:300] if self.summary else '',
            'link':      self.link,
            'published': self.published.isoformat() if self.published else None,
            'category':  self.category,
        }


class MacroFeedAggregator:
    """Pulls and categorizes macro news from official RSS feeds."""

    def __init__(self):
        self._articles: List[MacroArticle] = []
        self._last_fetch: Optional[datetime] = None

    def fetch_all(self, lookback_hours: int = 48):
        if not FEEDPARSER_AVAILABLE:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        all_articles: List[MacroArticle] = []

        for source_key, url in FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:25]:
                    pub = self._parse_date(entry)
                    if pub and pub < cutoff:
                        continue

                    title   = entry.get('title', '') or ''
                    summary = entry.get('summary', '') or entry.get('description', '') or ''
                    summary = re.sub(r'<[^>]+>', '', summary)  # strip HTML

                    art = MacroArticle(
                        source=source_key,
                        title=title.strip(),
                        summary=summary.strip(),
                        link=entry.get('link', ''),
                        published=pub,
                        category=self._categorize(source_key, title, summary),
                    )
                    all_articles.append(art)

            except (URLError, Exception) as e:
                log.warning(f'Feed {source_key} failed: {e}')

        # Dedupe by link, sort newest first
        seen = set()
        unique = []
        for art in all_articles:
            key = art.link or art.title
            if key in seen:
                continue
            seen.add(key)
            unique.append(art)

        unique.sort(key=lambda a: a.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        self._articles = unique
        self._last_fetch = datetime.now(timezone.utc)
        log.info(f'Macro feeds: {len(unique)} articles across {len(FEEDS)} sources')

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        for key in ('published_parsed', 'updated_parsed'):
            t = entry.get(key)
            if t:
                try:
                    return datetime(*t[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
        return None

    @staticmethod
    def _categorize(source: str, title: str, summary: str) -> str:
        text = (title + ' ' + summary).lower()
        if source.startswith('fed_'):
            return 'fed'
        if source == 'whitehouse':
            if any(kw in text for kw in ('tariff', 'trade', 'duty', 'sanction')):
                return 'tariff'
            return 'president'
        if source == 'treasury':
            if any(kw in text for kw in ('tariff', 'sanction', 'trade')):
                return 'tariff'
            return 'treasury'
        return 'general'

    # ── Accessors ─────────────────────────────────────────────────────────────

    def all(self, limit: int = 30) -> List[dict]:
        return [a.to_dict() for a in self._articles[:limit]]

    def by_category(self, category: str, limit: int = 10) -> List[MacroArticle]:
        return [a for a in self._articles if a.category == category][:limit]

    def fed_articles(self, hours: int = 24) -> List[MacroArticle]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self._articles
                if a.category == 'fed' and a.published and a.published >= cutoff]

    def presidential_articles(self, hours: int = 24) -> List[MacroArticle]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self._articles
                if a.category in ('president', 'tariff') and a.published and a.published >= cutoff]
