"""
Macro analysis: Fed hawkish/dovish detection, tariff/trade risk detection,
and FOMC schedule awareness.

Used to:
  1. Reduce position sizes on FOMC announcement days
  2. Skip aggressive trades after hawkish Fed statements
  3. Flag tariff / geopolitical risk events
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional


# ── Fed sentiment keywords ────────────────────────────────────────────────────

HAWKISH = {
    'raise', 'raises', 'raising', 'increase', 'increased', 'increases',
    'hike', 'hikes', 'hiking', 'tighten', 'tightening', 'restrictive',
    'persistent', 'sticky', 'elevated', 'stubborn', 'firm', 'firmer',
    'overheating', 'patient', 'premature', 'higher for longer',
    'combat inflation', 'fight inflation', 'inflation pressures',
    'upside risks', 'further increases',
}

DOVISH = {
    'cut', 'cuts', 'cutting', 'lower', 'lowering', 'reduce', 'reducing',
    'ease', 'easing', 'accommodative', 'pause', 'paused', 'pausing',
    'hold', 'patience', 'support growth', 'support employment',
    'softening', 'cooling', 'moderating', 'progress on inflation',
    'downside risks', 'rate cuts',
}

TARIFF_RISK = {
    'tariff', 'tariffs', 'duty', 'duties', 'trade war', 'trade dispute',
    'sanction', 'sanctions', 'embargo', 'executive order', 'retaliation',
    'protectionism', 'reciprocal', 'trade barrier',
}

GEOPOLITICAL_RISK = {
    'war', 'invasion', 'attack', 'strike', 'conflict', 'hostilities',
    'crisis', 'escalation', 'breakdown', 'collapse',
}


# ── Known FOMC meeting dates (2025–2026) ──────────────────────────────────────
# Update annually — published at federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES = {
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),  date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
}


@dataclass
class MacroRisk:
    fed_bias:         str = 'neutral'   # 'hawkish' | 'dovish' | 'neutral'
    fed_score:        float = 0.0       # -1 (hawkish) to +1 (dovish)
    tariff_risk:      bool = False
    geopolitical_risk: bool = False
    fomc_today:       bool = False
    fomc_tomorrow:    bool = False
    next_fomc_days:   Optional[int] = None
    recent_headlines: List[str] = None

    @property
    def risk_level(self) -> str:
        """'low' | 'moderate' | 'high' — overall macro risk."""
        if self.fomc_today or self.geopolitical_risk:
            return 'high'
        if self.fomc_tomorrow or self.tariff_risk or self.fed_score < -0.5:
            return 'moderate'
        return 'low'

    @property
    def should_reduce_risk(self) -> bool:
        return self.risk_level in ('high', 'moderate')

    @property
    def risk_multiplier(self) -> float:
        """Position-size multiplier based on macro risk."""
        if self.risk_level == 'high':     return 0.3   # 30% normal size
        if self.risk_level == 'moderate': return 0.6   # 60% normal size
        return 1.0

    def to_dict(self) -> dict:
        return {
            'fed_bias':          self.fed_bias,
            'fed_score':         round(self.fed_score, 3),
            'tariff_risk':       self.tariff_risk,
            'geopolitical_risk': self.geopolitical_risk,
            'fomc_today':        self.fomc_today,
            'fomc_tomorrow':     self.fomc_tomorrow,
            'next_fomc_days':    self.next_fomc_days,
            'risk_level':        self.risk_level,
            'risk_multiplier':   self.risk_multiplier,
            'recent_headlines':  self.recent_headlines or [],
        }


def _phrase_count(text: str, phrases: set) -> int:
    text = text.lower()
    return sum(1 for p in phrases if p in text)


def analyze_fed_articles(articles: List) -> tuple[str, float]:
    """
    Compute hawkish/dovish bias from recent Fed articles.
    Returns (bias_label, score in [-1, +1] where -1=hawkish, +1=dovish).
    """
    if not articles:
        return 'neutral', 0.0

    hawk_total = 0
    dove_total = 0
    for art in articles:
        text = f"{art.title} {art.summary}"
        hawk_total += _phrase_count(text, HAWKISH)
        dove_total += _phrase_count(text, DOVISH)

    if hawk_total + dove_total == 0:
        return 'neutral', 0.0

    score = (dove_total - hawk_total) / (hawk_total + dove_total)
    if score > 0.25:    return 'dovish', score
    if score < -0.25:   return 'hawkish', score
    return 'neutral', score


def detect_tariff_risk(articles: List) -> bool:
    for art in articles:
        text = f"{art.title} {art.summary}".lower()
        if any(kw in text for kw in TARIFF_RISK):
            return True
    return False


def detect_geopolitical_risk(articles: List) -> bool:
    for art in articles:
        text = f"{art.title} {art.summary}".lower()
        if any(kw in text for kw in GEOPOLITICAL_RISK):
            return True
    return False


def fomc_status() -> tuple[bool, bool, Optional[int]]:
    """Returns (today, tomorrow, days_until_next)."""
    today = date.today()
    future = sorted(d for d in FOMC_DATES if d >= today)
    if not future:
        return False, False, None
    next_d = future[0]
    delta = (next_d - today).days
    return (delta == 0), (delta == 1), delta


def assess(macro_feeds) -> MacroRisk:
    """One-shot macro risk assessment from the MacroFeedAggregator."""
    fed_articles  = macro_feeds.fed_articles(hours=48)
    pres_articles = macro_feeds.presidential_articles(hours=48)

    bias, score = analyze_fed_articles(fed_articles)
    tariff = detect_tariff_risk(pres_articles + fed_articles)
    geo    = detect_geopolitical_risk(pres_articles)
    fomc_today, fomc_tomorrow, days_to_fomc = fomc_status()

    recent = [a.title for a in (fed_articles + pres_articles)[:5]]

    return MacroRisk(
        fed_bias=bias,
        fed_score=score,
        tariff_risk=tariff,
        geopolitical_risk=geo,
        fomc_today=fomc_today,
        fomc_tomorrow=fomc_tomorrow,
        next_fomc_days=days_to_fomc,
        recent_headlines=recent,
    )
