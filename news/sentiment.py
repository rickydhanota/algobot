"""
Lightweight keyword-based sentiment scoring for news headlines.

Not a replacement for FinBERT or a paid NLP model — but useful as a
*defensive filter* to skip trades when recent news is overwhelmingly
negative, and as a small tiebreaker when news is overwhelmingly positive.
"""
from __future__ import annotations
import re
from typing import List

POSITIVE = {
    # Beats / outperformance
    'beat', 'beats', 'beating', 'exceeded', 'exceeds', 'exceeding',
    'outperform', 'outperforms', 'outperformed', 'tops', 'topped',
    'record', 'records', 'milestone',
    # Price action language
    'surge', 'surges', 'surged', 'jump', 'jumps', 'jumped',
    'rally', 'rallies', 'rallied', 'soar', 'soars', 'soared',
    'climb', 'climbs', 'climbed', 'rise', 'rises', 'rose',
    'gain', 'gains', 'gained', 'higher', 'rebound', 'rebounds',
    # Analyst / corporate
    'upgrade', 'upgraded', 'upgrades', 'buy', 'overweight',
    'partnership', 'acquisition', 'merger', 'expand', 'expands',
    'launch', 'launches', 'launched', 'wins', 'won', 'awarded',
    'approved', 'approval', 'breakthrough', 'success', 'successful',
    # Fundamentals
    'profit', 'profitable', 'strong', 'growth', 'growing',
    'bullish', 'positive', 'optimistic', 'confident',
}

NEGATIVE = {
    # Misses / underperformance
    'miss', 'misses', 'missed', 'missing', 'shortfall',
    'underperform', 'underperformed', 'disappoint', 'disappoints',
    # Price action
    'plunge', 'plunges', 'plunged', 'fall', 'falls', 'fell',
    'drop', 'drops', 'dropped', 'tumble', 'tumbles', 'crash', 'crashes',
    'sink', 'sinks', 'slump', 'slumped', 'decline', 'declines',
    'lower', 'losses', 'loss', 'lost',
    # Analyst / corporate
    'downgrade', 'downgraded', 'sell', 'underweight',
    'lawsuit', 'sued', 'investigation', 'probe', 'subpoena',
    'layoff', 'layoffs', 'fired', 'cuts', 'cut', 'slash',
    'recall', 'delay', 'delayed', 'fails', 'failed', 'failure',
    # Severe
    'fraud', 'scandal', 'bankruptcy', 'bankrupt', 'restructure',
    # Sentiment language
    'warns', 'warning', 'weak', 'weakness', 'concern', 'concerns',
    'risk', 'risks', 'risky', 'bearish', 'negative', 'pessimistic',
}


def score_text(text: str) -> float:
    """Sentiment in [-1, +1] from keyword counts. Returns 0 for neutral/no-match."""
    if not text:
        return 0.0
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def score_articles(articles: List[dict]) -> dict:
    """
    Aggregate sentiment across articles.
    Returns: {score, count, positive, negative, neutral, label}
    """
    if not articles:
        return {
            'score': 0.0, 'count': 0,
            'positive': 0, 'negative': 0, 'neutral': 0,
            'label': 'neutral',
        }

    pos, neg, neu = 0, 0, 0
    total = 0.0
    for art in articles:
        text = f"{art.get('headline','')} {art.get('summary','')}"
        s = score_text(text)
        total += s
        if s > 0.2:
            pos += 1
        elif s < -0.2:
            neg += 1
        else:
            neu += 1

    avg = total / len(articles)
    label = 'positive' if avg > 0.20 else ('negative' if avg < -0.20 else 'neutral')

    return {
        'score':    round(avg, 3),
        'count':    len(articles),
        'positive': pos,
        'negative': neg,
        'neutral':  neu,
        'label':    label,
    }
