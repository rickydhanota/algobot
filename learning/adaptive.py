"""
Adaptive threshold learning.

Tracks per-(symbol, strategy) win rate over a rolling window and adjusts
the score threshold required to enter new trades. Bias is asymmetric:

  • Losing streaks → tighten threshold quickly (defensive)
  • Winning streaks → loosen slowly (don't get overconfident)

This isn't ML — it's a feedback loop tuned for honesty. Use enough sample
size (≥10 trades per bucket) before applying any adjustment.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent.parent / 'learning.db'

MIN_SAMPLE_SIZE = 10           # need ≥10 trades before adjusting thresholds
TIGHTEN_PER_LOSS = 1.5         # raise threshold ~1.5pts per loss below baseline win rate
LOOSEN_PER_WIN = 0.5           # lower threshold ~0.5pts per win above baseline (slow)
MAX_TIGHTEN = 15               # cap upward adjustment at +15 pts
MAX_LOOSEN = -5                # cap downward adjustment at -5 pts (don't go below baseline-5)
BASELINE_WIN_RATE = 0.55       # what we consider "neutral" performance


@dataclass
class SignalRecord:
    signal_id: str
    symbol: str
    strategy: str
    score: int
    imbalance: float
    rvol: float
    confluence_score: int       # from priority_tape if applicable
    direction: str
    timestamp: str
    outcome: Optional[str] = None   # 'win' | 'loss' | None until closed
    pnl: Optional[float] = None


@dataclass
class BucketStats:
    symbol: str
    strategy: str
    sample_size: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    threshold_adjustment: int   # add this many points to MIN_SIGNAL_SCORE


class AdaptiveLearner:
    """SQLite-backed adaptive threshold adjuster."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_records (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    strategy TEXT,
                    score INTEGER,
                    imbalance REAL,
                    rvol REAL,
                    confluence_score INTEGER,
                    direction TEXT,
                    timestamp TEXT,
                    outcome TEXT,
                    pnl REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sym_strat ON signal_records(symbol, strategy)')
            # Lightweight migration: add session_window column if missing
            try:
                conn.execute('ALTER TABLE signal_records ADD COLUMN session_window TEXT')
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.commit()

    # ── Recording ────────────────────────────────────────────────────────────

    def record_signal(
        self,
        signal_id: str,
        symbol: str,
        strategy: str,
        score: int,
        imbalance: float = 0.0,
        rvol: float = 0.0,
        confluence_score: int = 0,
        direction: str = '',
        session_window: str = '',
    ):
        with self._conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO signal_records
                (signal_id, symbol, strategy, score, imbalance, rvol,
                 confluence_score, direction, timestamp, outcome, pnl, session_window)
                VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,?)
            ''', (
                signal_id, symbol, strategy, score, imbalance, rvol,
                confluence_score, direction, datetime.now(timezone.utc).isoformat(),
                session_window,
            ))
            conn.commit()

    def record_outcome(self, signal_id: str, pnl: float):
        outcome = 'win' if pnl > 0 else ('loss' if pnl < 0 else 'breakeven')
        with self._conn() as conn:
            conn.execute(
                'UPDATE signal_records SET outcome=?, pnl=? WHERE signal_id=?',
                (outcome, pnl, signal_id),
            )
            conn.commit()

    # ── Stats & adjustments ──────────────────────────────────────────────────

    def bucket_stats(
        self,
        symbol: str,
        strategy: str,
        last_n: int = 30,
        session_window: Optional[str] = None,
    ) -> Optional[BucketStats]:
        """
        Rolling stats for a (symbol, strategy) pair.
        If session_window is provided, filter to that window only.
        """
        with self._conn() as conn:
            if session_window:
                rows = conn.execute('''
                    SELECT outcome, pnl FROM signal_records
                    WHERE symbol=? AND strategy=? AND session_window=?
                          AND outcome IS NOT NULL
                    ORDER BY timestamp DESC LIMIT ?
                ''', (symbol, strategy, session_window, last_n)).fetchall()
            else:
                rows = conn.execute('''
                    SELECT outcome, pnl FROM signal_records
                    WHERE symbol=? AND strategy=? AND outcome IS NOT NULL
                    ORDER BY timestamp DESC LIMIT ?
                ''', (symbol, strategy, last_n)).fetchall()

        if not rows:
            return None

        outcomes = [r[0] for r in rows]
        pnls = [r[1] or 0 for r in rows]
        wins = outcomes.count('win')
        losses = outcomes.count('loss')
        total = wins + losses
        if total == 0:
            return None

        win_rate = wins / total
        avg_pnl = sum(pnls) / len(pnls)

        if total < MIN_SAMPLE_SIZE:
            adj = 0
        else:
            delta = BASELINE_WIN_RATE - win_rate
            if delta > 0:
                adj = int(min(MAX_TIGHTEN, delta * 100 * TIGHTEN_PER_LOSS / 10))
            else:
                adj = int(max(MAX_LOOSEN, delta * 100 * LOOSEN_PER_WIN / 10))

        return BucketStats(
            symbol=symbol,
            strategy=strategy,
            sample_size=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=sum(pnls),
            avg_pnl=avg_pnl,
            threshold_adjustment=adj,
        )

    def threshold_adjustment(
        self,
        symbol: str,
        strategy: str,
        session_window: Optional[str] = None,
    ) -> int:
        """
        Returns points to add to MIN_SIGNAL_SCORE for this bucket.
        If session_window is provided AND has ≥ MIN_SAMPLE_SIZE samples,
        use the per-window adjustment. Otherwise fall back to overall.
        """
        if session_window:
            window_stats = self.bucket_stats(symbol, strategy, session_window=session_window)
            if window_stats and window_stats.sample_size >= MIN_SAMPLE_SIZE:
                return window_stats.threshold_adjustment
        stats = self.bucket_stats(symbol, strategy)
        return stats.threshold_adjustment if stats else 0

    def session_stats(self) -> List[dict]:
        """Per-session-window stats across all (symbol, strategy) pairs."""
        with self._conn() as conn:
            rows = conn.execute('''
                SELECT session_window,
                       COUNT(*) AS total,
                       SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) AS losses,
                       SUM(COALESCE(pnl, 0)) AS total_pnl
                FROM signal_records
                WHERE outcome IS NOT NULL AND session_window IS NOT NULL
                GROUP BY session_window
            ''').fetchall()
        out = []
        for window, total, wins, losses, total_pnl in rows:
            if not total:
                continue
            decided = (wins or 0) + (losses or 0)
            out.append({
                'session_window': window or 'unknown',
                'sample_size':    total,
                'wins':           wins or 0,
                'losses':         losses or 0,
                'win_rate':       round((wins or 0) / decided, 3) if decided else 0.0,
                'total_pnl':      round(total_pnl or 0, 2),
            })
        out.sort(key=lambda x: -x['sample_size'])
        return out

    def all_stats(self) -> List[dict]:
        """Returns stats for all (symbol, strategy) pairs that have closed trades."""
        with self._conn() as conn:
            pairs = conn.execute('''
                SELECT DISTINCT symbol, strategy FROM signal_records
                WHERE outcome IS NOT NULL
            ''').fetchall()

        out = []
        for sym, strat in pairs:
            s = self.bucket_stats(sym, strat)
            if s:
                out.append({
                    'symbol':              s.symbol,
                    'strategy':            s.strategy,
                    'sample_size':         s.sample_size,
                    'wins':                s.wins,
                    'losses':              s.losses,
                    'win_rate':            round(s.win_rate, 3),
                    'total_pnl':           round(s.total_pnl, 2),
                    'avg_pnl':             round(s.avg_pnl, 2),
                    'threshold_adjustment': s.threshold_adjustment,
                })
        out.sort(key=lambda x: -x['sample_size'])
        return out
