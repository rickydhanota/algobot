"""
Shadow learning — observe-only outcomes for every signal evaluated, even
when no actual trade fires.

Why this exists: the adaptive learner only updates from CLOSED real
trades. Quiet days produce zero learning. Shadow learning fills the gap
by recording every directional signal the bot detects, then evaluating
"would this have worked?" 30 minutes later by checking the underlying's
actual price move.

Outcome rules (simplified — proxy for what an options play would do):
  • LONG  signal: underlying up   >0.3% in 30 min → shadow_win
                  underlying down >0.3% in 30 min → shadow_loss
                  within ±0.3%                    → shadow_neutral
  • SHORT signal: mirror

These shadow outcomes feed into the same `record_outcome` channel as
real trades, with `shadow=True` flag so we can weight them differently
or filter them out.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict

DB_PATH = Path(__file__).parent.parent / 'shadow.db'
SHADOW_HORIZON_MINUTES = 30           # evaluate signals 30 min after recording
SHADOW_WIN_THRESHOLD_PCT = 0.003      # 0.3% underlying move in our direction = win
SHADOW_THROTTLE_MINUTES = 5           # don't record duplicate signal for same (sym, strat) within N min


class ShadowEvaluator:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # In-memory dedupe: (symbol, strategy) -> last_record_ts
        self._last_recorded: Dict[tuple, datetime] = {}
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            c.execute('''
                CREATE TABLE IF NOT EXISTS shadow_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    score INTEGER,
                    direction TEXT,
                    session_window TEXT,
                    underlying_entry REAL,
                    rejection_reason TEXT,
                    evaluated INTEGER DEFAULT 0,
                    underlying_exit REAL,
                    move_pct REAL,
                    outcome TEXT
                )
            ''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_shadow_eval ON shadow_signals(evaluated, timestamp)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_shadow_strat ON shadow_signals(strategy, session_window)')

    # ── Recording ────────────────────────────────────────────────────────────

    def record(
        self,
        symbol: str,
        strategy: str,
        score: int,
        direction: str,
        session_window: str,
        underlying_price: float,
        rejection_reason: str = '',
    ) -> bool:
        """
        Record a shadow signal. Returns True if recorded, False if
        throttled (same (symbol, strategy) recorded within last N min).
        """
        if not underlying_price or underlying_price <= 0:
            return False

        now = datetime.now(timezone.utc)
        key = (symbol, strategy)
        last_ts = self._last_recorded.get(key)
        if last_ts and (now - last_ts).total_seconds() < SHADOW_THROTTLE_MINUTES * 60:
            return False
        self._last_recorded[key] = now

        with self._conn() as c:
            c.execute('''
                INSERT INTO shadow_signals
                (timestamp, symbol, strategy, score, direction, session_window,
                 underlying_entry, rejection_reason)
                VALUES (?,?,?,?,?,?,?,?)
            ''', (
                now.isoformat(), symbol, strategy, score, direction,
                session_window, underlying_price, rejection_reason or '',
            ))
        return True

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_pending(self, current_prices: Dict[str, float], horizon_min: int = SHADOW_HORIZON_MINUTES):
        """
        For every shadow signal at least horizon_min old that hasn't been
        evaluated yet, compute the hypothetical outcome from underlying move.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=horizon_min)).isoformat()
        evaluated = 0
        with self._conn() as c:
            pending = c.execute('''
                SELECT id, symbol, direction, underlying_entry
                FROM shadow_signals
                WHERE evaluated = 0 AND timestamp <= ?
            ''', (cutoff,)).fetchall()

            for sid, sym, direction, entry in pending:
                current = current_prices.get(sym)
                if not current or not entry:
                    continue

                move_pct = (current - entry) / entry
                long_side = direction in ('long', 'buy', 'call')

                if long_side:
                    if   move_pct >  SHADOW_WIN_THRESHOLD_PCT:  outcome = 'shadow_win'
                    elif move_pct < -SHADOW_WIN_THRESHOLD_PCT:  outcome = 'shadow_loss'
                    else:                                       outcome = 'shadow_neutral'
                else:
                    if   move_pct < -SHADOW_WIN_THRESHOLD_PCT:  outcome = 'shadow_win'
                    elif move_pct >  SHADOW_WIN_THRESHOLD_PCT:  outcome = 'shadow_loss'
                    else:                                       outcome = 'shadow_neutral'

                c.execute('''
                    UPDATE shadow_signals
                    SET evaluated = 1, underlying_exit = ?, move_pct = ?, outcome = ?
                    WHERE id = ?
                ''', (current, move_pct, outcome, sid))
                evaluated += 1
        return evaluated

    # ── Analytics for dashboard ──────────────────────────────────────────────

    def stats_overall(self) -> dict:
        with self._conn() as c:
            row = c.execute('''
                SELECT
                    COUNT(*) AS evaluated,
                    SUM(CASE WHEN outcome='shadow_win'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome='shadow_loss' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN outcome='shadow_neutral' THEN 1 ELSE 0 END) AS neutral
                FROM shadow_signals WHERE evaluated = 1
            ''').fetchone()
            pending_row = c.execute(
                'SELECT COUNT(*) FROM shadow_signals WHERE evaluated = 0'
            ).fetchone()
        if not row or row[0] == 0:
            return {
                'evaluated': 0, 'wins': 0, 'losses': 0, 'neutral': 0,
                'win_rate': 0.0, 'pending': pending_row[0] if pending_row else 0,
            }
        total, wins, losses, neutral = row
        decided = (wins or 0) + (losses or 0)
        return {
            'evaluated': total,
            'wins':      wins or 0,
            'losses':    losses or 0,
            'neutral':   neutral or 0,
            'win_rate':  round((wins or 0) / decided, 3) if decided else 0.0,
            'pending':   pending_row[0] if pending_row else 0,
        }

    def stats_by_bucket(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute('''
                SELECT
                    symbol, strategy, session_window,
                    COUNT(*)                                        AS total,
                    SUM(CASE WHEN outcome='shadow_win'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome='shadow_loss' THEN 1 ELSE 0 END) AS losses,
                    AVG(score)                                      AS avg_score,
                    AVG(move_pct)                                   AS avg_move
                FROM shadow_signals
                WHERE evaluated = 1
                GROUP BY symbol, strategy, session_window
                HAVING total >= 3
                ORDER BY total DESC
            ''').fetchall()
        out = []
        for r in rows:
            sym, strat, win, total, wins, losses, avg_score, avg_move = r
            decided = (wins or 0) + (losses or 0)
            out.append({
                'symbol':         sym,
                'strategy':       strat,
                'session_window': win,
                'total':          total,
                'wins':           wins or 0,
                'losses':         losses or 0,
                'win_rate':       round((wins or 0) / decided, 3) if decided else 0.0,
                'avg_score':      round(avg_score or 0, 1),
                'avg_move_pct':   round((avg_move or 0) * 100, 3),
            })
        return out

    def recent(self, limit: int = 20) -> List[dict]:
        with self._conn() as c:
            rows = c.execute('''
                SELECT timestamp, symbol, strategy, score, direction,
                       session_window, underlying_entry, underlying_exit,
                       move_pct, outcome
                FROM shadow_signals
                WHERE evaluated = 1
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,)).fetchall()
        return [{
            'timestamp':        r[0],
            'symbol':           r[1],
            'strategy':         r[2],
            'score':            r[3],
            'direction':        r[4],
            'session_window':   r[5],
            'underlying_entry': r[6],
            'underlying_exit':  r[7],
            'move_pct':         r[8],
            'outcome':          r[9],
        } for r in rows]
