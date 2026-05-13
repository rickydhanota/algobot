"""
Performance tracker: SQLite-backed trade log with win-rate and P&L stats.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent.parent / 'trades.db'


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    asset_type: str
    direction: str
    strategy: str
    entry_price: float
    exit_price: float
    shares: int
    stop_price: float
    target_price: float
    realized_pnl: float
    outcome: str           # 'win' | 'loss' | 'breakeven'
    entry_time: str
    exit_time: str
    score: int
    rvol: float
    tape_imbalance: float


@dataclass
class SessionStats:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_score: float
    best_strategy: str


class PerformanceTracker:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    asset_type TEXT,
                    direction TEXT,
                    strategy TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    shares INTEGER,
                    stop_price REAL,
                    target_price REAL,
                    realized_pnl REAL,
                    outcome TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    score INTEGER,
                    rvol REAL,
                    tape_imbalance REAL
                )
            ''')
            conn.commit()

    def record(self, trade, score: int = 0, rvol: float = 0.0, tape_imbalance: float = 0.0):
        """Record a closed trade. `trade` is an ActiveTrade instance."""
        if trade.realized_pnl is None:
            return
        outcome = 'win' if trade.realized_pnl > 0.01 else ('loss' if trade.realized_pnl < -0.01 else 'breakeven')
        with self._conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO trades VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                trade.trade_id,
                trade.symbol,
                trade.asset_type,
                trade.direction,
                trade.strategy,
                trade.entry_price,
                trade.exit_price or 0.0,
                trade.shares,
                trade.stop_price,
                trade.target_price,
                trade.realized_pnl,
                outcome,
                trade.entry_time.isoformat() if trade.entry_time else '',
                trade.exit_time.isoformat() if trade.exit_time else '',
                score,
                rvol,
                tape_imbalance,
            ))
            conn.commit()

    def stats(self, since: Optional[date] = None) -> SessionStats:
        with self._conn() as conn:
            where = ''
            params = []
            if since:
                where = "WHERE entry_time >= ?"
                params = [since.isoformat()]
            rows = conn.execute(
                f'SELECT * FROM trades {where}', params
            ).fetchall()

        if not rows:
            return SessionStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'none')

        pnls = [r[10] for r in rows]
        outcomes = [r[11] for r in rows]
        strategies = [r[4] for r in rows]
        scores = [r[14] for r in rows]

        wins = outcomes.count('win')
        losses = outcomes.count('loss')
        total = len(rows)
        win_pnls = [p for p, o in zip(pnls, outcomes) if o == 'win']
        loss_pnls = [p for p, o in zip(pnls, outcomes) if o == 'loss']

        total_win = sum(win_pnls) if win_pnls else 0
        total_loss = abs(sum(loss_pnls)) if loss_pnls else 1e-9

        strat_pnl: Dict[str, float] = {}
        for strat, pnl in zip(strategies, pnls):
            strat_pnl[strat] = strat_pnl.get(strat, 0) + pnl
        best_strategy = max(strat_pnl, key=strat_pnl.get) if strat_pnl else 'none'

        return SessionStats(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=wins / total if total else 0.0,
            total_pnl=sum(pnls),
            avg_win=total_win / wins if wins else 0.0,
            avg_loss=sum(loss_pnls) / losses if losses else 0.0,
            profit_factor=total_win / total_loss,
            largest_win=max(win_pnls) if win_pnls else 0.0,
            largest_loss=min(loss_pnls) if loss_pnls else 0.0,
            avg_score=sum(scores) / total if total else 0.0,
            best_strategy=best_strategy,
        )

    def today_stats(self) -> SessionStats:
        return self.stats(since=date.today())

    def recent_trades(self, n: int = 10) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT trade_id, symbol, direction, strategy, entry_price, '
                'exit_price, realized_pnl, outcome, exit_time '
                'FROM trades ORDER BY exit_time DESC LIMIT ?', (n,)
            ).fetchall()
        cols = ['id', 'symbol', 'dir', 'strategy', 'entry', 'exit', 'pnl', 'outcome', 'time']
        return [dict(zip(cols, r)) for r in rows]

    def all_trades_for_chart(self, since_days: int = 365) -> List[dict]:
        """Return all closed trades for charting, newest first."""
        cutoff = (date.today() - timedelta(days=since_days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT symbol, asset_type, realized_pnl, outcome, exit_time '
                'FROM trades WHERE exit_time >= ? ORDER BY exit_time DESC',
                (cutoff,)
            ).fetchall()
        return [
            {
                'symbol':    r[0],
                'asset_type': r[1],
                'pnl':       r[2],
                'outcome':   r[3],
                'exit_time': r[4],
            }
            for r in rows
        ]
