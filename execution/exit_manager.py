"""
Exit manager: tiered profit-taking + tape-aware dynamic stops.

Two responsibilities run on every tick:

1. TIERED TRIMMING
   At each profit threshold, peel off a fraction of the position:
     +20%  →  exit 60% of contracts  (lock in majority of gains)
     +35%  →  exit another 20%
     +60%  →  exit another 15%
     +100% →  exit another 4%
     leaves ~1% as a moon-runner

2. DYNAMIC STOPS (option positions only)
   At −15% loss, normally exit. But override:
     • If the underlying's priority tape still confirms our direction
       (confluence ≥ 60 + direction matches), HOLD and let it work.
     • If tape goes neutral or against us, EXIT immediately.
   Also: any time tape FLIPS by ≥ 0.4 imbalance against us, exit
   regardless of P&L.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from execution.order_manager import OrderManager, ActiveTrade
    from signals.priority_tape import PriorityTapeReader

log = logging.getLogger(__name__)


# (profit_pct_threshold, fraction_of_ORIGINAL_position_to_close)
TRIM_TIERS = [
    (0.20, 0.60),
    (0.35, 0.20),
    (0.60, 0.15),
    (1.00, 0.04),
    # remaining 0.01 = moon runner
]

SOFT_STOP_PCT = -0.15           # "watch zone" — hold by default, exit only on negative evidence
HARD_STOP_PCT = -0.40           # catastrophic backstop — always exit
TAPE_FLIP_THRESHOLD = 0.40      # opposite-direction imbalance triggers exit
TAPE_AGAINST_CONF_MIN = 40      # at soft stop, tape confluence ≥ this against us = exit
TAPE_AGAINST_IMBALANCE_MIN = 0.30


@dataclass
class ExitDecision:
    trade_id: str
    action: str                  # 'trim' | 'stop' | 'hold'
    quantity: int                # contracts (or shares for stocks) to close
    reason: str
    pnl_pct: float


class ExitManager:
    def __init__(self, order_manager: 'OrderManager'):
        self.orders = order_manager
        # Underlying mapping for options (e.g., AAPL250117C00200000 → AAPL)
        # Populated when the option order is placed via `register_underlying`.
        self._underlying: Dict[str, str] = {}

    def register_underlying(self, trade_id: str, underlying: str):
        self._underlying[trade_id] = underlying

    # ── Main loop entry ──────────────────────────────────────────────────────

    def process(
        self,
        live_prices: Dict[str, float],
        priority_tape: Optional['PriorityTapeReader'] = None,
        volume_monitor=None,
    ) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        for trade_id, trade in list(self.orders.active.items()):
            if trade.status != 'open':
                continue
            if trade.asset_type != 'option':
                continue  # tiered trimming is options-only by request

            current = live_prices.get(trade.symbol)
            if not current or not trade.entry_price:
                continue

            pnl_pct = (current - trade.entry_price) / trade.entry_price

            # 1. Tiered trimming
            decision = self._maybe_trim(trade, current, pnl_pct)
            if decision:
                decisions.append(decision)
                continue

            # 2. Dynamic stop with tape + volume confirmation
            decision = self._maybe_dynamic_stop(
                trade, current, pnl_pct, priority_tape, volume_monitor,
            )
            if decision:
                decisions.append(decision)
        return decisions

    # ── Tiered trimming ──────────────────────────────────────────────────────

    def _maybe_trim(self, trade, current: float, pnl_pct: float) -> Optional[ExitDecision]:
        # Init metadata if missing (older trades from previous versions)
        if not hasattr(trade, 'tier_hits') or trade.tier_hits is None:
            trade.tier_hits = []
        if not getattr(trade, 'original_contracts', None):
            trade.original_contracts = trade.contracts or max(1, trade.shares // 100)

        for idx, (threshold, frac) in enumerate(TRIM_TIERS):
            if idx in trade.tier_hits:
                continue
            if pnl_pct < threshold:
                continue
            # Hit this tier
            qty = max(1, math.floor(trade.original_contracts * frac))
            remaining = trade.contracts or 0
            if remaining <= 1:
                # Down to runner already — don't trim, mark hit so we don't loop
                trade.tier_hits.append(idx)
                continue
            qty = min(qty, remaining - 1)  # always keep ≥ 1 contract as runner
            if qty <= 0:
                trade.tier_hits.append(idx)
                continue

            try:
                self.orders._partial_close_option(trade, qty, f'trim_t{idx+1}')
                trade.tier_hits.append(idx)
                return ExitDecision(
                    trade_id=trade.trade_id,
                    action='trim',
                    quantity=qty,
                    reason=f'tier{idx+1} @ {threshold*100:.0f}% gain',
                    pnl_pct=pnl_pct,
                )
            except Exception as e:
                log.error(f'[{trade.symbol}] Trim failed: {e}')
                return None
        return None

    # ── Dynamic stops ────────────────────────────────────────────────────────

    def _maybe_dynamic_stop(
        self,
        trade,
        current: float,
        pnl_pct: float,
        priority_tape,
        volume_monitor=None,
    ) -> Optional[ExitDecision]:
        """
        Dynamic stop philosophy:
          • At -15% (SOFT): trades dip — that's normal. HOLD by default.
            ONLY exit on negative evidence: tape clearly turning against us,
            OR volume drying up / stale tape (can't read the market).
          • At any P&L: if tape FLIPS hard against us (sweep or imbalance
            ≥ 0.40 opposite our direction with confluence ≥ 50) → exit.
          • At -40% (HARD): catastrophic — exit no matter what to preserve
            capital.
        """
        underlying = self._underlying.get(trade.trade_id) or self._derive_underlying(trade.symbol)
        tape_sig = None
        if priority_tape and underlying:
            try:
                tape_sig = priority_tape.analyze(underlying)
            except Exception:
                tape_sig = None

        vol_state = None
        if volume_monitor and underlying:
            try:
                vol_state = volume_monitor.analyze(underlying)
            except Exception:
                vol_state = None

        our_dir = 'buy' if trade.option_type == 'call' else 'sell'
        opp_dir = 'sell' if our_dir == 'buy' else 'buy'

        # ── 1. Hard backstop FIRST: catastrophic loss → always exit ──────────
        if pnl_pct <= HARD_STOP_PCT:
            self.orders._close_trade(trade, current, 'hard_stop_40pct')
            return ExitDecision(
                trade_id=trade.trade_id, action='stop', quantity=trade.contracts or 0,
                reason=f'hard stop {pnl_pct*100:.1f}% (catastrophic)',
                pnl_pct=pnl_pct,
            )

        # ── 2. Tape FLIP (any P&L): clear reversal → exit ────────────────────
        if tape_sig:
            if (tape_sig.direction == opp_dir and
                    abs(tape_sig.imbalance_100) >= TAPE_FLIP_THRESHOLD and
                    tape_sig.confluence_score >= 50):
                self.orders._close_trade(trade, current, 'tape_flip')
                return ExitDecision(
                    trade_id=trade.trade_id, action='stop', quantity=trade.contracts or 0,
                    reason=f'tape flipped {opp_dir} (imb {tape_sig.imbalance_100:+.2f}, conf {tape_sig.confluence_score})',
                    pnl_pct=pnl_pct,
                )

        # ── 3. Above soft stop → no action ───────────────────────────────────
        if pnl_pct > SOFT_STOP_PCT:
            return None

        # ── 4. At/below soft stop: hold by default, exit on NEGATIVE evidence

        # 4a. Tape clearly against us → exit
        if tape_sig:
            if (tape_sig.direction == opp_dir and
                    tape_sig.confluence_score >= TAPE_AGAINST_CONF_MIN and
                    abs(tape_sig.imbalance_100) >= TAPE_AGAINST_IMBALANCE_MIN):
                self.orders._close_trade(trade, current, 'soft_stop_tape_against')
                return ExitDecision(
                    trade_id=trade.trade_id, action='stop', quantity=trade.contracts or 0,
                    reason=f'tape against us at {pnl_pct*100:.1f}% (imb {tape_sig.imbalance_100:+.2f})',
                    pnl_pct=pnl_pct,
                )

        # 4b. Volume gone (stale or unhealthy) → can't read → exit
        if vol_state and (vol_state.is_stale or not vol_state.is_healthy):
            self.orders._close_trade(trade, current, 'soft_stop_no_volume')
            return ExitDecision(
                trade_id=trade.trade_id, action='stop', quantity=trade.contracts or 0,
                reason=f'volume unhealthy at {pnl_pct*100:.1f}% ({vol_state.health_label})',
                pnl_pct=pnl_pct,
            )

        # 4c. Default: HOLD — log once per minute so we know it's intentional
        last_log_key = f'_hold_log_{trade.trade_id}'
        import time
        now_ts = int(time.time())
        last = getattr(self, last_log_key, 0)
        if now_ts - last > 60:
            tape_info = (
                f'tape={tape_sig.direction}/conf={tape_sig.confluence_score}'
                if tape_sig else 'no tape'
            )
            log.info(
                f'[{trade.symbol}] {pnl_pct*100:+.1f}% — HOLDING (no negative evidence; {tape_info})'
            )
            setattr(self, last_log_key, now_ts)
        return None
        return ExitDecision(
            trade_id=trade.trade_id, action='stop', quantity=trade.contracts or 0,
            reason=f'-15% stop, tape not supporting',
            pnl_pct=pnl_pct,
        )

    @staticmethod
    def _derive_underlying(option_symbol: str) -> str:
        """OCC symbol → underlying ticker. e.g. AAPL250117C00200000 → AAPL."""
        idx = 0
        while idx < len(option_symbol) and option_symbol[idx].isalpha():
            idx += 1
        return option_symbol[:idx]
