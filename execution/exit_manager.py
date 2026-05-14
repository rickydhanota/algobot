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

# "Thesis broken" check — covers the gap between -15% soft stop and +20% tier 1
# where no other exit logic operates. Closes the trade when the conditions
# that justified entry have evaporated.
THESIS_PNL_LOW  = -0.15             # only fires inside this band...
THESIS_PNL_HIGH = 0.20              # ...where no other exit logic operates
THESIS_VOL_RATE_MAX = 0.50          # volume must be < 0.5× normal
THESIS_SUSTAINED_SECONDS = 60       # ...continuously for 60s before exit fires
THESIS_MIN_CHECKS = 6               # AND at least 6 consecutive bad samples (~30s @ 5s loop)
THESIS_PROGRESS_LOG_EVERY = 3       # log progression every N checks

# "Profit protect" — data-driven full-exit when the trade is in profit but
# tape/volume support is fading. Specifically handles two cases the trim
# ladder can't cover:
#   1. Single-contract positions (trims round to 0 contracts)
#   2. Multi-contract positions that have exhausted all trim tiers
PROFIT_PROTECT_MIN_PCT = 0.05         # only engage when at least +5%
PROFIT_PROTECT_VOL_THRESHOLD = 0.70   # volume below 70% normal = warning
PROFIT_PROTECT_SUSTAINED_SECONDS = 30 # 30s sustained (faster than 60s thesis)
PROFIT_PROTECT_MIN_CHECKS = 3         # 3+ consecutive bad samples (≈15s @ 5s loop)

# Volume-aware trim acceleration (protects runners when momentum fades)
VOLUME_TRIM_RATE_THRESHOLD = 0.70   # rate_ratio below this = "weakening"
VOLUME_TRIM_TAPE_CONF_THRESHOLD = 30 # confluence below this + neutral dir = "tape fading"
VOLUME_TRIM_BUFFER_PCT = 0.05       # need pnl > last_tier + 5% before accelerating


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
        # Active thesis-broken watches: trade_id → state dict
        # {'start': datetime, 'checks': int, 'last_vol': float, 'last_conf': int, 'last_pnl': float}
        # Reset to empty whenever conditions recover — exit only fires when
        # the watch survives every check between start and 60s elapsed.
        self._thesis_watches: Dict[str, dict] = {}
        # Active profit-protect watches (positions in profit but conditions fading)
        self._profit_watches: Dict[str, dict] = {}

    def register_underlying(self, trade_id: str, underlying: str):
        self._underlying[trade_id] = underlying

    # ── Main loop entry ──────────────────────────────────────────────────────

    def process(
        self,
        live_prices: Dict[str, float],
        priority_tape: Optional['PriorityTapeReader'] = None,
        volume_monitor=None,
        fallback_tape=None,
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

            # 0. Thesis-broken check — covers the −15% to +20% no-man's-land
            decision = self._maybe_thesis_broken(
                trade, current, pnl_pct, priority_tape, volume_monitor,
                fallback_tape=fallback_tape,
            )
            if decision:
                decisions.append(decision)
                continue

            # 1. Natural tiered trimming (profit thresholds)
            decision = self._maybe_trim(trade, current, pnl_pct)
            if decision:
                decisions.append(decision)
                continue

            # 1b. Volume-aware acceleration (runner protection)
            decision = self._maybe_volume_trim(
                trade, current, pnl_pct, priority_tape, volume_monitor,
            )
            if decision:
                decisions.append(decision)
                continue

            # 1c. Profit-protect full exit — for positions in profit where the
            # trim ladder cannot help (single contract or tiers exhausted)
            decision = self._maybe_profit_protect(
                trade, current, pnl_pct, priority_tape, volume_monitor, fallback_tape,
            )
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

    # ── Thesis-broken check ──────────────────────────────────────────────────

    def _maybe_thesis_broken(
        self,
        trade,
        current: float,
        pnl_pct: float,
        priority_tape,
        volume_monitor,
        fallback_tape=None,
    ) -> Optional[ExitDecision]:
        """
        Covers the gap between -15% (soft stop) and +20% (tier 1 trim) where
        no other exit logic operates.

        If for SUSTAINED 60+ seconds:
          • Volume rate falls below 0.5× normal
          • AND underlying tape direction is no longer ours
        the original thesis is dead — exit before theta eats us.
        """
        from datetime import datetime, timezone

        tid = trade.trade_id

        # Out of the gap zone → discard any active watch
        if not (THESIS_PNL_LOW < pnl_pct < THESIS_PNL_HIGH):
            if tid in self._thesis_watches:
                log.info(f'[{trade.symbol}] thesis watch ended — P&L left -15%..+20% zone')
            self._thesis_watches.pop(tid, None)
            return None

        underlying = self._underlying.get(tid) or self._derive_underlying(trade.symbol)
        now = datetime.now(timezone.utc)

        # Re-read BOTH indicators from current tape/volume state
        vol_rate = None
        vol_weak = False
        if volume_monitor and underlying:
            try:
                v = volume_monitor.analyze(underlying)
                if v:
                    vol_rate = v.rate_ratio
                    if v.rate_ratio < THESIS_VOL_RATE_MAX:
                        vol_weak = True
            except Exception:
                pass

        tape_conf = None
        tape_dir = None
        tape_not_supporting = False
        our_dir = 'buy' if trade.option_type == 'call' else 'sell'

        # Try priority tape first (richer signal for SPY/SPX/TSLA)
        if priority_tape and underlying:
            try:
                t = priority_tape.analyze(underlying)
                if t:
                    tape_dir = t.direction
                    tape_conf = t.confluence_score
                    if t.direction != our_dir:
                        tape_not_supporting = True
            except Exception:
                pass

        # Fallback to regular tape reader for non-priority symbols (AAPL,
        # PLTR, IWM, etc.) — direction is enough for the thesis check
        if tape_dir is None and fallback_tape and underlying:
            try:
                t = fallback_tape.analyze(underlying)
                if t:
                    tape_dir = t.direction
                    # Regular TapeSignal doesn't have confluence_score —
                    # use strength × 100 as a proxy for display
                    tape_conf = int((t.strength or 0) * 100)
                    if t.direction != our_dir:
                        tape_not_supporting = True
            except Exception:
                pass

        # Either condition recovered → WIPE the watch (zero tolerance for flicker)
        if not (vol_weak and tape_not_supporting):
            if tid in self._thesis_watches:
                state = self._thesis_watches[tid]
                log.info(
                    f'[{trade.symbol}] thesis watch RESET after {state["checks"]} bad checks '
                    f'({(now - state["start"]).total_seconds():.0f}s) — '
                    f'now vol={vol_rate if vol_rate is not None else "?"} '
                    f'tape={tape_dir or "?"} conf={tape_conf if tape_conf is not None else "?"}'
                )
                self._thesis_watches.pop(tid, None)
            return None

        # Both conditions bad — start or extend the watch
        state = self._thesis_watches.get(tid)
        if state is None:
            self._thesis_watches[tid] = {
                'start':      now,
                'checks':     1,
                'last_vol':   vol_rate,
                'last_conf':  tape_conf,
                'last_pnl':   pnl_pct,
                'symbol':     trade.symbol,
                'underlying': underlying,
            }
            log.info(
                f'[{trade.symbol}] 🔍 thesis watch START @ {pnl_pct*100:+.1f}% '
                f'(vol {vol_rate:.2f}× < {THESIS_VOL_RATE_MAX}, tape {tape_dir} conf={tape_conf}) — '
                f'need {THESIS_SUSTAINED_SECONDS}s + {THESIS_MIN_CHECKS} checks sustained'
            )
            return None

        # Extend watch — record this check
        state['checks']    += 1
        state['last_vol']   = vol_rate
        state['last_conf']  = tape_conf
        state['last_pnl']   = pnl_pct
        elapsed = (now - state['start']).total_seconds()

        # Periodic progress log
        if state['checks'] % THESIS_PROGRESS_LOG_EVERY == 0:
            log.info(
                f'[{trade.symbol}] thesis watch: {elapsed:.0f}/{THESIS_SUSTAINED_SECONDS}s, '
                f'check {state["checks"]}/{THESIS_MIN_CHECKS} — still bad '
                f'(vol {vol_rate:.2f}×, tape {tape_dir} conf={tape_conf}, pnl {pnl_pct*100:+.1f}%)'
            )

        # Need BOTH thresholds satisfied: enough time AND enough samples
        if elapsed < THESIS_SUSTAINED_SECONDS or state['checks'] < THESIS_MIN_CHECKS:
            return None

        # Survived — exit
        try:
            self.orders._close_trade(trade, current, 'thesis_broken')
            self._thesis_watches.pop(tid, None)
            return ExitDecision(
                trade_id=tid,
                action='stop',
                quantity=trade.contracts or 0,
                reason=(f'thesis broken @ {pnl_pct*100:+.1f}% — sustained {elapsed:.0f}s / '
                        f'{state["checks"]} checks: vol {vol_rate:.2f}×, tape {tape_dir} conf={tape_conf}'),
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            log.error(f'[{trade.symbol}] Thesis-broken close failed: {e}')
            return None

    # ── Profit-protect check ─────────────────────────────────────────────────

    def _maybe_profit_protect(
        self,
        trade,
        current: float,
        pnl_pct: float,
        priority_tape,
        volume_monitor,
        fallback_tape=None,
    ) -> Optional[ExitDecision]:
        """
        Data-driven full-exit for positions in profit where the trim ladder
        can't help (1 contract, or all tiers already hit).

        Fires when in profit ≥ +5% AND for 30+ seconds sustained:
          • Volume rate < 0.7× normal, OR
          • Tape direction has flipped away from our position
        (Either condition alone — vs thesis-broken's AND — because we're
        protecting realized gains and faster action is justified.)

        Closes the FULL remaining position.
        """
        from datetime import datetime, timezone

        tid = trade.trade_id

        # Only protect meaningful profits
        if pnl_pct < PROFIT_PROTECT_MIN_PCT:
            self._profit_watches.pop(tid, None)
            return None

        # If the trim ladder could fire naturally, let it. We only act
        # when (a) single contract or (b) every tier already hit.
        if trade.contracts and trade.contracts > 1:
            next_unhit_threshold = None
            for idx, (threshold, _) in enumerate(TRIM_TIERS):
                if idx not in trade.tier_hits and pnl_pct >= threshold:
                    next_unhit_threshold = threshold
                    break
            if next_unhit_threshold is not None:
                self._profit_watches.pop(tid, None)
                return None

        underlying = self._underlying.get(tid) or self._derive_underlying(trade.symbol)
        our_dir = 'buy' if trade.option_type == 'call' else 'sell'
        now = datetime.now(timezone.utc)

        # Read volume + tape
        vol_rate = None
        vol_weak = False
        if volume_monitor and underlying:
            try:
                v = volume_monitor.analyze(underlying)
                if v:
                    vol_rate = v.rate_ratio
                    if v.rate_ratio < PROFIT_PROTECT_VOL_THRESHOLD:
                        vol_weak = True
            except Exception:
                pass

        tape_dir = None
        tape_conf = None
        tape_weak = False
        if priority_tape and underlying:
            try:
                t = priority_tape.analyze(underlying)
                if t:
                    tape_dir = t.direction
                    tape_conf = t.confluence_score
                    if t.direction != our_dir:
                        tape_weak = True
            except Exception:
                pass
        if tape_dir is None and fallback_tape and underlying:
            try:
                t = fallback_tape.analyze(underlying)
                if t:
                    tape_dir = t.direction
                    tape_conf = int((t.strength or 0) * 100)
                    if t.direction != our_dir:
                        tape_weak = True
            except Exception:
                pass

        # EITHER signal weak triggers — vs thesis-broken's AND
        if not (vol_weak or tape_weak):
            if tid in self._profit_watches:
                state = self._profit_watches[tid]
                log.info(
                    f'[{trade.symbol}] profit-protect RESET after '
                    f'{state["checks"]} bad checks ({(now - state["start"]).total_seconds():.0f}s) — '
                    f'support recovered (vol {vol_rate}, tape {tape_dir} conf {tape_conf})'
                )
            self._profit_watches.pop(tid, None)
            return None

        # Start or extend the watch
        state = self._profit_watches.get(tid)
        if state is None:
            self._profit_watches[tid] = {
                'start':      now,
                'checks':     1,
                'last_vol':   vol_rate,
                'last_conf':  tape_conf,
                'last_pnl':   pnl_pct,
                'symbol':     trade.symbol,
                'underlying': underlying,
                'vol_weak':   vol_weak,
                'tape_weak':  tape_weak,
            }
            reasons = []
            if vol_weak:  reasons.append(f'vol {vol_rate:.2f}× < {PROFIT_PROTECT_VOL_THRESHOLD}')
            if tape_weak: reasons.append(f'tape {tape_dir} (not our {our_dir})')
            log.info(
                f'[{trade.symbol}] 💰 profit-protect watch START @ {pnl_pct*100:+.1f}% — '
                f'{", ".join(reasons)} — exit after {PROFIT_PROTECT_SUSTAINED_SECONDS}s + '
                f'{PROFIT_PROTECT_MIN_CHECKS} checks sustained'
            )
            return None

        state['checks']   += 1
        state['last_vol']  = vol_rate
        state['last_conf'] = tape_conf
        state['last_pnl']  = pnl_pct
        state['vol_weak']  = vol_weak
        state['tape_weak'] = tape_weak
        elapsed = (now - state['start']).total_seconds()

        if elapsed < PROFIT_PROTECT_SUSTAINED_SECONDS or state['checks'] < PROFIT_PROTECT_MIN_CHECKS:
            return None

        # Survived — fire full exit
        reasons = []
        if vol_weak:  reasons.append(f'vol {vol_rate:.2f}×')
        if tape_weak: reasons.append(f'tape {tape_dir}')
        try:
            self.orders._close_trade(trade, current, 'profit_protect')
            self._profit_watches.pop(tid, None)
            return ExitDecision(
                trade_id=tid,
                action='stop',
                quantity=trade.contracts or 0,
                reason=(f'profit-protect @ {pnl_pct*100:+.1f}%: {", ".join(reasons)} '
                        f'sustained {elapsed:.0f}s / {state["checks"]} checks'),
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            log.error(f'[{trade.symbol}] profit-protect close failed: {e}')
            return None

    def active_profit_watches(self) -> List[dict]:
        """Snapshot of in-progress profit-protect watches for dashboard."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        out = []
        for tid, s in self._profit_watches.items():
            elapsed = (now - s['start']).total_seconds()
            out.append({
                'trade_id':   tid,
                'symbol':     s.get('symbol'),
                'underlying': s.get('underlying'),
                'elapsed_s':  round(elapsed, 1),
                'required_s': PROFIT_PROTECT_SUSTAINED_SECONDS,
                'checks':     s['checks'],
                'required_checks': PROFIT_PROTECT_MIN_CHECKS,
                'last_vol':   s.get('last_vol'),
                'last_conf':  s.get('last_conf'),
                'last_pnl':   s.get('last_pnl'),
                'vol_weak':   s.get('vol_weak'),
                'tape_weak':  s.get('tape_weak'),
                'progress':   min(1.0, elapsed / PROFIT_PROTECT_SUSTAINED_SECONDS),
                'type':       'profit_protect',
            })
        return out

    def active_thesis_watches(self) -> List[dict]:
        """Snapshot of in-progress thesis watches for dashboard display."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        out = []
        for tid, s in self._thesis_watches.items():
            elapsed = (now - s['start']).total_seconds()
            out.append({
                'trade_id':   tid,
                'symbol':     s.get('symbol'),
                'underlying': s.get('underlying'),
                'elapsed_s':  round(elapsed, 1),
                'required_s': THESIS_SUSTAINED_SECONDS,
                'checks':     s['checks'],
                'required_checks': THESIS_MIN_CHECKS,
                'last_vol':   s.get('last_vol'),
                'last_conf':  s.get('last_conf'),
                'last_pnl':   s.get('last_pnl'),
                'progress':   min(1.0, elapsed / THESIS_SUSTAINED_SECONDS),
                'type':       'thesis_broken',
            })
        return out

    def all_active_watches(self) -> List[dict]:
        """Combined list of all in-progress defensive exit watches."""
        return self.active_thesis_watches() + self.active_profit_watches()

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

    # ── Volume-aware runner protection ───────────────────────────────────────

    def _maybe_volume_trim(
        self,
        trade,
        current: float,
        pnl_pct: float,
        priority_tape,
        volume_monitor,
    ) -> Optional[ExitDecision]:
        """
        Accelerate the next trim tier when volume or tape weakens while
        we're in profit (runner protection).

        Only fires when:
          • Trade is profitable (pnl > 0)
          • At least one natural tier has fired (i.e. it's a "runner")
          • Pnl is at least 5pp past the last hit tier (gives the trade room)
          • Volume rate dropped below 70% of normal
            OR underlying tape lost direction/confluence
        """
        if pnl_pct <= 0:
            return None
        if not trade.tier_hits:
            return None  # natural tier 1 must fire first

        # Find next unhit tier
        next_idx = None
        for idx, _ in enumerate(TRIM_TIERS):
            if idx not in trade.tier_hits:
                next_idx = idx
                break
        if next_idx is None:
            return None

        # Require buffer past last hit tier so we don't immediately fire
        # tier N+1 the moment tier N completes
        last_hit_threshold = max(TRIM_TIERS[i][0] for i in trade.tier_hits)
        if pnl_pct < last_hit_threshold + VOLUME_TRIM_BUFFER_PCT:
            return None

        # Read volume and tape state
        underlying = self._underlying.get(trade.trade_id) or self._derive_underlying(trade.symbol)
        vol_weak  = False
        tape_weak = False
        vol_label  = '—'
        tape_label = '—'

        if volume_monitor and underlying:
            try:
                vol = volume_monitor.analyze(underlying)
                if vol and vol.rate_ratio < VOLUME_TRIM_RATE_THRESHOLD:
                    vol_weak = True
                    vol_label = f'rate {vol.rate_ratio:.2f}×'
            except Exception:
                pass

        if priority_tape and underlying:
            try:
                tape = priority_tape.analyze(underlying)
                if tape:
                    our_dir = 'buy' if trade.option_type == 'call' else 'sell'
                    # Weak = lost direction OR confluence dropped low
                    if tape.direction != our_dir and tape.confluence_score < VOLUME_TRIM_TAPE_CONF_THRESHOLD + 20:
                        tape_weak = True
                        tape_label = f'dir={tape.direction}, conf={tape.confluence_score}'
                    elif tape.confluence_score < VOLUME_TRIM_TAPE_CONF_THRESHOLD:
                        tape_weak = True
                        tape_label = f'conf={tape.confluence_score}'
            except Exception:
                pass

        if not (vol_weak or tape_weak):
            return None

        # Fire the next tier early
        threshold, frac = TRIM_TIERS[next_idx]
        qty = max(1, math.floor(trade.original_contracts * frac))
        remaining = trade.contracts or 0
        if remaining <= 1:
            trade.tier_hits.append(next_idx)
            return None
        qty = min(qty, remaining - 1)  # always preserve runner
        if qty <= 0:
            trade.tier_hits.append(next_idx)
            return None

        reasons = []
        if vol_weak:  reasons.append(f'vol weak ({vol_label})')
        if tape_weak: reasons.append(f'tape weak ({tape_label})')
        why = ', '.join(reasons)

        try:
            self.orders._partial_close_option(trade, qty, f'vol_trim_t{next_idx+1}')
            trade.tier_hits.append(next_idx)
            return ExitDecision(
                trade_id=trade.trade_id,
                action='trim',
                quantity=qty,
                reason=f'EARLY tier{next_idx+1} @ {pnl_pct*100:.0f}% — {why}',
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            log.error(f'[{trade.symbol}] Volume trim failed: {e}')
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
