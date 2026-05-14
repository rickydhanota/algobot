"""
Trading bot main loop.

Flow:
  Pre-market  → load watchlist bars, reset VWAP accumulators
  09:30-09:45 → capture opening range (ORB), watch tape — no trades
  10:00+      → evaluate signals every 30 seconds, place orders
  15:30       → stop new positions
  16:00       → close all, print session report

Run:
  python main.py
  python main.py --watchlist AAPL TSLA NVDA   # override watchlist
  python main.py --paper false                 # live trading (careful!)
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import pytz
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from data.alpaca_client import AlpacaClients
from data.market_data import MarketDataManager
from signals.tape_reader import TapeReader
from signals.priority_tape import PriorityTapeReader, PRIORITY_SYMBOLS
from signals.technical import TechnicalAnalyzer
from signals.volume_monitor import VolumeMonitor
from signals.entry_quality import EntryQualityChecker, MIN_QUALITY_TO_TRADE
from signals import session as session_mod
from learning.adaptive import AdaptiveLearner
from strategy.risk_manager import RiskManager
from strategy.stock_strategy import StockStrategy
from strategy.options_strategy import OptionsStrategy
from execution.order_manager import OrderManager, ActiveTrade
from execution.exit_manager import ExitManager
from performance.tracker import PerformanceTracker
from dashboard_server import DashboardServer
from news.alpaca_news import AlpacaNewsFeed
from news.earnings import EarningsCalendar, IPOCalendar
from news.sentiment import score_articles, score_text
from news.macro_feeds import MacroFeedAggregator
from news.macro_analysis import assess as assess_macro, MacroRisk
from intel.aggregator import IntelAggregator

ET = pytz.timezone('America/New_York')
log = logging.getLogger('bot')
console = Console()


def et_now() -> datetime:
    return datetime.now(ET)


def market_phase(now: datetime) -> str:
    t = (now.hour, now.minute)
    if t < (9, 30):   return 'premarket'
    if t < (9, 45):   return 'orb_capture'
    if t < (10, 0):   return 'settle'
    if t < (15, 30):  return 'active'
    if t < (16, 0):   return 'closing'
    return 'closed'


# Day-trade-only policy: force-close all positions before market close
# so nothing gets held overnight.
EOD_CLOSE_TIME_ET = (15, 55)   # 3:55 PM Eastern — 5 min before market close


class TradingBot:
    def __init__(self, watchlist: List[str], use_options: bool = True):
        # Ensure SPY and TSLA are always in the watchlist (SPX is proxied from SPY)
        wl = list(dict.fromkeys(list(watchlist) + ['SPY', 'TSLA']))
        self.watchlist = wl
        self.use_options = use_options

        self.market_data = MarketDataManager()
        self.tape = TapeReader()
        self.priority_tape = PriorityTapeReader()
        self.volume = VolumeMonitor()
        self.entry_quality = EntryQualityChecker()
        self.adaptive = AdaptiveLearner()
        self.analyzer = TechnicalAnalyzer()
        self.risk = RiskManager()
        self.stock_strategy = StockStrategy()
        self.options_strategy = OptionsStrategy()
        self.orders = OrderManager(self.risk)
        self.exit_manager = ExitManager(self.orders)
        self.tracker = PerformanceTracker()
        self.dashboard_server = DashboardServer(self)

        # News & calendars
        self.news = AlpacaNewsFeed()
        self.earnings = EarningsCalendar()
        self.ipos = IPOCalendar()
        self.macro_feeds = MacroFeedAggregator()
        self.macro_risk: MacroRisk = MacroRisk()
        self.intel = IntelAggregator()

        self._traded_today: Set[str] = set()
        # Ring buffer of recent rejections — what didn't quite fire and why
        from collections import deque as _deque
        self._rejections = _deque(maxlen=60)
        self._running = False
        self._paused = False                     # set by dashboard /api/pause
        self._force_close_requested = False      # set by dashboard /api/close-all
        self._phase = 'premarket'
        self._bars_cache = {}
        self._live_prices: Dict[str, float] = {}
        self._live_vwap: Dict[str, float] = {}
        self._last_news_fetch: Optional[datetime] = None
        self._last_macro_fetch: Optional[datetime] = None
        self._last_intel_fetch: Optional[datetime] = None
        self._sentiment_cache: Dict[str, dict] = {}

    # ── Market data streaming ─────────────────────────────────────────────────

    async def _stream_trades(self):
        """Subscribe to real-time trade stream for tape reading."""
        stream = AlpacaClients.stock_stream()

        async def on_trade(trade):
            sym = trade.symbol
            price = float(trade.price)
            size = int(trade.size)
            self._live_prices[sym] = price
            vwap = self.market_data.update_vwap(sym, price, size)
            self._live_vwap[sym] = vwap
            self.tape.record_trade(sym, price, size)
            # Priority symbols get richer tracking
            if sym in PRIORITY_SYMBOLS or sym == 'SPY':
                self.priority_tape.record_trade(sym, price, size)
                self.volume.record_trade(sym, size)
                self.entry_quality.record_price(sym, price)
                if sym == 'SPY':
                    # also feed SPX proxy
                    self.entry_quality.record_price('SPX', price)

        stream.subscribe_trades(on_trade, *self.watchlist)
        # stream.run() is sync — it spawns its own event loop, which fails
        # silently when called from inside ours. Use _run_forever() directly.
        await stream._run_forever()

    # ── Session setup ─────────────────────────────────────────────────────────

    def _load_bars(self):
        log.info(f'Loading bars for {len(self.watchlist)} symbols...')
        for sym in self.watchlist:
            try:
                bars = self.market_data.get_bars(sym, lookback_days=5, force_refresh=True)
                self._bars_cache[sym] = bars
                # Reset VWAP accumulator for new session
                self.market_data.reset_vwap(sym)
            except Exception as e:
                log.warning(f'[{sym}] Could not load bars: {e}')

    def _capture_orb(self):
        log.info('Capturing opening range...')
        for sym in self.watchlist:
            bars = self._bars_cache.get(sym)
            if bars is not None and not bars.empty:
                orb = self.market_data.compute_orb_from_bars(sym, bars)
                if orb:
                    log.info(f'[{sym}] ORB: H={orb["high"]:.2f} L={orb["low"]:.2f}')

    def _update_account(self):
        try:
            acct = AlpacaClients.get_account()
            self.risk.update_account(float(acct.equity))
        except Exception:
            pass

    def _refresh_option_quotes(self):
        """
        Fetch the latest mid price for every open option position and
        write it into self._live_prices keyed by the OCC symbol. Without
        this, the trim ladder, dynamic stops, and dashboard P&L all read
        stale entry prices because we don't stream individual contracts.
        """
        open_options = [
            t for t in self.orders.get_open_trades() if t.asset_type == 'option'
        ]
        if not open_options:
            return

        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            client = AlpacaClients.option_hist()
            if client is None:
                return
            symbols = list({t.symbol for t in open_options})
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = client.get_option_latest_quote(req)
            for sym, q in quotes.items():
                bid = getattr(q, 'bid_price', 0) or 0
                ask = getattr(q, 'ask_price', 0) or 0
                if bid and ask:
                    self._live_prices[sym] = (bid + ask) / 2.0
                elif ask:
                    self._live_prices[sym] = ask
                elif bid:
                    self._live_prices[sym] = bid
        except Exception as e:
            log.debug(f'Option quote refresh failed: {e}')

    def _refresh_news_if_due(self):
        """Refresh news every NEWS_FETCH_INTERVAL_MIN minutes."""
        now_utc = datetime.now(timezone.utc)
        if (self._last_news_fetch and
                (now_utc - self._last_news_fetch).total_seconds() < config.NEWS_FETCH_INTERVAL_MIN * 60):
            return

        self.news.fetch(self.watchlist, hours=config.NEWS_LOOKBACK_HOURS)
        for sym in self.watchlist:
            arts = self.news.for_symbol(sym)
            self._sentiment_cache[sym] = score_articles(arts)
        self._last_news_fetch = now_utc

    def _refresh_macro_if_due(self):
        """Refresh Fed / WH / Treasury feeds and reassess macro risk."""
        now_utc = datetime.now(timezone.utc)
        if (self._last_macro_fetch and
                (now_utc - self._last_macro_fetch).total_seconds() < config.MACRO_FETCH_INTERVAL_MIN * 60):
            return

        self.macro_feeds.fetch_all(lookback_hours=config.MACRO_LOOKBACK_HOURS)
        self.macro_risk = assess_macro(self.macro_feeds)
        self._apply_macro_risk()
        self._last_macro_fetch = now_utc

        log.info(
            f'Macro: fed={self.macro_risk.fed_bias} ({self.macro_risk.fed_score:+.2f})  '
            f'tariff={self.macro_risk.tariff_risk}  geo={self.macro_risk.geopolitical_risk}  '
            f'FOMC={self.macro_risk.next_fomc_days}d  risk={self.macro_risk.risk_level}'
        )

    def _refresh_intel_if_due(self):
        """Refresh 13F filings + congressional trades every INTEL_REFRESH_HOURS."""
        now_utc = datetime.now(timezone.utc)
        if (self._last_intel_fetch and
                (now_utc - self._last_intel_fetch).total_seconds() < config.INTEL_REFRESH_HOURS * 3600):
            return

        try:
            self.intel.refresh(self.watchlist)
            self._last_intel_fetch = now_utc
        except Exception as e:
            log.warning(f'Intel refresh failed: {e}')

    def _check_eod_close(self):
        """
        Day-trade-only policy: force-close everything still open after 3:50 PM ET.
        Idempotent — calling repeatedly is fine; close_all() only acts on 'open'
        trades. We log only when there are actually positions to close.
        """
        now = et_now()
        if (now.hour, now.minute) < EOD_CLOSE_TIME_ET:
            return
        open_trades = self.orders.get_open_trades()
        if not open_trades:
            return
        log.warning(
            f'⏰ EOD force-close ({now.strftime("%H:%M ET")}): closing '
            f'{len(open_trades)} position(s) to honour day-trade-only policy'
        )
        self.orders.close_all()

    def _apply_macro_risk(self):
        """Translate MacroRisk + current session window into risk-manager state."""
        macro_mult = self.macro_risk.risk_multiplier
        session_mult = session_mod.size_multiplier(session_mod.current_window(et_now()))
        # Both multipliers compound — risk_manager.max_risk_dollars reads this
        self.risk.macro_risk_multiplier = macro_mult * session_mult
        halt = ''
        if config.HALT_ON_FOMC_DAY and self.macro_risk.fomc_today:
            halt = 'FOMC announcement today'
        elif config.HALT_ON_GEOPOLITICAL and self.macro_risk.geopolitical_risk:
            halt = 'major geopolitical event'
        self.risk.macro_halt_reason = halt

    def _news_filter(self, symbol: str) -> tuple[bool, int, str]:
        """
        Apply news/earnings filter.
        Returns: (allowed, score_adjustment, reason)
        """
        # Earnings circuit breaker
        if self.earnings.has_earnings_within(symbol, days=config.SKIP_EARNINGS_DAYS):
            days = self.earnings.days_until_earnings(symbol) or 0
            return False, 0, f'earnings in {days}d'

        # News sentiment
        sent = self._sentiment_cache.get(symbol, {})
        score = sent.get('score', 0.0)
        adjustment = 0

        if score <= config.NEWS_NEGATIVE_THRESHOLD:
            return False, 0, f'negative news ({score:+.2f})'

        if score >= config.NEWS_POSITIVE_BOOST:
            adjustment = config.NEWS_SCORE_BONUS

        return True, adjustment, ''

    # ── Signal evaluation ─────────────────────────────────────────────────────

    def _track_rejection(self, symbol: str, category: str, detail: str = '',
                         score: Optional[int] = None, threshold: Optional[int] = None):
        """Log a near-miss so we can see what's almost firing on the dashboard."""
        self._rejections.append({
            'symbol':    symbol,
            'category':  category,
            'detail':    detail,
            'score':     score,
            'threshold': threshold,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

    def _evaluate_symbol(self, sym: str) -> Optional[dict]:
        """Return a trade setup dict if symbol has a high-quality signal."""
        if sym in self._traded_today:
            return None

        # Session window — drives score boost/penalty by time of day
        session_window = session_mod.current_window(et_now())
        if session_window == 'off_hours':
            return None
        session_adj = session_mod.score_adjustment(session_window)

        # News & earnings filter — gate BEFORE expensive analysis
        allowed, score_adj, reason = self._news_filter(sym)
        if not allowed:
            log.debug(f'[{sym}] Skipped: {reason}')
            self._track_rejection(sym, 'news_filter', reason)
            return None

        price = self._live_prices.get(sym)
        if not price:
            return None

        bars = self._bars_cache.get(sym)
        if bars is None or bars.empty:
            return None

        try:
            bars = self.market_data.get_bars(sym)
            self._bars_cache[sym] = bars
        except Exception:
            pass

        vwap = self._live_vwap.get(sym, 0.0)
        orb = self.market_data.get_orb(sym)

        tech = self.analyzer.analyze(sym, bars, price, vwap, orb)
        if tech is None:
            return None

        tape_signal = self.tape.analyze(sym)

        # ── Priority tape boost for SPY / SPX / TSLA
        priority_sig = None
        priority_boost = 0
        volume_state = None
        if sym in PRIORITY_SYMBOLS:
            # Hard volume gate: skip thin or stale tape outright
            ok, why = self.volume.is_tradeable(sym)
            if not ok:
                log.debug(f'[{sym}] Volume gate: {why}')
                self._track_rejection(sym, 'volume_gate', why)
                return None
            volume_state = self.volume.analyze(sym)
            priority_sig = self.priority_tape.analyze(sym)
            if priority_sig and priority_sig.is_strong:
                priority_boost = 10
            elif priority_sig and priority_sig.is_moderate:
                priority_boost = 5

        # ────────────────────────────────────────────────────────────────────
        # OPTIONS-FIRST EVALUATION
        # The bot is an options trader. Every symbol tries options first.
        # Stocks are computed as directional context but only entered when
        # OPTIONS_ONLY_MODE is False AND options aren't viable.
        # ────────────────────────────────────────────────────────────────────

        # Compute the underlying directional signal via the stock strategy.
        # This determines whether the options play is bullish or bearish and
        # gives us a quality score for the underlying move.
        underlying_setup = self.stock_strategy.evaluate(tech, tape_signal)

        # ── PRIMARY PATH: try options first ──────────────────────────────────
        if self.use_options:
            chain = self.market_data.get_option_chain(sym)
            if chain:
                opt_setup = self.options_strategy.evaluate(sym, chain, tech, tape_signal)
                if opt_setup and opt_setup.is_valid:
                    # Stack all bonuses on options score
                    intel_boost = min(
                        config.INTEL_SCORE_BOOST_MAX,
                        self.intel.score_boost(sym, 'long' if opt_setup.option_type == 'call' else 'short'),
                    )
                    opt_setup.score = max(0, min(
                        100,
                        opt_setup.score + priority_boost + score_adj + intel_boost + session_adj,
                    ))
                    if score_adj > 0:    opt_setup.notes.append(f'news+{score_adj}')
                    if priority_boost:   opt_setup.notes.append(f'priority+{priority_boost}')
                    if intel_boost:      opt_setup.notes.append(f'smartmoney+{intel_boost}')
                    if session_adj:      opt_setup.notes.append(f'{session_window}{session_adj:+d}')

                    # Re-check threshold (per-session adaptive adjustment)
                    strat_name = f'options_{opt_setup.option_type}'
                    thr_adj = self.adaptive.threshold_adjustment(sym, strat_name, session_window=session_window)
                    threshold = config.MIN_SIGNAL_SCORE + thr_adj
                    if opt_setup.score < threshold:
                        self._track_rejection(
                            sym, 'score_below_threshold',
                            f'{strat_name} {opt_setup.option_type.upper()} ${opt_setup.strike:.0f}',
                            score=opt_setup.score, threshold=threshold,
                        )
                        return None

                    self.adaptive.record_signal(
                        signal_id=f'{sym}-{strat_name}-{int(opt_setup.timestamp.timestamp())}',
                        symbol=sym,
                        strategy=strat_name,
                        score=opt_setup.score,
                        imbalance=tape_signal.imbalance if tape_signal else 0.0,
                        rvol=tech.rvol if tech else 0.0,
                        confluence_score=priority_sig.confluence_score if priority_sig else 0,
                        direction='long' if opt_setup.option_type == 'call' else 'short',
                        session_window=session_window,
                    )
                    return {'type': 'option', 'setup': opt_setup, 'tech': tech, 'tape': tape_signal}

        # ── No viable options. In options-only mode, stop here. ──────────────
        if config.OPTIONS_ONLY_MODE:
            if underlying_setup:
                log.debug(
                    f'[{sym}] Underlying signal valid ({underlying_setup.strategy} '
                    f'score={underlying_setup.score}) but no viable options — skipping (OPTIONS_ONLY)'
                )
                self._track_rejection(
                    sym, 'no_valid_options',
                    (f'underlying {underlying_setup.strategy} ok (stock score {underlying_setup.score}); '
                     f'no option contract met DTE 7-21 + |Δ| 0.25-0.50 + spread <10% + score ≥{config.MIN_SIGNAL_SCORE}'),
                )
            return None

        # ── FALLBACK PATH: stock trade (only when OPTIONS_ONLY_MODE = False) ─
        if underlying_setup:
            stock_setup = underlying_setup
            intel_boost = min(
                config.INTEL_SCORE_BOOST_MAX,
                self.intel.score_boost(sym, stock_setup.direction),
            )
            stock_setup.score = max(0, min(100,
                stock_setup.score + score_adj + priority_boost + intel_boost + session_adj,
            ))
            if score_adj > 0:    stock_setup.notes.append(f'news+{score_adj}')
            if priority_boost:   stock_setup.notes.append(f'priority+{priority_boost}')
            if intel_boost:      stock_setup.notes.append(f'smartmoney+{intel_boost}')
            if session_adj:      stock_setup.notes.append(f'{session_window}{session_adj:+d}')

            adj = self.adaptive.threshold_adjustment(sym, stock_setup.strategy, session_window=session_window)
            if stock_setup.score < config.MIN_SIGNAL_SCORE + adj:
                return None

            self.adaptive.record_signal(
                signal_id=f'{sym}-{stock_setup.strategy}-{int(stock_setup.timestamp.timestamp())}',
                symbol=sym,
                strategy=stock_setup.strategy,
                score=stock_setup.score,
                imbalance=tape_signal.imbalance if tape_signal else 0.0,
                rvol=stock_setup.rvol,
                confluence_score=priority_sig.confluence_score if priority_sig else 0,
                direction=stock_setup.direction,
                session_window=session_window,
            )
            return {'type': 'stock', 'setup': stock_setup, 'tech': tech, 'tape': tape_signal}

        return None

    def _entry_quality_ok(self, sym: str, result: dict) -> bool:
        """Run pre-trade entry checks; reject chasing, wide spreads, etc."""
        setup = result['setup']
        tech = result.get('tech')
        tape = result.get('tape')

        # For stocks: validate against the underlying symbol.
        # For options: validate against the underlying instead of the option symbol.
        underlying = sym  # main loop iterates watchlist (underlying) symbols
        price = self._live_prices.get(underlying, 0)
        vwap = self._live_vwap.get(underlying, 0)

        # Best-effort bid/ask from cached snapshot
        snap = self.market_data._quotes.get(underlying)
        bid = snap.bid if snap else 0
        ask = snap.ask if snap else 0

        # Volume health for priority symbols
        vol_healthy = True
        if underlying in PRIORITY_SYMBOLS:
            v = self.volume.analyze(underlying)
            vol_healthy = v.is_healthy if v else True

        direction = 'long' if setup.direction in ('long', 'buy') else 'short'
        imb = tape.imbalance if tape else 0

        verdict = self.entry_quality.check(
            symbol=underlying,
            direction=direction,
            entry_price=price,
            vwap=vwap,
            bid=bid,
            ask=ask,
            imbalance=imb,
            volume_healthy=vol_healthy,
        )

        if not verdict.allowed:
            log.info(f'[{underlying}] Entry rejected: {verdict.reason}')
            self._track_rejection(
                underlying, 'entry_quality', verdict.reason,
                score=verdict.score,
            )
            return False

        if hasattr(setup, 'notes'):
            setup.notes.append(f'entryQ={verdict.score}')
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _trading_loop(self):
        while self._running:
            now = et_now()
            phase = market_phase(now)
            self._phase = phase

            if phase == 'closed':
                # Market is closed — stop trading but KEEP the dashboard
                # server alive so the user can review the session.
                # Still refresh option quotes once a minute so dashboard
                # reflects the most recent fills before/at the close.
                self._refresh_option_quotes()
                await asyncio.sleep(30)
                continue

            if phase == 'premarket':
                self.earnings.refresh(self.watchlist)
                self._refresh_news_if_due()
                self._refresh_macro_if_due()
                self._refresh_intel_if_due()
                await asyncio.sleep(30)
                continue

            if phase == 'orb_capture':
                self._capture_orb()
                self._refresh_news_if_due()
                self._refresh_macro_if_due()
                await asyncio.sleep(60)
                continue

            if phase == 'settle':
                await asyncio.sleep(10)
                continue

            if phase == 'closing':
                # Only manage existing positions, no new trades
                self._update_account()
                self._refresh_option_quotes()
                self.orders.check_exits(self._live_prices)
                self._check_eod_close()      # day-trade-only enforcement
                await asyncio.sleep(15)
                continue

            # ── Force close-all request from dashboard?
            if self._force_close_requested:
                log.warning('🛑 Force close-all in progress')
                self.orders.close_all()
                self._force_close_requested = False

            # ── Active trading
            allowed, reason = self.risk.is_trading_allowed()
            if not allowed:
                log.warning(f'Trading halted: {reason}')
                await asyncio.sleep(60)
                continue

            self._update_account()
            self._refresh_news_if_due()
            self._refresh_macro_if_due()
            self._apply_macro_risk()        # refresh session-window multiplier each tick
            self._refresh_option_quotes()   # keep open-option prices fresh for P&L + stops
            self.orders.check_exits(self._live_prices)
            self._check_eod_close()    # honours day-trade-only policy

            # Tiered trimming + tape-aware dynamic stops (options only)
            decisions = self.exit_manager.process(
                self._live_prices, self.priority_tape, self.volume,
            )
            for d in decisions:
                log.info(f'[exit] {d.action}={d.quantity} ({d.reason}) pnl={d.pnl_pct*100:+.1f}%')

            # If paused, skip new entries but keep managing existing
            if self._paused:
                await asyncio.sleep(5)
                continue

            # Scan watchlist for setups
            for sym in self.watchlist:
                result = self._evaluate_symbol(sym)
                if result is None:
                    continue

                # Pre-trade entry-quality gate: don't chase, don't enter
                # on wide spreads, don't trade far from VWAP
                if not self._entry_quality_ok(sym, result):
                    continue

                if result['type'] == 'stock':
                    setup = result['setup']
                    trade = self.orders.place_stock_trade(setup)
                    if trade:
                        self._traded_today.add(sym)
                        self._record_closed_on_exit(trade, setup.score, setup.rvol, setup.tape_imbalance)

                elif result['type'] == 'option':
                    setup = result['setup']
                    trade = self.orders.place_options_trade(setup)
                    if trade:
                        self._traded_today.add(sym)
                        # Register underlying so dynamic stop can read its tape
                        self.exit_manager.register_underlying(trade.trade_id, sym)
                    else:
                        # Order failed (risk manager, sizing, or broker error)
                        self._track_rejection(
                            sym, 'risk_or_order_fail',
                            f'{setup.option_type.upper()} ${setup.strike:.0f} premium ${setup.premium:.2f}',
                        )

            # Faster loop iteration for snappier trimming/stops
            await asyncio.sleep(5)

    def _record_closed_on_exit(self, trade: ActiveTrade, score: int, rvol: float, imb: float):
        """Record closed trades into tracker AND adaptive learner."""
        closed = self.orders.get_closed_trades()
        for t in closed:
            if t.realized_pnl is None:
                continue
            self.tracker.record(t, score=score, rvol=rvol, tape_imbalance=imb)
            # Feed outcome to adaptive learner — uses our signal_id format
            sig_id = f'{t.symbol}-{t.strategy}-{int(t.entry_time.timestamp())}'
            self.adaptive.record_outcome(sig_id, t.realized_pnl)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _build_dashboard(self) -> Layout:
        now = et_now()
        stats = self.tracker.today_stats()
        daily_pnl = self.risk.daily_pnl
        equity = self.risk.account_size

        layout = Layout()
        layout.split_column(
            Layout(name='header', size=4),
            Layout(name='open', size=14),
            Layout(name='closed'),
        )

        # ── Header bar ───────────────────────────────────────────────────────
        pnl_color = 'green' if daily_pnl >= 0 else 'red'
        wr_str = f'{stats.win_rate*100:.0f}% ({stats.wins}W/{stats.losses}L)' if stats.total_trades else '—'
        pf_str = f'{stats.profit_factor:.2f}' if stats.total_trades else '—'
        header_text = (
            f"[bold white]AlgoBot[/bold white]   "
            f"Equity [cyan]${equity:,.2f}[/cyan]   "
            f"Day P&L [{pnl_color}]{daily_pnl:+.2f}[/{pnl_color}]   "
            f"Win Rate [yellow]{wr_str}[/yellow]   "
            f"Profit Factor [yellow]{pf_str}[/yellow]   "
            f"Phase [magenta]{self._phase.upper()}[/magenta]   "
            f"[dim]{now.strftime('%a %b %d  %H:%M:%S ET')}[/dim]"
        )
        layout['header'].update(Panel(header_text))

        # ── Open positions ────────────────────────────────────────────────────
        open_table = Table(
            title='[bold]Open Positions[/bold]',
            expand=True,
            show_lines=True,
        )
        open_table.add_column('Ticker',    style='bold cyan',  no_wrap=True)
        open_table.add_column('Type',      justify='center')
        open_table.add_column('Detail',    justify='center')   # strike + C/P, or STOCK
        open_table.add_column('Qty',       justify='right')
        open_table.add_column('Entry',     justify='right')
        open_table.add_column('Current',   justify='right')
        open_table.add_column('Stop',      justify='right',  style='dim')
        open_table.add_column('Target',    justify='right',  style='dim')
        open_table.add_column('Unreal P&L', justify='right')
        open_table.add_column('Opened',    justify='center', style='dim')

        for trade in self.orders.get_open_trades():
            curr = self._live_prices.get(trade.symbol, trade.entry_price)
            mult = 1 if trade.direction in ('long', 'buy') else -1
            upnl = (curr - trade.entry_price) * trade.shares * mult
            pnl_color = 'green' if upnl >= 0 else 'red'

            if trade.asset_type == 'option':
                type_str = '[magenta]OPTION[/magenta]'
                opt_label = trade.option_type.upper() if trade.option_type else '?'
                opt_color = 'green' if opt_label == 'CALL' else 'red'
                strike_str = f'${trade.strike:.0f}' if trade.strike else '?'
                detail = f'[{opt_color}]{opt_label}[/{opt_color}] {strike_str}'
                qty_str = f'{trade.contracts}c' if trade.contracts else f'{trade.shares//100}c'
            else:
                type_str = '[blue]STOCK[/blue]'
                detail = trade.direction.upper()
                qty_str = f'{trade.shares}sh'

            entry_time_et = trade.entry_time.astimezone(ET)
            opened_str = entry_time_et.strftime('%m/%d %H:%M')

            open_table.add_row(
                trade.symbol,
                type_str,
                detail,
                qty_str,
                f'${trade.entry_price:.2f}',
                f'${curr:.2f}',
                f'${trade.stop_price:.2f}',
                f'${trade.target_price:.2f}',
                f'[{pnl_color}]{upnl:+.2f}[/{pnl_color}]',
                opened_str,
            )

        layout['open'].update(Panel(open_table))

        # ── Closed trades today ───────────────────────────────────────────────
        closed_table = Table(
            title='[bold]Today\'s Closed Trades[/bold]',
            expand=True,
            show_lines=True,
        )
        closed_table.add_column('Ticker',   style='bold cyan', no_wrap=True)
        closed_table.add_column('Type',     justify='center')
        closed_table.add_column('Detail',   justify='center')
        closed_table.add_column('Qty',      justify='right')
        closed_table.add_column('Entry',    justify='right')
        closed_table.add_column('Exit',     justify='right')
        closed_table.add_column('P&L',      justify='right')
        closed_table.add_column('Result',   justify='center')
        closed_table.add_column('Opened',   justify='center', style='dim')
        closed_table.add_column('Closed',   justify='center', style='dim')

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        closed = [
            t for t in self.orders.get_closed_trades()
            if t.entry_time and t.entry_time.astimezone(ET) >= today_start
        ]
        # Most recent first
        closed.sort(key=lambda t: t.exit_time or t.entry_time, reverse=True)

        for trade in closed:
            pnl = trade.realized_pnl or 0.0
            pnl_color = 'green' if pnl >= 0 else 'red'
            result_str = '[green]WIN[/green]' if pnl > 0 else '[red]LOSS[/red]'

            if trade.asset_type == 'option':
                type_str = '[magenta]OPTION[/magenta]'
                opt_label = trade.option_type.upper() if trade.option_type else '?'
                opt_color = 'green' if opt_label == 'CALL' else 'red'
                strike_str = f'${trade.strike:.0f}' if trade.strike else '?'
                detail = f'[{opt_color}]{opt_label}[/{opt_color}] {strike_str}'
                qty_str = f'{trade.contracts}c' if trade.contracts else f'{trade.shares//100}c'
            else:
                type_str = '[blue]STOCK[/blue]'
                detail = trade.direction.upper()
                qty_str = f'{trade.shares}sh'

            entry_et = trade.entry_time.astimezone(ET).strftime('%m/%d %H:%M') if trade.entry_time else '—'
            exit_et  = trade.exit_time.astimezone(ET).strftime('%m/%d %H:%M')  if trade.exit_time  else '—'

            closed_table.add_row(
                trade.symbol,
                type_str,
                detail,
                qty_str,
                f'${trade.entry_price:.2f}',
                f'${trade.exit_price:.2f}' if trade.exit_price else '—',
                f'[{pnl_color}]{pnl:+.2f}[/{pnl_color}]',
                result_str,
                entry_et,
                exit_et,
            )

        layout['closed'].update(Panel(closed_table))
        return layout

    # ── HTML data writer ──────────────────────────────────────────────────────

    def _write_html_data(self):
        """Write dashboard_data.js so dashboard.html can read live state."""
        import json
        from pathlib import Path

        now = et_now()
        stats = self.tracker.today_stats()

        def trade_to_dict(t, current_price=None):
            cp = current_price or t.entry_price
            mult = 1 if t.direction in ('long', 'buy') else -1
            upnl = (cp - t.entry_price) * t.shares * mult
            return {
                'trade_id':      t.trade_id,
                'symbol':        t.symbol,
                'asset_type':    t.asset_type,
                'option_type':   t.option_type,
                'strike':        t.strike,
                'contracts':     t.contracts,
                'original_contracts': getattr(t, 'original_contracts', None),
                'direction':     t.direction,
                'shares':        t.shares,
                'entry_price':   t.entry_price,
                'current_price': cp,
                'exit_price':    t.exit_price,
                'stop_price':    t.stop_price,
                'target_price':  t.target_price,
                'unrealized_pnl': round(upnl, 2),
                'realized_partial_pnl': round(getattr(t, 'realized_partial_pnl', 0), 2),
                'partial_exits': getattr(t, 'partial_exits', []),
                'tier_hits':     getattr(t, 'tier_hits', []),
                'realized_pnl':  t.realized_pnl,
                'outcome':       t.status,
                'entry_time':    t.entry_time.isoformat() if t.entry_time else None,
                'exit_time':     t.exit_time.isoformat()  if t.exit_time  else None,
            }

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        open_trades = self.orders.get_open_trades()
        closed = sorted(
            [t for t in self.orders.get_closed_trades()
             if t.entry_time and t.entry_time.astimezone(ET) >= today_start],
            key=lambda t: t.exit_time or t.entry_time,
            reverse=True,
        )

        all_time_stats = self.tracker.stats()

        # News / earnings context for dashboard
        news_articles = self.news.all_recent(limit=25)
        for art in news_articles:
            art['sentiment'] = round(score_text(
                f"{art.get('headline','')} {art.get('summary','')}"
            ), 3)

        earnings_upcoming = self.earnings.upcoming(self.watchlist, within_days=14)

        data = {
            'generated_at':    now.isoformat(),
            'phase':           self._phase,
            'options_only':    config.OPTIONS_ONLY_MODE,
            'equity':          self.risk.account_size,
            'daily_pnl':       round(self.risk.daily_pnl, 2),
            'win_rate':        round(stats.win_rate, 4),
            'wins':            stats.wins,
            'losses':          stats.losses,
            'profit_factor':   round(stats.profit_factor, 2),
            'all_time_pnl':    round(all_time_stats.total_pnl, 2),
            'open_positions':  [trade_to_dict(t, self._live_prices.get(t.symbol)) for t in open_trades],
            'closed_trades':   [trade_to_dict(t) for t in closed],
            'historical_trades': self.tracker.all_trades_for_chart(since_days=365),
            'news':            news_articles,
            'earnings':        earnings_upcoming,
            'sentiment':       self._sentiment_cache,
            'macro_news':      self.macro_feeds.all(limit=20),
            'macro_risk':      self.macro_risk.to_dict(),
            'priority_tape':   self.priority_tape.snapshot_all(),
            'priority_volume': self.volume.snapshot_all(),
            'adaptive_stats':  self.adaptive.all_stats(),
            'session':         session_mod.to_dict(et_now()),
            'session_stats':   self.adaptive.session_stats(),
            'recent_rejections': list(self._rejections)[-40:][::-1],   # newest first
            'thesis_watches':  self.exit_manager.active_thesis_watches(),
            'intel':           self.intel.dashboard_data(),
        }

        js = f'window.DASHBOARD_DATA = {json.dumps(data, indent=2)};\n'
        Path(__file__).parent.joinpath('dashboard_data.js').write_text(js)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        log.info('=== AlgoBot starting ===')

        # Dashboard HTTP server (force-close button, live polling)
        try:
            await self.dashboard_server.start()
        except Exception as e:
            log.warning(f'Dashboard server failed to start: {e}')

        self._load_bars()

        # One-shot refresh of all news/intel feeds at startup so we have
        # current state regardless of when in the session we launched.
        try:
            self.earnings.refresh(self.watchlist)
        except Exception as e:
            log.warning(f'Startup earnings refresh failed: {e}')
        try:
            self._refresh_news_if_due()
        except Exception as e:
            log.warning(f'Startup news refresh failed: {e}')
        try:
            self._refresh_macro_if_due()
        except Exception as e:
            log.warning(f'Startup macro refresh failed: {e}')
        try:
            self._refresh_intel_if_due()
        except Exception as e:
            log.warning(f'Startup intel refresh failed: {e}')

        stream_task = asyncio.create_task(self._stream_trades())
        await asyncio.sleep(3)

        trade_task = asyncio.create_task(self._trading_loop())
        has_tty = sys.stdout.isatty()

        try:
            if has_tty:
                with Live(self._build_dashboard(), refresh_per_second=1, console=console) as live:
                    while self._running:
                        live.update(self._build_dashboard())
                        self._write_html_data()
                        await asyncio.sleep(1)
            else:
                while self._running:
                    self._write_html_data()
                    await asyncio.sleep(2)    # snappy refresh for the dashboard server
        except asyncio.CancelledError:
            pass
        finally:
            trade_task.cancel()
            stream_task.cancel()
            self.orders.close_all()
            self._write_html_data()
            self._print_session_report()

    def _print_session_report(self):
        stats = self.tracker.today_stats()
        console.rule('[bold]SESSION REPORT[/bold]')
        console.print(f'Trades:        {stats.total_trades}')
        console.print(f'Win Rate:      {stats.win_rate*100:.1f}%  ({stats.wins}W / {stats.losses}L)')
        console.print(f'Total P&L:     ${stats.total_pnl:.2f}')
        console.print(f'Avg Win:       ${stats.avg_win:.2f}')
        console.print(f'Avg Loss:      ${stats.avg_loss:.2f}')
        console.print(f'Profit Factor: {stats.profit_factor:.2f}')
        console.print(f'Best Strategy: {stats.best_strategy}')
        console.print(f'Avg Score:     {stats.avg_score:.0f}/100')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AlgoBot paper trader')
    parser.add_argument('--watchlist', nargs='+', default=config.WATCHLIST)
    parser.add_argument('--no-options', action='store_true')
    parser.add_argument('--paper', default='true', choices=['true', 'false'])
    args = parser.parse_args()

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        console.print('[red]ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env[/red]')
        sys.exit(1)

    os.environ['PAPER_TRADING'] = args.paper
    bot = TradingBot(
        watchlist=args.watchlist,
        use_options=not args.no_options,
    )
    asyncio.run(bot.run())


if __name__ == '__main__':
    main()
