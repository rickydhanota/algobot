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
from learning.adaptive import AdaptiveLearner
from strategy.risk_manager import RiskManager
from strategy.stock_strategy import StockStrategy
from strategy.options_strategy import OptionsStrategy
from execution.order_manager import OrderManager, ActiveTrade
from performance.tracker import PerformanceTracker
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


class TradingBot:
    def __init__(self, watchlist: List[str], use_options: bool = True):
        # Ensure SPY and TSLA are always in the watchlist (SPX is proxied from SPY)
        wl = list(dict.fromkeys(list(watchlist) + ['SPY', 'TSLA']))
        self.watchlist = wl
        self.use_options = use_options

        self.market_data = MarketDataManager()
        self.tape = TapeReader()
        self.priority_tape = PriorityTapeReader()
        self.adaptive = AdaptiveLearner()
        self.analyzer = TechnicalAnalyzer()
        self.risk = RiskManager()
        self.stock_strategy = StockStrategy()
        self.options_strategy = OptionsStrategy()
        self.orders = OrderManager(self.risk)
        self.tracker = PerformanceTracker()

        # News & calendars
        self.news = AlpacaNewsFeed()
        self.earnings = EarningsCalendar()
        self.ipos = IPOCalendar()
        self.macro_feeds = MacroFeedAggregator()
        self.macro_risk: MacroRisk = MacroRisk()
        self.intel = IntelAggregator()

        self._traded_today: Set[str] = set()
        self._running = False
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
            # Enhanced reader (SPY trades also populate SPX proxy buffer)
            if sym in PRIORITY_SYMBOLS or sym == 'SPY':
                self.priority_tape.record_trade(sym, price, size)

        stream.subscribe_trades(on_trade, *self.watchlist)
        await stream.run()

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

    def _apply_macro_risk(self):
        """Translate MacroRisk into risk-manager state."""
        self.risk.macro_risk_multiplier = self.macro_risk.risk_multiplier
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

    def _evaluate_symbol(self, sym: str) -> Optional[dict]:
        """Return a trade setup dict if symbol has a high-quality signal."""
        if sym in self._traded_today:
            return None

        # News & earnings filter — gate BEFORE expensive analysis
        allowed, score_adj, reason = self._news_filter(sym)
        if not allowed:
            log.debug(f'[{sym}] Skipped: {reason}')
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
        if sym in PRIORITY_SYMBOLS:
            priority_sig = self.priority_tape.analyze(sym)
            if priority_sig and priority_sig.is_strong:
                priority_boost = 10
            elif priority_sig and priority_sig.is_moderate:
                priority_boost = 5

        # ── Stock setup
        stock_setup = self.stock_strategy.evaluate(tech, tape_signal)
        if stock_setup:
            intel_boost = min(
                config.INTEL_SCORE_BOOST_MAX,
                self.intel.score_boost(sym, stock_setup.direction),
            )
            stock_setup.score = min(100, stock_setup.score + score_adj + priority_boost + intel_boost)
            if score_adj > 0:
                stock_setup.notes.append(f'news+{score_adj}')
            if priority_boost > 0:
                stock_setup.notes.append(f'priority+{priority_boost}')
            if intel_boost > 0:
                stock_setup.notes.append(f'smartmoney+{intel_boost}')

            # Adaptive threshold per (symbol, strategy)
            adj = self.adaptive.threshold_adjustment(sym, stock_setup.strategy)
            min_required = config.MIN_SIGNAL_SCORE + adj
            if stock_setup.score < min_required:
                return None

            # Log the signal that fired (outcome recorded later when trade closes)
            self.adaptive.record_signal(
                signal_id=f'{sym}-{stock_setup.strategy}-{int(stock_setup.timestamp.timestamp())}',
                symbol=sym,
                strategy=stock_setup.strategy,
                score=stock_setup.score,
                imbalance=tape_signal.imbalance if tape_signal else 0.0,
                rvol=stock_setup.rvol,
                confluence_score=priority_sig.confluence_score if priority_sig else 0,
                direction=stock_setup.direction,
            )
            return {'type': 'stock', 'setup': stock_setup, 'tech': tech, 'tape': tape_signal}

        # ── Options setup (priority names benefit most here)
        if self.use_options:
            chain = self.market_data.get_option_chain(sym)
            if chain:
                opt_setup = self.options_strategy.evaluate(sym, chain, tech, tape_signal)
                if opt_setup and opt_setup.is_valid:
                    opt_setup.score = min(100, opt_setup.score + priority_boost)
                    return {'type': 'option', 'setup': opt_setup, 'tech': tech, 'tape': tape_signal}

        return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _trading_loop(self):
        while self._running:
            now = et_now()
            phase = market_phase(now)
            self._phase = phase

            if phase == 'closed':
                self._running = False
                break

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
                self.orders.check_exits(self._live_prices)
                await asyncio.sleep(30)
                continue

            # ── Active trading
            allowed, reason = self.risk.is_trading_allowed()
            if not allowed:
                log.warning(f'Trading halted: {reason}')
                await asyncio.sleep(60)
                continue

            self._update_account()
            self._refresh_news_if_due()
            self._refresh_macro_if_due()
            self.orders.check_exits(self._live_prices)

            # Scan watchlist for setups
            for sym in self.watchlist:
                result = self._evaluate_symbol(sym)
                if result is None:
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

            await asyncio.sleep(30)

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
                'symbol':        t.symbol,
                'asset_type':    t.asset_type,
                'option_type':   t.option_type,
                'strike':        t.strike,
                'contracts':     t.contracts,
                'direction':     t.direction,
                'shares':        t.shares,
                'entry_price':   t.entry_price,
                'current_price': cp,
                'exit_price':    t.exit_price,
                'stop_price':    t.stop_price,
                'target_price':  t.target_price,
                'unrealized_pnl': round(upnl, 2),
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
            'adaptive_stats':  self.adaptive.all_stats(),
            'intel':           self.intel.dashboard_data(),
        }

        js = f'window.DASHBOARD_DATA = {json.dumps(data, indent=2)};\n'
        Path(__file__).parent.joinpath('dashboard_data.js').write_text(js)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        log.info('=== AlgoBot starting ===')
        self._load_bars()

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
                    await asyncio.sleep(30)
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
