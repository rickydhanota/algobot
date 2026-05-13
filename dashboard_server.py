"""
Tiny aiohttp server for the dashboard.

Endpoints:
  GET  /                → serves dashboard.html
  GET  /api/data        → live JSON state (poll target — ~1s refresh)
  POST /api/close-all   → emergency close-all of every open position
  POST /api/pause       → stop opening new positions (existing kept)
  POST /api/resume      → resume normal trading
  GET  /api/status      → bot status + key flags

The server runs in the same asyncio loop as the bot, so it has direct
access to bot state. It binds to localhost only — never expose externally.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from main import TradingBot

log = logging.getLogger(__name__)


def _cors(headers: dict = None) -> dict:
    base = {
        'Access-Control-Allow-Origin':  '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Cache-Control':                'no-store',
    }
    if headers:
        base.update(headers)
    return base


class DashboardServer:
    def __init__(self, bot: 'TradingBot', host: str = '127.0.0.1', port: int = 8765):
        self.bot = bot
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None
        self._root = Path(__file__).parent

    async def start(self):
        app = web.Application()
        app.router.add_get  ('/',                 self._serve_index)
        app.router.add_get  ('/api/data',         self._serve_data)
        app.router.add_get  ('/api/status',       self._serve_status)
        app.router.add_post ('/api/close-all',    self._close_all)
        app.router.add_post ('/api/pause',        self._pause)
        app.router.add_post ('/api/resume',       self._resume)
        app.router.add_route('OPTIONS', '/{tail:.*}', self._preflight)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        log.info(f'Dashboard server listening at http://{self.host}:{self.port}')

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    # ── Routes ────────────────────────────────────────────────────────────────

    async def _serve_index(self, request):
        path = self._root / 'dashboard.html'
        return web.FileResponse(path, headers=_cors())

    async def _serve_data(self, request):
        path = self._root / 'dashboard_data.js'
        if not path.exists():
            return web.json_response({'error': 'no data yet'}, status=404, headers=_cors())
        content = path.read_text()
        json_str = content.replace('window.DASHBOARD_DATA = ', '').rstrip(';\n ')
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return web.json_response({'error': 'data corrupted'}, status=500, headers=_cors())
        return web.json_response(data, headers=_cors())

    async def _serve_status(self, request):
        return web.json_response({
            'running':      self.bot._running,
            'phase':        self.bot._phase,
            'paused':       getattr(self.bot, '_paused', False),
            'force_close':  getattr(self.bot, '_force_close_requested', False),
            'open_count':   len(self.bot.orders.get_open_trades()),
        }, headers=_cors())

    async def _close_all(self, request):
        self.bot._force_close_requested = True
        open_count = len(self.bot.orders.get_open_trades())
        log.warning(f'🛑 FORCE-CLOSE requested via dashboard ({open_count} open positions)')
        return web.json_response(
            {'status': 'close-all queued', 'open_count': open_count},
            headers=_cors(),
        )

    async def _pause(self, request):
        self.bot._paused = True
        log.info('Trading PAUSED via dashboard')
        return web.json_response({'status': 'paused'}, headers=_cors())

    async def _resume(self, request):
        self.bot._paused = False
        log.info('Trading RESUMED via dashboard')
        return web.json_response({'status': 'resumed'}, headers=_cors())

    async def _preflight(self, request):
        return web.Response(headers=_cors())
