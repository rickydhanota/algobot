"""
SEC EDGAR 13F filings — institutional positions for famous investors.

Filings are reported 45 days after each quarter end (NCT 13F-HR), so the
data is at minimum 45-90 days stale. Used as a bias layer, not a signal.

Workflow:
  1. For each tracked fund (CIK), pull recent filings JSON from
     data.sec.gov/submissions/CIK{padded}.json
  2. Find latest 13F-HR accession number
  3. Fetch filing's index.json to locate the information-table XML
  4. Parse XML, extract holdings, map CUSIP/name → ticker
"""
from __future__ import annotations
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# SEC requires User-Agent with company name + contact email
USER_AGENT = 'AlgoBot/1.0 (algobot-paper@example.com)'
TIMEOUT_SECONDS = 30

# ── Famous investor CIKs ─────────────────────────────────────────────────────
TRACKED_FUNDS = {
    'Berkshire Hathaway (Buffett)':       '0001067983',
    'Scion Asset Mgmt (Burry)':           '0001649339',
    'Bridgewater Associates (Dalio)':     '0001350694',
    'Pershing Square (Ackman)':           '0001336528',
    'Greenlight Capital (Einhorn)':       '0001079114',
    'Appaloosa LP (Tepper)':              '0001656456',
    'Tiger Global Management':            '0001167483',
    'Third Point LLC (Loeb)':             '0001040273',
    'Renaissance Technologies':           '0001037389',
    'Citadel Advisors':                   '0001423053',
    'Duquesne Family Office (Druck.)':    '0001536411',
    'Soros Fund Management':              '0001029160',
}

# ── CUSIP → ticker mapping for our watchlist ────────────────────────────────
CUSIP_TO_TICKER = {
    '037833100': 'AAPL',
    '88160R101': 'TSLA',
    '67066G104': 'NVDA',
    '594918104': 'MSFT',
    '023135106': 'AMZN',
    '30303M102': 'META',
    '02079K305': 'GOOGL',
    '02079K107': 'GOOG',
    '007903107': 'AMD',
    '78462F103': 'SPY',
    '46090E103': 'QQQ',
    '464287655': 'IWM',
    '19260Q107': 'COIN',
    '69608A108': 'PLTR',
    '83406F102': 'SOFI',
    '594972408': 'MSTR',
}

# ── Name → ticker (fallback when CUSIP unknown) ──────────────────────────────
NAME_PATTERNS = {
    'apple inc':              'AAPL',
    'tesla inc':              'TSLA',
    'nvidia corp':            'NVDA',
    'microsoft corp':         'MSFT',
    'amazon com':             'AMZN',
    'meta platforms':         'META',
    'alphabet inc':           'GOOGL',
    'advanced micro devices': 'AMD',
    'spdr s&p 500':           'SPY',
    'invesco qqq':            'QQQ',
    'coinbase global':        'COIN',
    'palantir technologies':  'PLTR',
    'sofi technologies':      'SOFI',
    'microstrategy':          'MSTR',
}


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read()


def _resolve_ticker(cusip: str, name: str) -> Optional[str]:
    if cusip:
        cusip_clean = cusip.replace(' ', '').upper()
        if cusip_clean in CUSIP_TO_TICKER:
            return CUSIP_TO_TICKER[cusip_clean]
    if name:
        n = name.lower().strip()
        for pat, ticker in NAME_PATTERNS.items():
            if pat in n:
                return ticker
    return None


class Form13FFetcher:
    """Fetches latest 13F-HR holdings for tracked funds."""

    def __init__(self):
        # fund_name → list of holdings
        self._holdings: Dict[str, List[dict]] = {}
        self._last_refresh: Optional[datetime] = None

    def refresh(self, max_funds: Optional[int] = None):
        funds = list(TRACKED_FUNDS.items())
        if max_funds:
            funds = funds[:max_funds]

        self._holdings.clear()
        for fund_name, cik in funds:
            try:
                holdings = self._fetch_latest_13f(cik)
                if holdings:
                    self._holdings[fund_name] = holdings
                    log.info(f'13F [{fund_name}]: {len(holdings)} positions')
            except Exception as e:
                log.warning(f'13F [{fund_name}] failed: {e}')

        self._last_refresh = datetime.now()

    def _fetch_latest_13f(self, cik: str) -> List[dict]:
        # 1. Get recent submissions
        padded = cik.zfill(10)
        submissions_url = f'https://data.sec.gov/submissions/CIK{padded}.json'
        try:
            body = _http_get(submissions_url)
            subs = json.loads(body)
        except Exception:
            return []

        recent = subs.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accs = recent.get('accessionNumber', [])
        if not forms:
            return []

        # 2. Find latest 13F-HR
        latest_acc = None
        for form, acc in zip(forms, accs):
            if form in ('13F-HR', '13F-HR/A'):
                latest_acc = acc
                break
        if not latest_acc:
            return []

        # 3. Get filing index.json to find the info-table XML
        acc_clean = latest_acc.replace('-', '')
        cik_short = str(int(cik))
        index_url = f'https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/index.json'
        try:
            body = _http_get(index_url)
            idx = json.loads(body)
        except Exception:
            return []

        items = idx.get('directory', {}).get('item', [])
        xml_filename = None
        for it in items:
            name = it.get('name', '').lower()
            # Common info-table filenames
            if 'infotable' in name and name.endswith('.xml'):
                xml_filename = it['name']
                break
            if name.endswith('.xml') and 'primary' not in name:
                xml_filename = it['name']
                break
        if not xml_filename:
            return []

        # 4. Fetch and parse XML
        xml_url = f'https://www.sec.gov/Archives/edgar/data/{cik_short}/{acc_clean}/{xml_filename}'
        try:
            xml_body = _http_get(xml_url)
            return self._parse_info_table(xml_body)
        except Exception:
            return []

    @staticmethod
    def _parse_info_table(xml_body: bytes) -> List[dict]:
        try:
            root = ET.fromstring(xml_body)
        except ET.ParseError:
            return []

        # 13F XML uses a namespace
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        holdings = []
        for entry in root.findall(f'.//{ns}infoTable'):
            name   = entry.findtext(f'{ns}nameOfIssuer', '').strip()
            cusip  = entry.findtext(f'{ns}cusip', '').strip()
            value  = entry.findtext(f'{ns}value', '0').strip()
            shares = entry.findtext(f'.//{ns}sshPrnamt', '0').strip()
            put_call = entry.findtext(f'{ns}putCall', '').strip()

            try:
                # Post-2022 SEC amendment: values are in actual dollars.
                # Pre-2022 filings reported in thousands. Detect: any value
                # under ~10M is almost certainly thousands (no major fund
                # has positions under $10M as their largest holdings).
                raw_value = int(value)
                value_usd = raw_value if raw_value > 10_000_000 else raw_value * 1000
                shares_n = int(shares)
            except ValueError:
                continue

            ticker = _resolve_ticker(cusip, name)
            holdings.append({
                'name':      name,
                'cusip':     cusip,
                'ticker':    ticker,
                'shares':    shares_n,
                'value_usd': value_usd,
                'put_call':  put_call or 'long',
            })

        # Sort by value descending
        holdings.sort(key=lambda h: -h['value_usd'])
        return holdings

    # ── Accessors ─────────────────────────────────────────────────────────────

    def all_holdings(self) -> Dict[str, List[dict]]:
        return self._holdings

    def funds_holding(self, ticker: str) -> List[dict]:
        """Return [(fund_name, position_dict), ...] for funds holding `ticker`."""
        out = []
        for fund, positions in self._holdings.items():
            for p in positions:
                if p.get('ticker') == ticker.upper():
                    out.append({
                        'fund':     fund,
                        'shares':   p['shares'],
                        'value_usd': p['value_usd'],
                        'put_call': p['put_call'],
                    })
        out.sort(key=lambda x: -x['value_usd'])
        return out

    def watchlist_summary(self, symbols: List[str]) -> List[dict]:
        """For each watchlist symbol: which funds hold it, total value."""
        out = []
        for sym in symbols:
            funds = self.funds_holding(sym)
            if not funds:
                continue
            total_value = sum(f['value_usd'] for f in funds)
            out.append({
                'symbol':       sym,
                'fund_count':   len(funds),
                'total_value':  total_value,
                'top_holders':  funds[:5],
            })
        out.sort(key=lambda x: -x['total_value'])
        return out
