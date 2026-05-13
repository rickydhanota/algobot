"""Singleton wrappers around Alpaca's trading and data clients."""
import config
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.live.option import OptionDataStream
    OPTIONS_AVAILABLE = True
except ImportError:
    OPTIONS_AVAILABLE = False


class AlpacaClients:
    _trading: TradingClient = None
    _stock_hist: StockHistoricalDataClient = None
    _option_hist: OptionHistoricalDataClient = None
    _stock_stream: StockDataStream = None

    @classmethod
    def trading(cls) -> TradingClient:
        if cls._trading is None:
            cls._trading = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=config.PAPER_TRADING,
            )
        return cls._trading

    @classmethod
    def stock_hist(cls) -> StockHistoricalDataClient:
        if cls._stock_hist is None:
            cls._stock_hist = StockHistoricalDataClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
            )
        return cls._stock_hist

    @classmethod
    def option_hist(cls):
        if not OPTIONS_AVAILABLE:
            return None
        if cls._option_hist is None:
            cls._option_hist = OptionHistoricalDataClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
            )
        return cls._option_hist

    @classmethod
    def stock_stream(cls) -> StockDataStream:
        if cls._stock_stream is None:
            cls._stock_stream = StockDataStream(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
            )
        return cls._stock_stream

    @classmethod
    def get_account(cls):
        return cls.trading().get_account()
