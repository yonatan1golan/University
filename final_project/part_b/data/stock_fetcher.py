import yfinance as yf
import pandas as pd

class StockDataFetcher:
    def __init__(self, ticker_name):
        self.Ticker = yf.Ticker(ticker_name)
        self.data = None
        self.name = ticker_name

    def _normalize_data(self, data):
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        return data[['Date','Close', 'Volume', 'Open']].rename(columns=lambda x: x.lower())

    def get_data(self):
        self.data = self.Ticker.history(period="max")
        return self._normalize_data(self.data)