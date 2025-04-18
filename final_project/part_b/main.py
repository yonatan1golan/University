from data.stock_fetcher import StockDataFetcher
from config import CONFIG
import datetime as dt
import pandas as pd
import numpy as np

def add_ttm_eps_values(specific_date: dt.date) -> float:
    """ adds ttm_eps values to the data, using the last 4 quarters of eps data """
    relevant_eps = CONFIG.TESLA_EPS.where(CONFIG.TESLA_EPS['publish_date'].dt.date <= specific_date).dropna()
    if len(relevant_eps) >= 4:
        return round(relevant_eps.tail(4)['quarterly_eps'].sum(), 2)
    return None

def calculate_pe(data: pd.Series) -> float:
    """ calculates pe values to the data, using the ttm_eps values """
    return data['close'] / data['ttm_eps'] if data['ttm_eps'] != 0 else None

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """ calculates the RSI (Relative Strength Index) for the given close prices """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_return(closing_prices: pd.Series) -> pd.Series:
    """ calculates the daily return for the given closing prices """
    return np.log(closing_prices / closing_prices.shift(1))

def create_market_returns(stocks: list[str]) -> pd.DataFrame:
    """ creates a dataframe of the market returns for the given stocks """
    all_returns_list = []
    for stock in stocks:
        stock = StockDataFetcher(stock)
        stock_data = stock.get_data()
        closing_prices = stock_data['close']
        returns = calculate_return(closing_prices).dropna()

        temp = pd.DataFrame(columns=['date', f'return_{stock.name}'])
        temp['date'] = stock_data['date']
        temp[f'return_{stock.name}'] = returns
        all_returns_list.append(temp.set_index('date'))
    all_returns = pd.concat(all_returns_list, axis=1)
    return all_returns.dropna().reset_index()

def calculate_csad(data: pd.DataFrame) -> pd.Series:
    """ calculates the CSAD for each day of Tesla's data against the sample stocks"""
    market_returns = create_market_returns(CONFIG.SAMPLE_STOCKS_FOR_CSAD)
    merged_data = data[['date', 'return']].merge(market_returns, on='date', how='left')
    merged_data = merged_data.dropna()
    merged_data.rename(columns={'return': 'return_TSLA'}, inplace=True)
    
    # TODO: calculate the CSAD
    csad = 0
    return csad
    
def generate_data(stock_data: StockDataFetcher):
    """ populates the data with additional features for future analysis and regression """
    # get the stock's data
    data = stock_data.get_data()

    # independent variables
    data['ttm_eps'] = data['date'].apply(add_ttm_eps_values)
    data['pe'] = data.apply(lambda x: calculate_pe(x), axis=1)
    data['rsi'] = calculate_rsi(data['close'])
    data['return'] = calculate_return(data['close'])
    data['volatility'] = data['return'].rolling(10).std()
    data['beta'] = stock_data.Ticker.info['beta']
    data['daily_trend'] = data['close'] / data['open']
    data['weekday'] = data['date'].apply(lambda x: x.weekday())
    data['quarter'] = data['date'].apply(lambda x: (x.month - 1) // 3 + 1)
    for i in range(1,6):
        data[f'delta_volume_{i}_days_back'] = data['volume'] - data['volume'].shift(i)

    # dependent variables
    # volumn,
    data['csad'] = calculate_csad(data)
    # close / close nasdaq same day
    # close / close snp500 same day
    # stock volumn / market volumn

    return data.dropna().reset_index(drop=True)

if __name__ == "__main__":
    tesla_stock = StockDataFetcher(CONFIG.TESLA_TICKER)
    # nasdaq_stock = StockDataFetcher(CONFIG.NASDAQ_TICKER)

    processed_tesla_data = generate_data(tesla_stock)
    print(processed_tesla_data)
    # processed_tesla_data.to_csv("tesla_processed_data.csv", index=False)
