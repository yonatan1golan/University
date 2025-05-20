from data.stock_fetcher import StockDataFetcher
import statsmodels.api as sm
from config import CONFIG
import datetime as dt
import pandas as pd
import numpy as np

def add_ttm_eps_values(specific_date: dt.date) -> float:
    """ adds ttm_eps values to the data, using the last 4 quarters of eps data """
    relevant_eps = CONFIG.TESLA_EPS.where(CONFIG.TESLA_EPS['publish_date'] <= pd.Timestamp(specific_date)).dropna()
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
    for stock_name in stocks:
        relevant_file_path = f"final_project/part_b/data/{stock_name.lower()}_stock.csv"
        # stock = StockDataFetcher(stock)
        stock = pd.read_csv(relevant_file_path)
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')  # NaT if parsing fails
        closing_prices = stock['Price']
        returns = calculate_return(closing_prices).dropna()

        temp = pd.DataFrame(columns=['date', f'return_{stock_name}'])
        temp['date'] = stock['Date']
        temp[f'return_{stock_name}'] = returns
        all_returns_list.append(temp.set_index('date'))
    all_returns = pd.concat(all_returns_list, axis=1)
    return all_returns.dropna().reset_index()

def calculate_csad(data: pd.DataFrame) -> pd.Series:
    market_returns = create_market_returns(CONFIG.SAMPLE_STOCKS_FOR_CSAD)
    merged_data = data[['date', 'return']].merge(market_returns, on='date', how='left')
    merged_data = merged_data.dropna().reset_index(drop=True)
    merged_data.rename(columns={'return': 'return_TSLA'}, inplace=True)

    stocks_data = [col for col in merged_data.columns if col.startswith('return_') and col != 'return_TSLA']
    csad = sum(
        (merged_data['return_TSLA'] - merged_data[stock]).abs() for stock in stocks_data
    ) / len(stocks_data)
    csad.index = merged_data['date']
    return csad

def generate_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """ populates the data with additional features for future analysis and regression """
    # get the stock's data starting from 2020-01-01
    stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')  # NaT if parsing fails
    data = stock_data.where(stock_data['date'] >= pd.Timestamp(dt.date(2020, 1, 1))).dropna()

    # independent variables
    data['ttm_eps'] = data['date'].apply(add_ttm_eps_values)
    data['pe'] = data.apply(lambda x: calculate_pe(x), axis=1)
    data['rsi'] = calculate_rsi(data['close'])
    data['return'] = calculate_return(data['close'])
    data['volatility'] = data['return'].rolling(10).std()
    data['beta'] = stock_data['beta']    
    data['daily_trend'] = data['close'] / data['open']
    data['weekday'] = data['date'].dt.weekday 
    data['quarter'] = data['date'].dt.quarter 
    for i in range(1,11):
        data[f'delta_volume_{i}_days_back'] = data['volume'] - data['volume'].shift(i)
    
    csad = calculate_csad(data)
    csad.name = 'csad'
    data = data.merge(csad, left_on='date', right_index=True, how='left')
    return data.dropna().reset_index(drop=True)

def normalize_feature_to_z_score(data: pd.DataFrame, feature: str) -> pd.Series:
    """ normalizes the given feature to z-scores """
    data[feature] = data[feature].astype(float)
    return (data[feature] - data[feature].mean()) / data[feature].std()

def generate_interaction_feature(data: pd.DataFrame, features: list[str]) -> pd.Series:
    """ generates interaction features for the given features, and normalizes them to z-scores """
    normalized = [normalize_feature_to_z_score(data, feature) for feature in features]
    interaction_df = pd.concat(normalized, axis=1)
    interaction_series = interaction_df.prod(axis=1)
    feature_name = "*".join(features)
    interaction_series.name = feature_name
    return interaction_series

class Regression:
    def __init__(self, data: pd.DataFrame, period: dict, x: list, y: str):
        self.data = data
        self.period = period
        self.x = x
        self.y = y
        self.model = self._run()
        self.equation = self._get_general_equation()

    def print_regression_results(self):
        """ prints the regression results """
        print(f"\nRegression results for period {self.period['start']} to {self.period['end']}:\n{self.model.summary()}\n")

    def _get_general_equation(self) -> str:
        """Returns the general symbolic regression equation (e.g., y = b0 + b1·x1 + b2·x2 + ...)"""
        params = list(self.model.params.keys())  # get variable names
        response_var = self.model.model.endog_names

        lines = [f"{response_var} ="]
        coef_idx = 0

        for name in params:
            if name == 'Intercept':
                term = f"    b0"
            else:
                coef_idx += 1
                term = f"    b{coef_idx}·{name}"

            lines.append(term)

        return "\n".join(lines)
    
    def _run(self) -> sm.OLS:
        """Runs the regression for the given period and handles non-numeric columns correctly."""
        start = pd.Timestamp(self.period['start'])
        end = pd.Timestamp(self.period['end'])
        data = self.data[(self.data['date'] >= start) & (self.data['date'] <= end)].reset_index(drop=True)

        x_columns = [
            col for col in self.x
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
        ]

        X = data[x_columns].astype(float)
        X = sm.add_constant(X)  # add intercept

        y = data[self.y].astype(float)
        model = sm.OLS(y, X).fit()
        return model

if __name__ == "__main__":
    tesla_stock = None
    # tesla_stock = StockDataFetcher(CONFIG.TESLA_TICKER)
    tesla_stock = pd.read_csv("final_project/part_b/data/tesla_stock.csv")
    processed_tesla_data = generate_features(tesla_stock)
    x_columns = processed_tesla_data.columns.difference(['volume'])
    
    INTERACTIONS = {
        "pe_daily_trend_volatility": ["pe", "daily_trend", "volatility"],
        "weekday_return_rsi": ["weekday", "return", "rsi"]
    }
    pe_daily_trend_volatility = generate_interaction_feature(processed_tesla_data, INTERACTIONS["pe_daily_trend_volatility"])
    weekday_return_rsi = generate_interaction_feature(processed_tesla_data, INTERACTIONS["weekday_return_rsi"])
    x_columns_with_interactions = list(x_columns) + [pe_daily_trend_volatility.name] + [weekday_return_rsi.name]

    processed_tesla_data_with_interactions = processed_tesla_data.copy()
    processed_tesla_data_with_interactions[pe_daily_trend_volatility.name] = pe_daily_trend_volatility
    processed_tesla_data_with_interactions[weekday_return_rsi.name] = weekday_return_rsi

    regression_model = Regression(
        data=processed_tesla_data_with_interactions,
        period=CONFIG.INTERESTING_PERIOD,
        x=x_columns_with_interactions,
        y='return'
    )
    
    regression_model.print_regression_results()