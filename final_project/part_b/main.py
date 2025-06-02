from data.stock_fetcher import StockDataFetcher

from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


from config import CONFIG
import datetime as dt
import pandas as pd
import numpy as np
import os

class Regression:
    def __init__(self, data: pd.DataFrame, period: dict, x: list, y: str, cov_type="nonrobust"):
        self.data = data
        self.period = period
        self.x = x
        self.y = y
        self.cov_type = cov_type
        self.model = self._run()
        self.equation = self._get_general_equation()

    def print_regression_results(self):
        print(f"\nRegression results for period {self.period['start']} to {self.period['end']}:\n{self.model.summary()}\n")

    def _get_general_equation(self) -> str:
        params = list(self.model.params.keys())
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

    def _run(self, x_vars=None):
        start = pd.Timestamp(self.period['start'])
        end = pd.Timestamp(self.period['end'])
        data = self.data[(self.data['date'] >= start) & (self.data['date'] <= end)].copy().reset_index(drop=True)

        # Use provided x_vars if given, otherwise default to self.x
        predictors = x_vars if x_vars is not None else self.x
        formula = f"{self.y} ~ {' + '.join(predictors)}"

        model = smf.ols(formula=formula, data=data).fit(cov_type=self.cov_type)
        return model

    def run_anova(self):
        """
        Performs ANOVA and shows how much each variable contributes to explaining the variance.
        """
        anova_results = anova_lm(self.model, typ=2)

        ss_total = anova_results['sum_sq'].sum()
        anova_results['explained_pct'] = (anova_results['sum_sq'] / ss_total)
        # anova_results['explained_pct'] = anova_results['explained_pct'].map(lambda x: f"{x:.2f}%")

        print(f"\nANOVA results for period {self.period['start']} to {self.period['end']}:\n")
        print(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'explained_pct']])
        
        return anova_results.sort_values(by='explained_pct', ascending=False)
    
    def backward_selection(self, file_name):
        """
        Performs backward selection and logs output to a file.

        Returns:
            list: the list of variables remaining after selection
        """
        log_path = file_name + "_backward_selection_log.txt"
        remaining_vars = self.x.copy()
        with open(log_path, 'w') as log:
            while True:
                model = self._run(x_vars=remaining_vars)
                pvalues = model.pvalues.drop('Intercept', errors='ignore')

                if (pvalues <= 0.05).all():
                    log.write("\n✅ Final model with all significant variables (p <= 0.05):\n")
                    log.write(model.summary().as_text())
                    break

                worst_var_fullname = pvalues.idxmax()
                max_p = pvalues[worst_var_fullname]

                log.write(f"\n🗑 Removing '{worst_var_fullname}' with p-value {max_p:.4f}\n")
                remaining_vars.remove(worst_var_fullname)
                log.write(f"\n📉 Updated model with variables: {remaining_vars}\n")
                log.write(model.summary().as_text() + "\n\n")

        # Final update
        self.x = remaining_vars
        self.model = model
        self.equation = self._get_general_equation()
        print(f"\n✅ Backward selection complete. Full log written to '{log_path}'")

        return self.x 

    def vif(self) -> pd.DataFrame:
        """
        Calculates Variance Inflation Factor (VIF) for each feature.
        """
        start = pd.Timestamp(self.period['start'])
        end = pd.Timestamp(self.period['end'])
        data = self.data[(self.data['date'] >= start) & (self.data['date'] <= end)].copy().reset_index(drop=True)

        # Construct design matrix using Patsy
        formula = f"{self.y} ~ {' + '.join(self.x)}"
        y, X = dmatrices(formula, data, return_type='dataframe')

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        return vif_data

def plot_residuals_vs_fitted(model, title="Residuals vs Fitted"):
    fitted_vals = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(title)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()
    
def add_ttm_eps_values(specific_date: dt.date) -> float:
    relevant_eps = CONFIG.TESLA_EPS.where(CONFIG.TESLA_EPS['publish_date'] <= pd.Timestamp(specific_date)).dropna()
    if len(relevant_eps) >= 4:
        return round(relevant_eps.tail(4)['quarterly_eps'].sum(), 2)
    return None

def calculate_pe(data: pd.Series) -> float:
    return data['close'] / data['ttm_eps'] if data['ttm_eps'] != 0 else None

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_return(closing_prices: pd.Series) -> pd.Series:
    return np.log(closing_prices / closing_prices.shift(1))

def create_market_returns(stocks: list[str]) -> pd.DataFrame:
    all_returns_list = []
    for stock_name in stocks:
        relevant_file_path = f"final_project/part_b/data/{stock_name.lower()}_stock.csv"
        stock = pd.read_csv(relevant_file_path)
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')
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
    merged_data = data[['date', 'log_return']].merge(market_returns, on='date', how='left')
    merged_data = merged_data.dropna().reset_index(drop=True)
    merged_data.rename(columns={'log_return': 'return_TSLA'}, inplace=True)

    stocks_data = [col for col in merged_data.columns if col.startswith('return_') and col != 'return_TSLA']
    csad = sum(
        (merged_data['return_TSLA'] - merged_data[stock]).abs() for stock in stocks_data
    ) / len(stocks_data)
    csad.index = merged_data['date']
    return csad

def generate_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
    data = stock_data.where(stock_data['date'] >= pd.Timestamp(dt.date(2020, 1, 1))).dropna()

    data['ttm_eps'] = data['date'].apply(add_ttm_eps_values)
    data['pe'] = data.apply(lambda x: calculate_pe(x), axis=1)
    data['rsi'] = calculate_rsi(data['close']) # above 70 is overbought, below 30 is oversold
    data['log_return'] = calculate_return(data['close'])
    data['volatility'] = data['log_return'].rolling(10).std()
    data['beta'] = stock_data['beta']
    data['daily_trend'] = data['close'] / data['open']
    data['weekday'] = data['date'].dt.weekday 
    data['quarter'] = data['date'].dt.quarter.astype(float)
    for i in range(1, 11):
        data[f'delta_volume_{i}_days_back'] = data['volume'] - data['volume'].shift(i)

    csad = calculate_csad(data)
    csad.name = 'csad'
    data = data.merge(csad, left_on='date', right_index=True, how='left')
    return data.dropna().reset_index(drop=True)

def normalize_feature_to_z_score(data: pd.DataFrame, feature: str) -> pd.Series:
    data[feature] = data[feature].astype(float)
    return (data[feature] - data[feature].mean()) / data[feature].std()

def generate_interaction_feature(data: pd.DataFrame, features: list[str]) -> pd.Series:
    normalized = [normalize_feature_to_z_score(data, feature) for feature in features]
    interaction_df = pd.concat(normalized, axis=1)
    interaction_series = interaction_df.prod(axis=1)
    interaction_series.name = "_".join(features)
    return interaction_series

def bin_series_kbins(series: pd.Series, n_bins=3, labels=[1, 2, 3]) -> pd.Series:
    kbinner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    binned = kbinner.fit_transform(series.values.reshape(-1, 1)).astype(int).flatten()
    label_map = {i: label for i, label in enumerate(labels)}
    return pd.Series(binned, index=series.index).map(label_map)

def categorize(data: pd.DataFrame) -> pd.DataFrame:
    binned = data.copy()
    to_bin = binned.columns.difference(['weekday', 'quarter', 'daily_trend', 'beta', 'date', 'volume'])
    binned[to_bin] = binned[to_bin].apply(lambda x: bin_series_kbins(x), axis=0)
    binned['daily_trend'] = binned['daily_trend'].apply(lambda x: 1 if x > 1 else -1)
    return binned

def check_homoscedasticity(regression_obj):
    """
    Runs the Breusch-Pagan test for heteroscedasticity on a Regression object.

    Parameters:
        regression_obj (Regression): an instance of the Regression class

    Returns:
        dict: Breusch-Pagan test results including p-values and test statistics
    """
    residuals = regression_obj.model.resid
    exog = regression_obj.model.model.exog

    test = het_breuschpagan(residuals, exog)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    results = dict(zip(labels, test))

    print(f"\n📊 Breusch-Pagan Test for period {regression_obj.period['start']} to {regression_obj.period['end']}:")
    for label, value in results.items():
        print(f"{label}: {value:.4f}")
    
    if results['p-value'] < 0.05:
        print("🚨 Likely heteroscedasticity detected (p < 0.05)")
    else:
        print("✅ Likely homoscedasticity (constant variance) (p ≥ 0.05)")

    return results

def create_merged_delta_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates merged delta features to test if they improve regression performance.
    """
    data_with_merged = data.copy()
    
    # 1 - simple sum of all deltas
    delta_cols = [f'delta_volume_{i}_days_back' for i in range(1, 11)]
    data_with_merged['delta_volume_sum'] = data_with_merged[delta_cols].sum(axis=1)
    
    # 2- weighted sum (recent days have higher weight)
    weights = [1.0 / i for i in range(1, 11)]  # 1.0, 0.5, 0.33, 0.25, ...
    weighted_deltas = sum(data_with_merged[f'delta_volume_{i}_days_back'] * weights[i-1] 
                         for i in range(1, 11))
    data_with_merged['delta_volume_weighted'] = weighted_deltas
    
    # 3 - average of all deltas
    data_with_merged['delta_volume_avg'] = data_with_merged[delta_cols].mean(axis=1)
    
    # 4- short term vs Long-term (1-3 days vs 4-10 days)
    data_with_merged['delta_volume_short_term'] = data_with_merged[[f'delta_volume_{i}_days_back' for i in range(1, 4)]].sum(axis=1)
    data_with_merged['delta_volume_long_term'] = data_with_merged[[f'delta_volume_{i}_days_back' for i in range(4, 11)]].sum(axis=1)
    return data_with_merged

def compare_delta_approaches(data: pd.DataFrame, period: dict, base_x_columns: list, output_dir: str = 'final_project/part_b/delta_analysis') -> dict:
    """
    Compares regression performance with different delta approaches.
    
    Args:
        data: DataFrame with the data
        period: Period configuration
        base_x_columns: Base X columns for regression
        output_dir: Directory to save approach result files
    
    Returns:
        dict: Results comparing different approaches
    """
    
    results = {}    
    data_with_merged = create_merged_delta_features(data)    
    delta_individual_cols = [f'delta_volume_{i}_days_back' for i in range(1, 11)]
    
    os.makedirs(output_dir, exist_ok=True)
    approaches = {
        'individual_deltas': base_x_columns,
        'sum_delta': [col for col in base_x_columns if col not in delta_individual_cols] + ['delta_volume_sum'],
        'weighted_delta': [col for col in base_x_columns if col not in delta_individual_cols] + ['delta_volume_weighted'],
        'avg_delta': [col for col in base_x_columns if col not in delta_individual_cols] + ['delta_volume_avg'],
        'short_long_delta': [col for col in base_x_columns if col not in delta_individual_cols] + ['delta_volume_short_term', 'delta_volume_long_term'],
        'no_deltas': [col for col in base_x_columns if col not in delta_individual_cols]
    }
    
    print("🔍 Comparing different delta approaches...\n")
    for approach_name, x_vars in approaches.items():
        if approach_name in ['sum_delta', 'weighted_delta', 'avg_delta', 'short_long_delta']:
            regression = Regression(data_with_merged, period, x_vars, 'volume', 'HC3')
        else:
            regression = Regression(data, period, x_vars, 'volume', 'HC3')
        
        results[approach_name] = {
            'r_squared': regression.model.rsquared,
            'adj_r_squared': regression.model.rsquared_adj,
            'aic': regression.model.aic,
            'bic': regression.model.bic,
            'num_features': len(x_vars),
            'significant_vars': sum(regression.model.pvalues.drop('Intercept', errors='ignore') <= 0.05)
        }
        
        log_path = os.path.join(output_dir, f'{approach_name}_results.txt')
        with open(log_path, 'w') as approach_file:
            approach_file.write(f"DELTA APPROACH: {approach_name.upper().replace('_', ' ')}\n")
            approach_file.write("=" * 50 + "\n\n")
            approach_file.write(f"R²: {results[approach_name]['r_squared']:.4f}\n")
            approach_file.write(f"Adjusted R²: {results[approach_name]['adj_r_squared']:.4f}\n")
            approach_file.write(f"AIC: {results[approach_name]['aic']:.2f}\n")
            approach_file.write(f"BIC: {results[approach_name]['bic']:.2f}\n")
            approach_file.write(f"Number of Features: {results[approach_name]['num_features']}\n")
            approach_file.write(f"Significant Variables: {results[approach_name]['significant_vars']}\n\n")
            
            approach_file.write("FEATURES USED:\n")
            approach_file.write("-" * 20 + "\n")
            for i, var in enumerate(x_vars, 1):
                approach_file.write(f"{i:2d}. {var}\n")
            
            approach_file.write(f"\nMODEL SUMMARY:\n")
            approach_file.write("-" * 20 + "\n")
            approach_file.write(str(regression.model.summary()))
        print(f"Results saved to: {log_path}")
    return results

if __name__ == "__main__":
    tesla_stock = pd.read_csv("final_project/part_b/data/tesla_stock.csv")
    processed_tesla_data = generate_features(tesla_stock)
    x_columns = processed_tesla_data.columns.difference(['volume', 'date'])
    potentially_herding_variables = ['delta_volume_1_days_back', 'delta_volume_2_days_back', 'delta_volume_3_days_back',
                                     'delta_volume_4_days_back', 'delta_volume_5_days_back', 'delta_volume_6_days_back', 'delta_volume_7_days_back',
                                     'delta_volume_8_days_back', 'delta_volume_9_days_back', 'delta_volume_10_days_back', 'rsi', 'pe', 'csad']
    INTERACTIONS = {
        "pe_daily_trend_volatility": ["pe", "daily_trend", "volatility"],
        "weekday_return_rsi": ["weekday", "log_return", "rsi"]
    }

    pe_daily_trend_volatility = generate_interaction_feature(processed_tesla_data, INTERACTIONS["pe_daily_trend_volatility"])
    weekday_return_rsi = generate_interaction_feature(processed_tesla_data, INTERACTIONS["weekday_return_rsi"])
    x_columns_with_interactions = list(x_columns) + [pe_daily_trend_volatility.name, weekday_return_rsi.name]

    continuous_data = processed_tesla_data.copy()
    continuous_data[pe_daily_trend_volatility.name] = pe_daily_trend_volatility
    continuous_data[weekday_return_rsi.name] = weekday_return_rsi

    categorized_data = categorize(continuous_data)

    categorized_regression = Regression(categorized_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions, 'volume')
    continuous_regression = Regression(continuous_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions, 'volume')
    continous_regression_without_herding = Regression(continuous_data, CONFIG.INTERESTING_PERIOD, x_columns.difference(potentially_herding_variables), 'volume')
    # anova_table = categorized_regression.run_anova()
    # continuous_regression.backward_selection()

    # plot_residuals_vs_fitted(continuous_regression.model, "Categorized Regression Residuals vs Fitted")
    # check_homoscedasticity(continuous_regression)

    without_hc3_continuous = Regression(continuous_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions, 'volume')
    with_hc3_continuous = Regression(continuous_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions, 'volume', 'HC3')

    # delta_analysis_results = compare_delta_approaches(continuous_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions)
    # with_hc3_categorized = Regression(categorized_data, CONFIG.INTERESTING_PERIOD, x_columns_with_interactions, 'volume', 'HC3')
    # hc3_anova = with_hc3_categorized.run_anova()
    # without_hc3_continuous.backward_selection("hc3_continuous")