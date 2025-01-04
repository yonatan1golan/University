from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os

file_name = 'XY_train.csv'
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'XY_train.csv')

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a dataframe that contains missing values
    Returns: a dataframe that doesn't contain any missing values using knn imputation
    """

    columns_with_missing_values = df.columns[df.isnull().any()].tolist()
    imputer = KNNImputer(n_neighbors = 5)
    df[columns_with_missing_values] = imputer.fit_transform(df[columns_with_missing_values])
    return df

def fix_irrational_driving_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a dataframe that contains the driving age column with irrational values
    Returns: a dataframe with the driving age column fixed with minimum value of 16
    """
    df['LICENSE_AGE'] = df['AGE'] - df['DRIVING_EXPERIENCE']
    df['DRIVING_EXPERIENCE'] = np.where(df['LICENSE_AGE'] < 16, df['AGE'] - 16, df['DRIVING_EXPERIENCE'])
    df['LICENSE_AGE'] = df['AGE'] - df['DRIVING_EXPERIENCE'] # update the license age
    return df.drop(columns = ['LICENSE_AGE'])

def get_categorial_mapping(df: pd.DataFrame, categorical_columns: pd.DataFrame):
    """
    Input: categorial columns and the dataframe itself
    Returns: a dictionary that contains the mapping of the categorial values to integers
    """
    indexed_mappings = {}
    for col in categorical_columns.columns:
        unique_values = df[col].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        indexed_mappings[col] = mapping
    return indexed_mappings

def kbin_normal_distribution(column: pd.Series, n_bins: int = 3):
    """
    Input: a column that contains continous numerical values
    Returns: a column that contains the binned values of the input column
    """
    if column.nunique() == 1:
        return pd.Series([0] * len(column), index=column.index)

    column_reshaped = column.values.reshape(-1, 1)
    n_bins = min(n_bins, column.nunique())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Feature 0 is constant.*")
        warnings.filterwarnings("ignore", message=".*Bins whose width are too small.*")
        kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        binned_column = kbin.fit_transform(column_reshaped).astype(int)
    return pd.Series(binned_column.flatten(), index=column.index)

def kbin_exp_distribution(column: pd.Series, n_bins: int = 3):
    """
    Input: a column that contains continous numerical values but with exponential distribution
    Returns: a column that contains the binned values of the input column
    """
    if column.nunique() == 1:
        return pd.Series([0] * len(column), index=column.index)

    if column.min() <= 0:
        column = column + abs(column.min()) + 1

    log_column = np.log(column)
    column_reshaped = log_column.values.reshape(-1, 1)
    n_bins = min(n_bins, column.nunique())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Feature 0 is constant.*")
        warnings.filterwarnings("ignore", message=".*Bins whose width are too small.*")
        kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        binned_column = kbin.fit_transform(column_reshaped).astype(int)
    return pd.Series(binned_column.flatten(), index=column.index)

def categorize_age_attribute(age: int):
    """
    Input: age in the form of integer
    Returns: a string that represents the age category
    Logic:
    Age will be categorized as weight, so the very young and very old people will be considered as high risk
    """
    if age <= 24: return 3 
    elif age <= 65: return 1
    return 2

def categorize_driving_experience(years: int):
    """
    Input: years of driving experience in the form of integer
    Returns: a string that represents the driving experience category
    Logic:
    The less experience the driver has, the more risk he/she will be
    """
    if years <= 2: return 3
    elif years <= 8: return 2
    return 1

def categorize_risk_index(risk_index: float):
    """
    Input: risk index in the form of float
    Returns: a string that represents the risk index category
    """
    if risk_index <= 4: return 1
    elif risk_index <= 6: return 2
    return 3

def categorize_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a dataframe that contains the raw data but proccessed
    Returns: a dataframe that contains the categorial columns mapped and the numerical columns binned
    """
    categorical_columns = df.select_dtypes(exclude=['float64', 'int64'])
    categorial_mapping = get_categorial_mapping(df, categorical_columns)
    for col, mapping in categorial_mapping.items(): # assign the mapping to the categorial columns
        df[col] = df[col].map(mapping)

    df['CREDIT_SCORE'] = kbin_normal_distribution(df['CREDIT_SCORE'])
    df['ANNUAL_MILEAGE'] = kbin_normal_distribution(df['ANNUAL_MILEAGE'])
    df['AGE'] = df['AGE'].apply(categorize_age_attribute)
    df['DRIVING_EXPERIENCE'] = df['DRIVING_EXPERIENCE'].apply(categorize_driving_experience)
    df['DRIVER_RISK_INDEX'] = 0.4 * df['SPEEDING_VIOLATIONS'] + 0.4 * df['PAST_ACCIDENTS'] + 0.2 * df['DRIVING_EXPERIENCE'] * df['AGE']
    df['DRIVER_RISK_INDEX'] = df['DRIVER_RISK_INDEX'].apply(categorize_risk_index)
    df['SPEEDING_VIOLATIONS'] = kbin_exp_distribution(df['SPEEDING_VIOLATIONS'])
    df['PAST_ACCIDENTS'] = kbin_exp_distribution(df['PAST_ACCIDENTS'])
    return df

def pre_process(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a dataframe that contains the raw data
    Returns: a dataframe that has been pre-processed using the logic below
    """
    df = fill_missing_values(raw)
    df = fix_irrational_driving_age(df)
    df = categorize_attributes(df)
    return df

def split_train_test(df: pd.DataFrame, ratio: float):
    """
    Input: a dataframe that has been pre-processed and a ratio for the train and test set
    Returns: train and test sets
    """
    train = df.sample(frac = ratio)
    test = df.drop(train.index)
    return train, test
    
def build_tree(df: pd.DataFrame, depth: int = None, criterion: str = 'gini', splitter: str = 'best'):
    """
    Input: a dataframe that has been pre-processed and contains the train set
    Returns: a decision tree that has been trained on the given set
    """
    return DecisionTreeClassifier(max_depth=depth, criterion=criterion, splitter=splitter).fit(df.drop(columns = ['OUTCOME']), df['OUTCOME'])

def visualize_tree(tree, feature_names, class_names = ['0', '1']):
    """
    Input:
      - tree: The trained decision tree model
      - feature_names: List of feature names used in the training data
      - class_names: List of class names (e.g., ['0', '1'])
    Displays: A visualization of the decision tree
    """
    plt.figure(figsize=(20, 10))
    plot_tree(tree, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.show()

def get_maximum_accuracy_ratio(df: pd.DataFrame):
    """
    Input: a dataframe that has been pre-processed
    Returns: the maximum accuracy ratio of the decision tree
    """
    accuracy_map = []
    for i in range(10):
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            train, test = split_train_test(df, ratio=ratio)
            tree = build_tree(train)
            accuracy = tree.score(test.drop(columns=['OUTCOME']), test['OUTCOME'])
            accuracy_map.append({'Iteration': i, 'Ratio': ratio, 'Accuracy': accuracy, 'Tree': tree})
    accuracy_df = pd.DataFrame(accuracy_map)
    accuracy_summary = accuracy_df.groupby('Ratio').agg({'Accuracy': 'mean'}).reset_index()
    best_ratio_row = accuracy_summary[accuracy_summary['Accuracy'] == accuracy_summary['Accuracy'].max()]
    return best_ratio_row['Ratio'].values[0]

if __name__ == "__main__":
    raw_file = pd.read_csv(file_path).drop(columns = ['ID'])
    df = pre_process(raw_file)
    ratio = 0.8 # get_maximum_accuracy_ratio(df)
    train, test = split_train_test(df, ratio=ratio)
    tree = build_tree(train)
    
    # visualize_tree(maximum_tree, df.drop(columns=['OUTCOME']).columns)