from matplotlib.backends.backend_pdf import PdfPages
=from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from itertools import product
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

import json

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

def visualize_tree(tree, feature_names, pdf_path="decision_tree.pdf"):
    """
    Input:
      - tree: The trained decision tree model
      - feature_names: List of feature names used in the training data
    Outputs:
      - A PDF file with a simplified and more readable visualization of the decision tree.
    """
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(30, 20))  # Increase figure size for larger bricks
        plot_tree(
            tree,
            feature_names=feature_names,  # Use feature names for splits
            class_names=['0', '1'],       # Use '0' and '1' for classes
            filled=True,                  # Add colors
            rounded=True,                 # Rounded nodes
            fontsize=7                   # Set font size for better readability
        )
        plt.title("Decision Tree Visualization", fontsize=20)  # Larger title
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()
    print(f"Decision tree saved to {pdf_path}")

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

def grid_search_tree(df: pd.DataFrame, max_depth_range: range, criteria: list, splitters: list, ratio: float):
    """
    Input:
        - df: Pre-processed DataFrame
        - max_depth_range: Range of max depths to test
        - criteria: List of criteria to test (e.g., ['gini', 'entropy', 'log_loss'])
        - splitters: List of splitters to test (e.g., ['best', 'random'])
        - ratio: Train-test split ratio
    Returns:
        - A DataFrame with grid search results
    """
    results = []
    train, test = split_train_test(df, ratio=ratio)

    for criterion in criteria:
        for splitter in splitters:
            for depth in max_depth_range:
                tree = build_tree(train, depth=depth, criterion=criterion, splitter=splitter)
                accuracy = tree.score(test.drop(columns=['OUTCOME']), test['OUTCOME'])
                results.append({
                    'Criterion': criterion,
                    'Splitter': splitter,
                    'Max Depth': depth,
                    'Accuracy': accuracy
                })
    
    return pd.DataFrame(results)

def plot_heatmaps(results: pd.DataFrame):
    """
    Input: Results DataFrame from grid search
    Displays: Heatmaps for Accuracy with Splitter as separate heatmaps
    """
    splitters = results['Splitter'].unique()  # Get unique splitters
    for splitter in splitters:
        splitter_data = results[results['Splitter'] == splitter]
        heatmap_data = splitter_data.pivot(index='Criterion', columns='Max Depth', values='Accuracy')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='YlGnBu', cbar=True)
        plt.title(f"Heatmap of Grid Search Results for Splitter = {splitter}")
        plt.xlabel('Max Depth')
        plt.ylabel('Criterion')
        plt.show()

def find_best_combination(results: pd.DataFrame):
    """
    Input: Results DataFrame from grid search
    Returns: The best combination of max_depth, criterion, and splitter with the highest accuracy
    """
    best_row = results.loc[results['Accuracy'].idxmax()]
    return best_row

def plot_3d_results(results: pd.DataFrame):
    """
    Input: Results DataFrame from grid search
    Displays: An interactive 3D scatter plot of Accuracy with Criterion, Splitter, and Max Depth
    """
    fig = px.scatter_3d(
        results,
        x='Max Depth',      # X-axis
        y='Criterion',      # Y-axis
        z='Accuracy',       # Z-axis
        color='Splitter',   # Color by splitter
        symbol='Splitter',  # Different symbols for splitters
        size='Accuracy',    # Size based on accuracy
        title="3D Visualization of Grid Search Results",
        labels={
            'Max Depth': 'Max Depth',
            'Criterion': 'Criterion',
            'Accuracy': 'Accuracy',
            'Splitter': 'Splitter'
        }
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='Criterion',
        zaxis_title='Accuracy'
    ))
    fig.show()

def get_decision_tree(df: pd.DataFrame):
    """
    Input: a dataframe that has been pre-processed
    Returns: the decision tree with the highest accuracy

    Logic:
    1. Split the data into train and test sets
    2. Build a decision tree using the train set
    3. Evaluate the accuracy of the decision tree on the test set
    4. Repeat the process for 10 iterations
    5. Find the best ratio of the train and test set in terms of tree accuracy
    6. Find the best combination of the decision tree hyperparameters
    7. Return the best decision tree
    """
    # ratio = get_maximum_accuracy_ratio(df)
    # max_depth_range = range(1, 20)
    # criteria = ['gini', 'entropy', 'log_loss']
    # splitters = ['best', 'random']
    # results = grid_search_tree(df, max_depth_range, criteria, splitters, ratio)
    # best_combination = find_best_combination(results)
    # print(f"The best combination is:\n{best_combination}")
    # plot_heatmaps(results)
    # plot_3d_results(results)
    return build_tree(df, depth = 8, criterion = 'entropy', splitter = 'best')

def decision_tree_section(train: pd.DataFrame, test: pd.DataFrame):
    """
    Gathers the decision tree section of the code
    """
    decision_tree = get_decision_tree(train)
    print(f"The accuracy of the decision tree is: {decision_tree.score(test.drop(columns=['OUTCOME']), test['OUTCOME'])}")
    feature_importance = decision_tree.feature_importances_
    feature_names = train.drop(columns=['OUTCOME']).columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    visualize_tree(decision_tree, train.drop(columns=['OUTCOME']).columns)

def nn_model_with_modified_params(x_train_scaled: pd.DataFrame, y_train: pd.Series, params: dict) -> MLPClassifier:
    """
    Input: train set to train NN model
    Returns: NN model that has been trained on the train set with modified params
    """
    model = MLPClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train_scaled, y_train)
    # print()
    # print(f"The best params for the neural network model are:")
    # print(json.dumps(grid_search.best_params_, indent=4))
    # print()
    return MLPClassifier(**grid_search.best_params_)

def check_nn_model_accuracy(nn_model: MLPClassifier, x_train_scaled: pd.DataFrame, y_train: pd.Series, x_test_scaled: pd.DataFrame, y_test: pd.Series):
    """
    Input: NN model and train set
    Returns: the accuracy of the NN model on the train set
    """
    nn_model.fit(x_train_scaled, y_train)
    print(f"The accuracy of the neural network on the train set is: {nn_model.score(x_train_scaled, y_train)}")
    print(f"The accuracy of the neural network on the test set is: {nn_model.score(x_test_scaled, y_test)}")

def build_nn_model(x_train_scaled: pd.DataFrame, y_train: pd.Series, params: dict = None):
    """
    Input: train set to train NN model, params that can be used to modify the model
    Returns: NN model with the given params
    """
    if not params:
        return MLPClassifier()
    return nn_model_with_modified_params(x_train_scaled, y_train, params)

def plot_nn_heatmap(param_grid: dict, x_train_scaled: pd.DataFrame, y_train: pd.Series, x_test_scaled: pd.DataFrame, y_test: pd.Series):
    """
    Input: params that can be used to modify the model
    Returns: heatmap of the NN model accuracy
    """
    param_combinations = list(product(
        param_grid['hidden_layer_sizes'], 
        param_grid['activation'], 
        param_grid['alpha'], 
        param_grid['max_iter']
    ))

    results = []
    for params in param_combinations:
        hidden_layer_sizes, activation, alpha, max_iter = params
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(x_train_scaled, y_train)
        accuracy = model.score(x_test_scaled, y_test)
        results.append({
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'alpha': alpha,
            'max_iter': max_iter,
            'accuracy': accuracy
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    heatmap_plots = [
            {
                "pivot_index": "activation",
                "pivot_columns": "hidden_layer_sizes",
                "title": "Activation vs Hidden Layer Sizes (Fixed Alpha)",
                "constant_param": "alpha"
            },
            {
                "pivot_index": "alpha",
                "pivot_columns": "hidden_layer_sizes",
                "title": "Alpha vs Hidden Layer Sizes (Fixed Activation)",
                "constant_param": "activation"
            },
            {
                "pivot_index": "alpha",
                "pivot_columns": "activation",
                "title": "Alpha vs Activation (Fixed Hidden Layer Sizes)",
                "constant_param": "hidden_layer_sizes"
            }
        ]

    for plot_config in heatmap_plots:
        constant_param = plot_config["constant_param"]
        unique_constants = results_df[constant_param].unique()

        for constant_value in unique_constants:
            filtered_df = results_df[results_df[constant_param] == constant_value]
            heatmap_data = filtered_df.pivot_table(
                index=plot_config["pivot_index"],
                columns=plot_config["pivot_columns"],
                values="accuracy",
                aggfunc="mean"
            )

            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f", cbar=True)
            plt.title(f"{plot_config['title']} ({constant_param}={constant_value})")
            plt.xlabel(plot_config["pivot_columns"])
            plt.ylabel(plot_config["pivot_index"])
            plt.show()

def neural_network_section(train: pd.DataFrame, test: pd.DataFrame):
    """
    Gathers the neural network section of the code
    """
    # scaling the data
    minmax_scaler = MinMaxScaler()
    x_train = train.drop(columns=['OUTCOME'])
    y_train = train['OUTCOME']
    x_test = test.drop(columns=['OUTCOME'])
    y_test = test['OUTCOME']

    x_train_scaled = minmax_scaler.fit_transform(x_train)
    x_test_scaled = minmax_scaler.transform(x_test)

    params = {
        'hidden_layer_sizes': [(50, 50), (50, 50, 50),(100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant'],
        'max_iter': [100, 200]
    }

    base_nn_model = build_nn_model(x_train_scaled, y_train)
    modified_nn_model = build_nn_model(x_train_scaled, y_train, params)
    print("\nBase NN Model:")
    check_nn_model_accuracy(base_nn_model, x_train_scaled, y_train, x_test_scaled, y_test)
    print("\nModified NN Model:")
    check_nn_model_accuracy(modified_nn_model, x_train_scaled, y_train, x_test_scaled, y_test)
    plot_nn_heatmap(params, x_train_scaled, y_train, x_test_scaled, y_test)

    
if __name__ == "__main__":
    raw_file = pd.read_csv(file_path).drop(columns = ['ID'])
    data = pre_process(raw_file)
    train, test = split_train_test(data, ratio = 0.8)

    # decision_tree_section(train, test)
    neural_network_section(train, test)