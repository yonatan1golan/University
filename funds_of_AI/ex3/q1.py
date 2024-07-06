# imports
from sklearn.preprocessing import KBinsDiscretizer
from multiprocessing import Pool, cpu_count
from anytree import Node, RenderTree  # visualize
from scipy.stats import chi2 
import pandas as pd
import numpy as np
import logging

# defaults
logging.basicConfig(level=logging.INFO)
file_path = "flightdelay.csv"
target_column = 'DEP_DEL15'
learning_set = []
testing_set = []
attributes = []
data = None
tree = None

# input
input_data = [1, 7, '0800-0859', 2, 1, 25, 143, 'Southwest Airlines Co.', 13056, 107363, 5873, 1903352, 13382999, 6.178236301460919e-05,
              9.889412309998219e-05, 8, 'McCarran International', 36.08, -115.152, 'NONE', 0, 0, 0, 65, 2.91]

# classes
class Tree:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, node):
        self.children.append(node)
    
    def to_anytree(self, parent=None):
        root = Node(self.label, parent=parent)
        for child in self.children:
            child.to_anytree(root)
        return root
# end classes

def categorize_day_of_week(day): # categorize a day in the week as a weekday or a weekend
    return 'Weekday' if day in [1, 2, 3, 4, 5] else 'Weekend'   

def categorize_month(month): # categorize a month as a winter, spring, summer or fall
    month_season_map = {
    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
    7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
}
    return month_season_map[month]

def categorize_time_block(time_block): # categorize a time block as a morning, afternoon, evening or night
    start, end = time_block.split('-')
    start_hour = int(start[:2])
    start_minute = int(start[2:])
    end_hour = int(end[:2])
    end_minute = int(end[2:])
    
    start_total_minutes = start_hour * 60 + start_minute
    end_total_minutes = end_hour * 60 + end_minute

    if 0 <= start_total_minutes and end_total_minutes <= 360: # 00:00 - 05:59
        return 'Early Morning'
    elif 360 <= start_total_minutes and end_total_minutes <= 720: # 06:00 - 11:59
        return 'Morning'
    elif 720 <= start_total_minutes and end_total_minutes <= 1080: # 12:00 - 17:59
        return 'Afternoon'
    elif 1080 <= start_total_minutes and end_total_minutes <= 1320: # 18:00 - 21:59
        return 'Evening'
    else: # 22:00 - 23:59 or mixed start and end times
        return 'Night'

def categorize_attributes_values(df): # categorize attributes values
    logging.info('Categorizing...')
    continuous_columns = ['DISTANCE_GROUP', 'SEGMENT_NUMBER', 'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS',
                      'AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH', 'AIRLINE_AIRPORT_FLIGHTS_MONTH',
                      'AVG_MONTHLY_PASS_AIRPORT', 'AVG_MONTHLY_PASS_AIRLINE', 'FLT_ATTENDANTS_PER_PASS',
                      'GROUND_SERV_PER_PASS', 'PLANE_AGE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND']
    categorial_columns = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']

    df['MONTH'] = df['MONTH'].apply(categorize_month)
    df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].apply(categorize_day_of_week)
    df['DEP_TIME_BLK'] = df['DEP_TIME_BLK'].apply(categorize_time_block)

    # discretize continuous attributes
    for column in continuous_columns:
        est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
        df[column] = est.fit_transform(df[[column]])

    discrete_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}
    for column in continuous_columns:
        df[column] = df[column].map(discrete_map) # mapping the values as low, medium, high, and very high

    for column in categorial_columns:
        top_categories = df[column].value_counts().nlargest(10).index
        df[column] = df[column].apply(lambda x: x if x in top_categories else 'Other')

    return df

def get_data(): # get the data
    logging.info('Getting the data...')
    global data
    if data is None:
        data = pd.read_csv(file_path)
        data = categorize_attributes_values(data)
    return data

def divide_datasets(data, target_column):  # will divide the dataset into a learning set and a test set
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

def get_attributes(data):  # returns the tree attributes <=> data columns \ target column
    return data.columns.drop(target_column)

def pre_process(ratio):  # reads the data and divides it into a learning set and a testing set and returns the attributes (columns \ target)
    global learning_set, testing_set, attributes
    assert 0 <= ratio <= 1, 'ratio must be between 0 and 1'
    data = get_data()
    learning_set = data.sample(frac=ratio, random_state=42)
    testing_set = data.drop(learning_set.index)
    attributes = get_attributes(learning_set)
    logging.info(f'The dataset was divided into {len(learning_set)}/{len(testing_set)} learning/testing sets.')
    return learning_set, testing_set, attributes

def plurality_value(exs):  # returns most common value in the target column (is the flight delayed or not)
    target_values = exs[target_column]
    return target_values.mode()[0]

def same_classification(exs):  # will return a tuple of (T, classification) or (F, None)
    target_values = exs[target_column]
    if len(target_values.unique()) == 1:
        return True, target_values.iloc[0]
    else:
        return False, None

def entropy(y): # calculates the entropy of an attribute value column
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y_left, y_right): # calculates the information gain of an attribute value column
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    return entropy(y) - (p_left * entropy(y_left) + p_right * entropy(y_right))

def chi2_pruning(y, y_left, y_right, alpha=0.05): # chi squared pruning
    observed_freq = [len(y_left), len(y_right)]
    expected_freq = [len(y) / 2, len(y) / 2]

    chi2_st = sum((observed - expected) ** 2 / expected for observed, expected in zip(observed_freq, expected_freq))
    chi2_cr = chi2_critical(1, alpha)
    return chi2_st < chi2_cr

def chi2_critical(degreefreedom, alpha=0.05): # chi squared table
    assert 0 <= alpha <= 1, 'alpha value must be in the range (0, 1)'
    return chi2.ppf(1 - alpha, degreefreedom)

def importance(attribute, examples): # calculating the importance of an attribute, how much more information it gives me
    y = examples[target_column]
    values = examples[attribute].unique()
    gain = 0
    for value in values:
        subset = examples[examples[attribute] == value]
        y_subset = subset[target_column]
        gain += information_gain(y, y[examples[attribute] == value], y_subset)
    return gain

def get_values(examples, attribute):  # returns the unique values under a specific attribute in a df
    return examples[attribute].unique()

def decision_tree_learning(examples, attributes, parent_examples): # builds the decision tree base on decision_tree_learning algo.
    if examples.empty:
        return plurality_value(parent_examples)
    elif same_classification(examples)[0]:
        return same_classification(examples)[1]
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        A = max(attributes, key=lambda a: importance(a, examples))
        tree = Tree(label=A)
        for v in get_values(examples, A):
            exs = examples[examples[A] == v]
            subtree = decision_tree_learning(exs, attributes.drop(A), examples)
            node = Tree(label=f'{A} = {v}')
            if isinstance(subtree, Tree):
                node.add_child(subtree)
            else:
                node.add_child(Tree(label=subtree))
            tree.add_child(node)
        return tree

def build_tree(ratio):  # builds the decision tree with a ratio for testing and learning division
    """
    i need to return how much does the tree is accurate
    """
    global learning_set, testing_set, attributes
    learning_set, testing_set, attributes = pre_process(ratio)
    tree = decision_tree_learning(learning_set, attributes, None)
    return tree

def parallel_k_fold_validation(args): # parallel k fold validation
    data, k = args
    return k_fold_validation(data, k)

def k_fold_validation(data, k): # k fold validation
    global attributes
    k_fold_sets = np.array_split(data.sample(frac=1, random_state=42), k)
    errors = []

    for i in range(k):
        logging.info(f"Processing fold {i+1}/{k}")
        validation_set = k_fold_sets[i]
        training_set = pd.concat([fold for j, fold in enumerate(k_fold_sets) if j != i])
        tree = decision_tree_learning(training_set, attributes, None)

        correct = 0
        for _, row in validation_set.iterrows():
            prediction = predict(tree, row[:-1].to_dict())
            if prediction == row[target_column]:
                correct += 1
        errors.append(1 - correct / len(validation_set))
    return np.mean(errors)

def tree_error(k): # tree error calculation
    data = get_data()
    num_cores = cpu_count()
    pool = Pool(num_cores)
    k_splits = [(data, k // num_cores) for _ in range(num_cores)]
    errors = pool.map(parallel_k_fold_validation, k_splits)
    pool.close()
    pool.join()
    return np.mean(errors)

def categorize_row(row): # categorize_row
    global attributes
    raw_df = pd.DataFrame(columns = attributes, data = row)
    raw_df = categorize_attributes_values(raw_df)
    return raw_df[0]

def predict(tree, row): # predicts if the flight is late
    row = categorize_row(row) # categorize_row
    def traverse_tree(node, row): # traverses the tree to check if the row input is late
        if not node.children:
            return node.label
        for child in node.children:
            attr, value = child.label.split(' = ')
            if row[attr] == value:
                return traverse_tree(child, row)
            
    return traverse_tree(tree.to_anytree(), row)

def is_late(row_input): # checks prediction for the input
    global tree
    logging.info("Building tree for prediction...")
    full_data = get_data()
    row_dict = dict(zip(full_data.columns[:-1], row_input))
    logging.info("Predicting...")
    return predict(tree, row_dict)

if __name__ == '__main__':
    tree = build_tree(ratio=0.1)
    root = tree.to_anytree()
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    
    logging.info("Predicting if the flight is late...")
    result = is_late(input_data)
    print(f"Prediction result: {'Late' if result else 'Not Late'}")