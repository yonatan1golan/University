# imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from anytree import Node, RenderTree  # visualize
from scipy.stats import chi2 
import pandas as pd
import numpy as np


# defaults
file_path = "flightdelay.csv"
target_column = 'DEP_DEL15'
learning_set = []
testing_set = []
attributes = []
original_attributes = []
data = None
tree = None

# input
input_data = [
    1, 7, '2300-2359', 5, 4, 26, 143, 'Southwest Airlines Co.', 13056, 107363, 5873, 1903352, 13382999, 6.18E-05, 9.89E-05, 17,
    'McCarran International', 36.08, -115.152, 'Albuquerque International Sunport', 0, 0, 0, 65, 2.91
]

# classes
class Tree:
    def __init__(self, label, header = False, leaf = False, value = 0):
        self.label = label
        self.children = []
        self.header = header
        self.leaf = leaf
        if leaf:
            self.value = value

    def add_child(self, node):
        self.children.append(node)
        if node.label in(0, 1):
            self.leaf = True
    
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
    continuous_columns = ['DISTANCE_GROUP', 'SEGMENT_NUMBER', 'CONCURRENT_FLIGHTS',
                          'AVG_MONTHLY_PASS_AIRPORT', 'FLT_ATTENDANTS_PER_PASS',
                          'GROUND_SERV_PER_PASS', 'PLANE_AGE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND']
    categorical_columns = ['CARRIER_NAME', 'DEPARTING_AIRPORT']
    unuseful_columns = ['NUMBER_OF_SEATS', 'LATITUDE', 'LONGITUDE', 'AIRLINE_FLIGHTS_MONTH', 'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AVG_MONTHLY_PASS_AIRLINE',
                        'PREVIOUS_AIRPORT', 'AIRPORT_FLIGHTS_MONTH']
    if df['MONTH'].dtype != 'object':
        df['MONTH'] = df['MONTH'].apply(categorize_month)
    if df['DAY_OF_WEEK'].dtype != 'object':
        df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].apply(categorize_day_of_week)
    if df['DEP_TIME_BLK'].dtype != 'object':
        df['DEP_TIME_BLK'] = df['DEP_TIME_BLK'].apply(categorize_time_block)
    df = df.drop(columns=unuseful_columns)
    
    for column in continuous_columns:
        if df[column].nunique() > 1:
            est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
            transformed = est.fit_transform(df[[column]])
            df[column] = transformed.T[0]
        else:
            df[column] = 0  # If the column has only one unique value, assign 0 to all entries

    discrete_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    for column in continuous_columns:
        df[column] = df[column].map(discrete_map)

    for column in categorical_columns:
        top_categories = df[column].value_counts().nlargest(5).index
        df[column] = df[column].apply(lambda x: x if x in top_categories else 'Other')

    return df

def get_data(): # get the data
    global data, original_attributes
    if data is None:
        data = pd.read_csv(file_path)
        data = data.drop_duplicates().reset_index(drop=True)
        original_attributes = get_attributes(data)
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

def chi_square_prune(tree, examples, target_attribute, alpha=0.05):
    if not isinstance(tree, Tree) or not tree.children:
        return tree  # Return if it's a leaf node

    split_attribute = tree.label.split(' = ')[0]
    subtrees = {}
    for child in tree.children:
        if isinstance(child.label, str) and ' = ' in child.label:
            attr_val = child.label.split(' = ')[1]
            subtrees[attr_val] = child

    # Prune each subtree recursively
    for attr_val, subtree in subtrees.items():
        subset_df = examples[examples[split_attribute] == attr_val]
        subtrees[attr_val] = chi_square_prune(subtree, subset_df, target_attribute, alpha)

    # After pruning subtrees, decide if we should prune this node
    if len(subtrees) < 2:
        return examples[target_attribute].mode()[0]  # Replace node with the most common target value

    attr_vals = list(subtrees.keys())
    y = examples[target_attribute]
    y_left = examples[examples[split_attribute] == attr_vals[0]][target_attribute]
    y_right = examples[examples[split_attribute] == attr_vals[1]][target_attribute]

    if chi2_pruning(y, y_left, y_right, alpha):
        return examples[target_attribute].mode()[0]  # Replace node with the most common target value

    return tree

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
        tree = Tree(label=A, header = True)
        for v in get_values(examples, A):
            exs = examples[examples[A] == v]
            new_attributes = [attr for attr in attributes if attr != A]
            subtree = decision_tree_learning(exs, new_attributes, examples)
            node = Tree(label=f'{A} = {v}')
            if isinstance(subtree, Tree):
                node.add_child(subtree)
            elif subtree in (0,1):
                node.add_child(Tree(label=subtree, leaf = True, value = subtree))
            else:
                node.add_child(Tree(label=subtree))
            tree.add_child(node)

    # chi2 prunning
    pruned_tree = chi_square_prune(tree, examples, target_column)
    return pruned_tree

def build_tree(ratio):  # builds the decision tree with a ratio for testing and learning division
    global learning_set, testing_set, attributes, tree
    print("Building tree..")
    learning_set, testing_set, attributes = pre_process(ratio)
    tree = decision_tree_learning(learning_set, attributes, None)
    print_tree(tree)
    return tree

def k_fold_validation(data, k): # performs k-fold validation
    global attributes, target_column
    print(f"Starting {k}-fold validations..")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    errors = []

    for fold, (train_index, test_index) in enumerate(skf.split(data, data[target_column]), 1):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        print(f"Building tree for fold {fold}..")
        tree = decision_tree_learning(train_data, attributes, None)
        correct = 0
        print(f"Checking the accuracy for the {fold} tree for {k}-validation")
        for _, row in test_data.iterrows():
            prediction = predict(tree, row)
            if prediction == row[target_column]:
                correct += 1
        fold_error = 1 - (correct / len(test_data))
        errors.append(fold_error)  
    return np.mean(errors) if errors else float('inf')

def tree_error(k): # performs 2-fold validation and presents the average error
    global testing_set
    avg_error = k_fold_validation(testing_set, k)
    avg_accuracy = (1 - avg_error) * 100
    print(f"The average error is: {avg_error*100:.2f}%")
    print(f"The quality of the tree is: {avg_accuracy:.2f}%")

def categorize_row(row): # categorizes the row according to the data categorization
    global attributes
    raw_df = pd.DataFrame(data=[row], columns=original_attributes)
    raw_df = categorize_attributes_values(raw_df)
    return raw_df.iloc[0]

def predict(tree, row): # predicts if the flight is late based on the tree
    if tree.header:
        return predict(tree.children[0], row)
    if tree.leaf:
        return tree.children[0].value
    for child in tree.children:
        if child.leaf:
            return child.value
        elif child.header:
            return predict(child, row)
        else:
            attr, value = child.label.split(' = ')
            row = categorize_row(row)
            if row[attr] == value:
                return predict(tree.children[0], row)

def is_late(row_input): # checks prediction for the input
    new_tree = building_tree(1)
    print("Categorizing and testing input row..")
    prediction = predict(new_tree, row_input)
    print(f"Prediction result: {1 if prediction else 0}")

def print_tree(tree): # prints the tree in a readable form
    print("Tree structure:")
    root = tree.to_anytree()
    for pre, _, node in RenderTree(root):
        print("%s%s" % (pre, node.name))

def building_tree(prediction, ratio=0.6):
    if prediction:
        return build_tree(1) # return a tree based on the full data
    else:
        build_tree(ratio) # build and prints the tree
        tree_error(2) # performs k-fold validation and presents the average error
        
if __name__ == '__main__':
    tree = building_tree(prediction = 0, ratio = 0.6)
    #result = is_late(input_data)