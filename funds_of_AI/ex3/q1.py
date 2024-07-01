# imports
from multiprocessing import Pool, cpu_count
from anytree import Node, RenderTree  # visualize
import pandas as pd
import numpy as np

# defaults
file_path = r"C:\Users\Yonat\OneDrive\Desktop\github\University\funds_of_AI\ex3\flightdelay.csv"
target_column = 'DEP_DEL15'
learning_set = []
testing_set = []
data = None

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

def get_data(): # get the data
    print('Getting the data....')
    global data
    if data is None:
        data = pd.read_csv(file_path)   
    return data

def divide_datasets(data, target_column):  # will divide the dataset into a learning set and a test set
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

def get_attributes(data):  # returns the tree attributes <=> data columns \ target column
    attributes = data.columns.drop(target_column)
    return attributes

def pre_process(ratio):  # reads the data and divides it into a learning set and a testing set and returns the attributes (columns \ target)
    global learning_set, testing_set
    assert 0 <= ratio <= 1, 'Ratio must be between 0 and 1'
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

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y_left, y_right):
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    return entropy(y) - (p_left * entropy(y_left) + p_right * entropy(y_right))

def chi2_pruning(y, y_left, y_right, alpha=0.05):
    observed_freq = [len(y_left), len(y_right)]
    expected_freq = [len(y) / 2, len(y) / 2]

    chi2_st = sum((observed - expected) ** 2 / expected for observed, expected in zip(observed_freq, expected_freq))
    chi2_cr = chi2_critical(1, alpha)
    return chi2_st < chi2_cr

def chi2_critical(df, alpha=0.05):  # chi squared table
    assert alpha in (0.01, 0.05, 0.1), 'alpha value must be in (0.01, 0.05, 0.1)'
    chi2_lookup = {
        0.10: {1: 2.71, 2: 4.61, 3: 6.25, 4: 7.78, 5: 9.24},
        0.05: {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07},
        0.01: {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09},
    }
    return chi2_lookup[alpha].get(df, None)

def importance(attribute, examples):
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

def build_tree(ratio):
    global learning_set, testing_set
    learning_set, testing_set, attributes = pre_process(ratio)
    tree = decision_tree_learning(learning_set, attributes, None)
    return tree

def parallel_k_fold_validation(args):
    data, k = args
    return k_fold_validation(data, k)

def k_fold_validation(data, k):
    k_fold_sets = np.array_split(data.sample(frac=1, random_state=42), k)
    errors = []

    for i in range(k):
        print(f"Processing fold {i+1}/{k}")
        validation_set = k_fold_sets[i]
        training_set = pd.concat(k_fold_sets[:i] + k_fold_sets[i+1:])
        attributes = get_attributes(training_set)
        tree = decision_tree_learning(training_set, attributes, None)

        correct = 0
        for _, row in validation_set.iterrows():
            prediction = predict(tree, row[:-1].to_dict())
            if prediction == row[target_column]:
                correct += 1

        errors.append(1 - correct / len(validation_set))

    return np.mean(errors)

def tree_error(k):
    data = get_data()
    num_cores = cpu_count()
    pool = Pool(num_cores)
    k_splits = [(data, k // num_cores) for _ in range(num_cores)]
    errors = pool.map(parallel_k_fold_validation, k_splits)
    pool.close()
    pool.join()
    return np.mean(errors)

def predict(tree, row):
    def traverse_tree(node, row):
        if not node.children:
            return node.label
        for child in node.children:
            attr, value = child.label.split(' = ')
            if row[attr] == value:
                return traverse_tree(child, row)
    return traverse_tree(tree.to_anytree(), row)

def is_late(row_input):
    print("Building tree for prediction...")
    full_data = get_data()
    _, _, attributes = pre_process(1.0)
    tree = decision_tree_learning(full_data, attributes, None)
    row_dict = dict(zip(full_data.columns[:-1], row_input))
    print("Predicting...")
    return predict(tree, row_dict)

if __name__ == '__main__':
    tree = build_tree(ratio=0.1)
    root = tree.to_anytree()
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    
    print("Predicting if the flight is late...")
    result = is_late(input_data)
    print(f"Prediction result: {'Late' if result else 'Not Late'}")