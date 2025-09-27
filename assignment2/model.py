''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol
import pandas as pd
from collections import Counter


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        self.majority_label = None


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.

        Sets self.majority_label to the most common label in y, returns nothing.
        '''
        if len(x) == 0:
            raise ValueError("There is no training data")
        if len(y) == 0:
            raise ValueError("There are no labels")
        if len(y) != len(x.index):
            raise ValueError("There should be the same number of training examples are there are labels (right now there is not)")
        self.majority_label = Counter(y).most_common()[0][0]
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.

        Returns a list of the majority label repeated for each row in x (so pass in the test set to this and get your predictions).
        '''
        if self.majority_label is None:
            raise ValueError("The model hasn't been trained yet")
        majorities = [self.majority_label]*len(x.index)
        return majorities
        

class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.root = None

    
    def entropy(self, y: list) -> float:
        if len(y) == 0:
            return 0.0
        else:
            num_elements = len(y)
            counts_dict = Counter(y)
            probs = []
            for label in counts_dict:
                probs.append(counts_dict[label] / num_elements)
            entropy = sum(-prob * log2(prob) for prob in probs)
            return entropy
        

    def information_gain(self, ig_criterion: str, split_attribute, x: pd.DataFrame, y: pd.Series) -> float:
        x = x.reset_index(drop = True)
        if split_attribute not in x.columns:
            raise ValueError("split_attribute should be a feature")
        values_of_attribute = x[split_attribute].unique()
        weighted_average_ig_criterion_values = []
        current_ig_criterion_value = self.entropy(y.tolist())
        for value in values_of_attribute:
            x_subset = x[x[split_attribute] == value].reset_index(drop = True)
            y_subset = y.iloc[x_subset.index]
            if ig_criterion == 'entropy':
                weighted_average_ig_criterion_values.append((len(y_subset) / len(y)) * self.entropy(y_subset.tolist()))
        return current_ig_criterion_value - sum(weighted_average_ig_criterion_values)
    

    def go_down_a_level(self, row: pd.Series, node: dict | None = None):
        if node is None:
            node = self.root #this gets set by build_tree()
        
        #for when you get to the bottom
        if node.get('feature') is None:
            return node.get('label')
        
        #for when you're in the middle
        feature_splitting_on = node.get('feature')
        if type(feature_splitting_on) == str:
            feature_value = row.loc[feature_splitting_on]
        elif type(feature_splitting_on) == int:
            feature_value = row.iloc[feature_splitting_on]
        else:#shouldn't happen
            raise ValueError("feature_split should be a string or an integer")
        
        #if the feature value is in the child dictionary, go down that branch
        #if the feature value isn't in the child dictionary, return majority label of current node
        if feature_value in node.get('children'):
            child_node = node.get('children').get(feature_value)
            return self.go_down_a_level(row, child_node)
        else:
            monte_cristo = Counter()
            count_dracula = [node]
            while count_dracula:
                current_node = count_dracula.pop()
                if current_node.get('feature') is None:
                    monte_cristo[current_node.get('label')] += 1
                else:
                    for child in current_node.get('children').values():
                        count_dracula.append(child)
            return monte_cristo.most_common()[0][0]        
    

    def build_tree(self, x: pd.DataFrame, y: pd.Series, current_depth: int):
        #If all labels are the same
        if y.nunique() == 1:
            return {"feature": None, "children": {}, "label": y.iloc[0]}
        
        #If we got to the depth limit
        if (self.depth_limit is not None and current_depth >= self.depth_limit):
            label_counts = Counter(y)
            most_common_label = label_counts.most_common()[0][0] #this allows errors but the output isn't deterministic
            return {"feature": None, "children": {}, "label": most_common_label}
        
        #Otherwise find the best feature to split on
        ig_per_feature = {}
        for feature in x.columns:
            ig_per_feature[feature] = self.information_gain(self.ig_criterion, feature, x, y)
        
        best_feature = max(ig_per_feature, key=lambda feature: ig_per_feature[feature])

        node = {}
        node['feature'] = best_feature
        node['children'] = {}

        for value in x[best_feature].unique():
            x_subset = x[x[best_feature] == value].reset_index(drop = True)
            y_subset = y[x[best_feature] == value].reset_index(drop = True)
            if len(y_subset) == 0:
                most_common_label = Counter(y).most_common()[0][0]
                node['children'][value] = {"feature": None, "children": {}, "label": most_common_label}
            else:
                node['children'][value] = self.build_tree(x_subset.drop(columns=[best_feature]), y_subset, current_depth + 1)
        return node


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''
        
        if len(x) == 0:
            raise ValueError("There is no training data")
        if len(y) == 0:
            raise ValueError("There are no labels")
        if len(y) != len(x.index):
            raise ValueError("There should be the same number of training examples are there are labels (right now there is not)")
        if self.ig_criterion not in ['entropy', 'collision']:
            raise ValueError("ig_criterion should be 'entropy' or 'collision'")
        if self.depth_limit is not None and self.depth_limit <= 0:
            raise ValueError("depth_limit should be a positive integer or None")
        
        y_series = pd.Series(y).reset_index(drop = True)
        x = x.reset_index(drop = True)

        self.root = self.build_tree(x, y_series, 0)


    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        predictions = []
        for _, row in x.iterrows():
            predictions.append(self.go_down_a_level(row))
        return predictions