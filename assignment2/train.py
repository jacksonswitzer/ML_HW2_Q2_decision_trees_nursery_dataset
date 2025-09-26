''' This file contains the functions for training and evaluating a model.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import numpy as np
import pandas as pd

from data import load_data
from model import DecisionTree, MajorityBaseline, Model



def train(model: Model, x: pd.DataFrame, y: list):
    '''
    Learn a model from training data.

    Just trains the model by calling the class's train method; returns nothing, because the class train method returns nothing.
    '''
    model.train(x, y)


def evaluate(model: Model, x: pd.DataFrame, y: list) -> float:
    '''
    Evaluate a trained model against a dataset.

    Assumes the model has already been trained. Pass in train or test data, either works, and the correct labels for the data you pass in. Returns the accuracy of the model on the provided data.
    '''
    preds = model.predict(x)
    return calculate_accuracy(y, preds)


def calculate_accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.

    Pass in two lists of the same length, one predicted labels and one true labels; returns the accuracy of the predictions.
    '''
    print(labels)
    print(predictions)
    if labels is None or predictions is None:
        raise ValueError("Both labels and predictions must not be None.")
    
    if not labels or not predictions:
        raise ValueError("Labels and predictions must not be empty.")
    
    if len(labels) != len(predictions):
        raise ValueError("The length of labels and predictions must be the same.")
    
    correct_predictions = sum(1 for lab, pred in zip(labels, predictions) if lab == pred)
    accuracy = correct_predictions / len(labels)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    
    parser.add_argument("-t", "--train_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("-e", "--eval_path", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument('--model_type', '-m', type=str, required=True, choices=['majority baseline', 'decision tree'], 
        help='Which model type to train')
    parser.add_argument('--depth_limit', '-d', type=int, default=None, 
        help='The maximum depth of a DecisionTree. Ignored if model_type is not "decision_tree".')
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use. Ignored if model_type is not "decision_tree".')
    args = parser.parse_args()


    # load data using the provided paths
    train_df = pd.read_csv(args.train_path)
    train_x = train_df.drop(train_df.columns[-1], axis=1)
    train_y = train_df[train_df.columns[-1]].tolist()

    test_df = pd.read_csv(args.eval_path)
    test_x = test_df.drop(test_df.columns[-1], axis=1)
    test_y = test_df[test_df.columns[-1]].tolist()


    # initialize the model
    if args.model_type == 'majority baseline':
        model = MajorityBaseline()
    else:
        model = DecisionTree(depth_limit=args.depth_limit, ig_criterion=args.ig_criterion)


    # train the model
    train(model=model, x=train_x, y=train_y)

    
    print(f"Model: {args.model_type}")
    if args.model_type == 'decision tree':
        print(f"IG Criterion: {args.ig_criterion}")
        # display 'None' if no depth limit is set
        depth = args.depth_limit if args.depth_limit is not None else "None"
        print(f"Depth limit: {depth}")
    else:
        print("IG Criterion: N/A")
        print("Depth limit: N/A")


    # evaluate model on train and test data
    train_accuracy = evaluate(model=model, x=train_x, y=train_y)
    print(f'Train accuracy: {train_accuracy:.4f}')
    test_accuracy = evaluate(model=model, x=test_x, y=test_y)
    print(f'Test accuracy: {test_accuracy:.4f}')