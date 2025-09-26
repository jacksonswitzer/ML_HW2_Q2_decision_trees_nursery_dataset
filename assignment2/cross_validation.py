''' This file contains the functions for performing cross-validation.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from data import load_data
from model import DecisionTree
from train import train, evaluate


def cross_validation(cv_folds: List[pd.DataFrame], depth_limit_values: List[int], ig_criterion: str) -> Tuple[int, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of dataframes, each corresponding to a fold of the data
        depth_limit_values (list): a list of depth_limit hyperparameter values to try
        ig_criterion (str): the information gain variant to use. Should be one of "entropy" or "collision".

    Returns:
        int: the best depth_limit hyperparameter discovered
        float: the average cross-validation accuracy corresponding to the best depth_limit
    '''
    best_depth = -1
    best_avg_accuracy = 0.0

    # YOUR CODE HERE
    # Implement the k-fold cross-validation logic.
    # For each depth in depth_limit_values, you should find the average accuracy
    # across the k folds and keep track of the best depth.


    return best_depth, best_avg_accuracy


if __name__ == '__main__':
    # setup to handle command-line arguments
    parser = argparse.ArgumentParser(description='Run cross-validation for a Decision Tree')
    parser.add_argument("-c", "--cv_path", type=str, required=True, help="Path to the cross-validation data directory")
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use.')
    args = parser.parse_args()

    # load data using the provided path
    data_dict = load_data(args.cv_path)
    cv_folds = list(data_dict.values())
    
    # --- STUDENT TASK ---
    # define the list of depths you want to test.
    # the assignment requires you to test depths up to 12.
    
    depths_to_test = list(range(1, 13))
    
    # run cross_validation using command-line arguments
    print(f"Running cross-validation for depths: {depths_to_test}...")
    best_depth, best_accuracy = cross_validation(
        cv_folds=cv_folds, 
        depth_limit_values=depths_to_test, 
        ig_criterion=args.ig_criterion)
    
    # print final results
    print('-----')
    print(f"Best depth found: {best_depth}")
    print(f"Best average CV accuracy: {best_accuracy:.4f}")
