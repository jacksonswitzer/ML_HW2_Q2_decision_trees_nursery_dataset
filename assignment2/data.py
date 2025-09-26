''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob
import os

import pandas as pd


def load_data(path: str) -> dict:
    '''
    Loads all the data required for this assignment from a given path.

    Args:
        path (str): the path to the data directory (e.g., 'data/cv/')

    Returns:
        dict: a dictionary containing the dataframes from the specified path
    '''

    data_dict = {}
    
    # use the provided path to find and load the cross-validation fold files
    for file_path in glob(os.path.join(path, '*.csv')):
        fold_name = os.path.splitext(os.path.basename(file_path))[0]
        fold_df = pd.read_csv(file_path)
        data_dict[fold_name] = fold_df

    return data_dict