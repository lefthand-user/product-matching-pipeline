"""
Generic utility functions 
"""

from typing import Tuple

import pandas as pd


def load_data(path_a: str, path_b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw product datasets for stores A and B.

    Parameters
    ----------
    path_a : str
        Path to grocery_store_a_raw_data.csv
    path_b : str
        Path to grocery_store_b_raw_data.csv

    Returns
    -------
    (df_a, df_b) : Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames for stores A and B.
    """
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    return df_a, df_b
