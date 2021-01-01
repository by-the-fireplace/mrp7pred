"""
Data preprocessing for training a new model

Input: A csv file with three columns:
        name, smiles, label

1. Drop nulls
2. Remove duplicates
3. Feature engineering
4. Randomly split train/test

Output: pandas dataframe as pickle files:
        X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
Columns:
        X -> name, [features]
        y -> name, label
"""

__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"

from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from feature_engineer import featurize


def load_data(path: str) -> DataFrame:
    """
    Open source data and clean
            remove duplicates
            remove entries without a valid SMILES

    Parameters
    --------
    path : str
            Directory of the compounds data

    Returns
    --------
    df : pandas.DataFrame
            Cleaned dataset
    """
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=["smiles"])

    # TODO: if name is empty, make smiles as name

    # drop rows with same (name, smiles)
    df = df.drop_duplicates(subset=["name", "smiles"])

    # TODO: check SMILES format

    df = df.reset_index(drop=True)
    return df


def split_train_test(
        df: DataFrame,
        ratio: float = 0.7
        ) -> Tuple[DataFrame, DataFrame]:
    """
    Split processed data into training and test data

    Parameters
    --------
    df : pandas.DataFrame
            Cleaned whole data
    ratio : float
            Ratio of training and test data (training / test)

    Returns
    --------
    (df_train, df_test) : DataFrame
    """
    # sample training data
    chosen_idx = np.random.choice(
        len(df),
        replace=False,
        size=int(ratio * (len(df)))
        )
    df_train = df.iloc[chosen_idx, :]
    mask = ~df.index.isin(df_train.index)
    df_test = df.loc[mask]
    return df_train, df_test


def featurize_and_split(
    df: DataFrame, ratio: float = 0.7
) -> Tuple[Union[DataFrame, ndarray]]:
    """
    Feturize and split

    Parameters
    --------
    df : DataFrame
            Cleaned dataframe

    Returns
    --------
    df : DataFrame
            Featurized data
    """

    print("Featurzing data ... ", end="", flush=True)
    df = featurize(df)
    print("Done!")

    print("Spliting training and test data ... ", end="", flush=True)
    df_train, df_test = split_train_test(df, ratio=ratio)

    # col0: "name", col1: "smiles", col2: "label", col3-130: feaatures
    name_train, name_test = df_train["name"], df_test["name"]
    X_train, y_train = df_train.iloc[:, 3:], df_train["label"]
    X_test, y_test = df_test.iloc[:, 3:], df_test["label"]
    print("Done!")

    return name_train, name_test, X_train, y_train, X_test, y_test
