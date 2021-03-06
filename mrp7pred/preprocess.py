"""
Automated data preprocessing for training a new model

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

from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from mrp7pred.featurization import featurize


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


def _split_train_test(
    df: DataFrame, ratio: float, random_state: Optional[int] = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Split processed data into training and test data

    Parameters
    --------
    df : pandas.DataFrame
        cleaned data with label
    ratio : float
        ratio of training and test data (training / test)
    random_state: Optional[int]
        random seed for repeat

    Returns
    --------
    (df_train, df_test) : DataFrame
    """
    y = df["label"]
    X = df.loc[:, df.columns != "label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=ratio, random_state=random_state
    )
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    return X_train, X_test, y_train, y_test


def featurize_and_split(
    df: DataFrame,
    ratio: float,
    featurized: bool = True,
    random_state: Optional[int] = None,
    feats_dir: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Tuple[Union[DataFrame, ndarray]]:
    """
    Feturize and split

    Parameters
    --------
    df : DataFrame
        Cleaned dataframe
    ratio: float
        Train-test ratio
    featurized: bool
        True if df is featurized
    random_state: Optional[int]
        random seed for repeat
    feats_dir: Optional[str]
        featurized data output directory

    Returns
    --------
    tuple : Union[DataFrame, ndarray]
        Featurized data: name_train, name_test, X_train, y_train, X_test, y_test
    """
    if not featurized:
        print("Featurzing data ... ")
        df = featurize(df, feats_dir=feats_dir, prefix=prefix)
        print("Done!")

    print("Spliting training and test data ... ", end="", flush=True)
    X_train, X_test, y_train, y_test = _split_train_test(
        df, ratio=ratio, random_state=random_state
    )

    name_train, name_test = X_train["name"], X_test["name"]
    X_train = X_train.drop(["name", "smiles"], axis=1)
    X_test = X_test.drop(["name", "smiles"], axis=1)
    print(
        f"Done!\ntrain_1: {np.sum(y_train)}; train_0: {len(y_train)-np.sum(y_train)}; test_1: {np.sum(y_test)}; test_0: {len(y_test)-np.sum(y_test)}"
    )

    return name_train, name_test, X_train, y_train, X_test, y_test
