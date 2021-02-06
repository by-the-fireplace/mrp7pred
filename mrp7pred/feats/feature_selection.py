"""
Automated feature selection based on training data

Steps:
    1. Remove features with low variance
    2. Remove similar (highly-linearly-correlated) features
    3. Feature selection pipeline:
        - sklearn.feature_selection.GenericUnivariateSelect()
            chi2
            f_classif
            mutual_info_classif
        - sklearn.feature_selection.SelectFromModel()
            l1-based
            tree-based
        - sklearn.feature_selection.RFECV()
        - sklearn.feature_selection.SequentialFeatureSelector()

Need to automate the process
"""

from sklearn.feature_selection import (
    VarianceThreshold,
    GenericUnivariateSelect,
    SelectorMixin,
    chi2,
    f_classif,
    mutual_info_regression,
    SelectKBest,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from typing import Union, Dict, List, Tuple, Callable
from pandas import DataFrame
from numpy import ndarray

import pandas as pd
import numpy as np

from mrp7pred.feats._correlation_graph import CorrelationGraph

# from mrp7pred.feats.params import FEATURE_SELECTION_PARAMS
from mrp7pred.utils import DummyClassifier


def _remove_low_variance_features(
    X: Union[ndarray, DataFrame], threshold=0.0
) -> Tuple[ndarray, DataFrame]:
    """
    Remove all low-variance features
    """
    selector = VarianceThreshold()
    selector.fit(X)
    return selector.get_support(indices=True), selector.fit_transform(X)


def _remove_similar_features(
    X: Union[ndarray, DataFrame], threshold: float = 0.9
) -> Tuple[ndarray, DataFrame]:
    """
    Remove features with high colinearity
    Use a graph-based method
    The goal is to remove actually identical features with different names

    Parameters
    --------
    X: DataFrame
        Featurized data without label and non-numeric columns
    threshold: float
        Above which is considered similar

    Returns
    --------
    support: List[int]
        List of column indices remained
    """
    print("Calculating correlation matrix ... ", end="", flush=True)
    if isinstance(X, ndarray):
        X = pd.DataFrame(X)
    correlation_matrix = X.corr().abs()
    print("Done!")
    print("Creating correlation graph ... ", end="", flush=True)
    cg = CorrelationGraph(correlation_matrix, threshold=threshold)
    print("Done!")
    to_drop = cg.prune()
    support = list(set(range(X.shape[1])) - set(to_drop))
    return np.array(support), X.drop(X.columns[to_drop], axis=1)


def _univariate(
    X: Union[ndarray, DataFrame],
    y: ndarray,
    n_features: int = 5,
    score_function: Callable[[Union[DataFrame, ndarray]], ndarray] = chi2,
) -> ndarray:
    """
    Feature selection (filtering) based on univariate selection

    Parameters
    --------
    X: Union[ndarray, DataFrame]
        Featurized training data
    y: ndarray
        Data labels
    n_features: int
        Number of features to select
    score_function: Callable[Union[DataFrame, ndarray], ndarray]
        Feature selection function

    Returns
    --------
    idx_features: ndarray
        Indices of remaining features
    """
    transformer = GenericUnivariateSelect(
        score_function, mode="k_best", param=n_features
    )
    transformer.fit(X, y)
    return transformer.get_support(indices=True)


def _from_model(
    X: Union[ndarray, DataFrame],
    y: ndarray,
    estimator: Union[BaseEstimator, Pipeline],
    max_features: int = 200,
) -> ndarray:
    """
    Feature selection using sklearn SelectFromModel()

    Parameters
    --------
    X: Union[ndarray, DataFrame]
        Featurized training data
    y: ndarray
        Data labels
    estimator: BaseEstimator
        Classifier used for select features
    max_features: int
        Number of features to select

    Returns
    --------
    idx_features: ndarray
        Indices of remaining features
    """
    selector = SelectFromModel(estimator=estimator, max_features=max_features)
    selector.fit(X, y)
    return selector.get_support(indices=True)


def _rfecv(
    X: Union[ndarray, DataFrame],
    y: ndarray,
    estimator: Union[BaseEstimator, Pipeline],
    step: Union[int, float] = 0.05,
    min_features_to_select: int = 1,
    cv=StratifiedKFold(n_splits=5, shuffle=False),
    verbose: int = 5,
    n_jobs=-1,
) -> ndarray:
    """
    Feature selection using RFECV
    """
    selector = RFECV(
        estimator,
        step=step,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        min_features_to_select=min_features_to_select,
    )
    selector.fit(X, y)
    return selector.get_support(indices=True)