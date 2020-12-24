"""
Main training script
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import pickle
from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold
)

from grid import grid
from utils import DATA, OUTPUT
from preprocess_training import load_data, featurize_and_split


__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"


class DummyClassifier(BaseEstimator):
    def __init__(self, estimator=XGBClassifier()):
        self.estimator = estimator
    
    
    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        self.estimator.fit(X, y, **kwargs)
        return self
    
    
    def predict(self, X: ndarray, y=None) -> ndarray:
        return self.estimator.predict(X)
    
    
    def predict_proba(self, X: ndarray, y=None) -> ndarray:
        return self.estimator.predict_proba(X)
    
    
    def score(self, X: ndarray, y: ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels
        """
        return self.estimator.score(X, y)
    
      
class NoScaler(BaseEstimator, TransformerMixin):
    # A Dummy scaler that does nothing
    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        return self
    
    
    def transform(self, X: ndarray) -> ndarray:
        return X
    
    
class DummyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=NoScaler()):
        self.scaler = scaler
        
        
    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        return self.scaler.fit(X, y, **kwargs)
    
    
    def transform(self, X: ndarray, **kwargs) -> Union[None, ndarray]:
        return self.scaler.transform(X, **kwargs)


def train(
        X_train: ndarray,
        y_train: ndarray,
        logging: bool=True,
        log_dir: str = "./output/scores.csv",
        model_dir: str = "./output/best_model.pkl"
    ) -> float:

    pipeline = Pipeline([("sclr", DummyScaler()), ("clf", DummyClassifier()),])

    # TODO: KFold vs StratifiedKfold
    mscv = GridSearchCV(
        pipeline,
        grid,
        cv=StratifiedKFold(
            n_splits=5, 
            shuffle=False
        ),
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
        refit="f1")

    mscv.fit(X_train, y_train)

    def logging(model, path=log_dir):
        score_df = pd.DataFrame(model.cv_results_).T
        print(f"Cross-validation scores {score_df}")
        score_df.to_csv(path)
    
    if logging:
        logging(mscv)
        
    clf_best = mscv.best_estimator_
    pickle.dump(clf_best, open(model_dir, "wb"))
    
    clf_best_score = mscv.best_score_
    return clf_best_score


def main() -> None:
    df = load_data(f"{DATA}/merged.csv")
    
    # clean, featurization, splitting
    X_train, y_train = featurize_and_split(df)
    
    print("Start training...", end="", flush=True)
    best_score = train(X_train, y_train)
    print("Done!")
    

