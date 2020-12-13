"""
Main training script
"""

import numpy as np
from numpy import ndarray
import pandas as pd
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
        log_dir: "./output/scores.csv"
    ) -> None:

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
        
    logging(mscv)
        
    clf_best = mscv.best_estimator_
    clf_best_score = mscv.best_score_





