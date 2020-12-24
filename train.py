"""
Main training script
"""

import numpy as np
from numpy import ndarray
import pandas as pd
import pickle
from typing import Union
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold
)

from grid import grid
from utils import DATA, OUTPUT, plot_roc_auc, get_scoring, get_current_time, ensure_folder
from preprocess import load_data, featurize_and_split


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
        log_dir: str = OUTPUT,
        model_dir: str = f"{OUTPUT}/model"
    ) -> Pipeline:
    
    ensure_folder(log_dir)
    ensure_folder(model_dir)
    
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

    def logging(model, path=f"{log_dir}/scores_{get_current_time()}.csv"):
        score_df = pd.DataFrame(model.cv_results_).T
        print(f"Cross-validation scores {score_df}")
        score_df.to_csv(path)
    
    if logging:
        logging(mscv)
        
    clf_best = mscv.best_estimator_
    with open(f"{model_dir}/best_model_{get_current_time()}.pkl", "wb") as mo:
        pickle.dump(clf_best, mo)
    
    clf_best_score = mscv.best_score_
    print(f"Best score: {clf_best_score}")
    
    return clf_best


def main() -> None:
    df = load_data(f"{DATA}/merged.csv")
    
    # clean, featurization, splitting
    RATIO = 0.8
    name_train, name_test, X_train, y_train, X_test, y_test = featurize_and_split(df, ratio=RATIO)
    
    print("Start training ...", end="", flush=True)
    clf_best = train(X_train, y_train)
    print("Done!")
    
    print(f"Best model:\n{clf_best}")
    
    print("Evaluate model on test data ... ", end="", flush=True)
    test_score = clf_best.score(X_test, y_test)
    y_pred = clf_best.predict(X_test)
    y_score = [score[1] for score in clf_best.predict_proba(X_test)]
    print(f"Done! Score: {test_score}")
    
    print("Getting full score set ... ", end="", flush=True)
    test_scores = get_scoring(y_test, y_score, y_pred)
    print(f"Done! Full scores: {test_scores}")
    
    print("Plotting ROC for test data ... ", end="", flush=True)
    plot_roc_auc(y_test, y_score, title=f"ROC Curve, train/test={RATIO}")
    print("Done!")

if __name__ == "__main__":
    main()
    

