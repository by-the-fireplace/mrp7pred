"""
Main training script
"""

import pickle
from typing import Union, Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

# from xgboost import XGBClassifier
from sklearn.utils import class_weight
from sklearn.metrics import recall_score

from mrp7pred.grid import grid
from mrp7pred.preprocess import _split_train_test, featurize_and_split, load_data
from mrp7pred.utils import (
    DATA,
    OUTPUT,
    ensure_folder,
    get_current_time,
    plot_roc_auc,
    DummyClassifier,
    NoScaler,
    DummyScaler,
)
from mrp7pred.scoring import get_scoring

__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"


def _train(
    X_train: ndarray,
    y_train: ndarray,
    grid: Dict[str, Union[List[Any], ndarray]],
    cv_n_splits: int,
    scoring: Union[str, callable] = "accuracy",
    verbose: int = 10,
    n_jobs: int = -1,
    print_log: bool = True,
    log_dir: str = OUTPUT,
) -> Pipeline:

    # ensure_folder(log_dir)

    pipeline = Pipeline(
        [
            ("sclr", DummyScaler()),
            ("clf", DummyClassifier()),
        ]
    )

    mscv = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=StratifiedKFold(n_splits=cv_n_splits, shuffle=False),
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
        refit="f1",
    )

    mscv.fit(X_train, y_train)

    # def logging(
    #     model, path=f"{log_dir}/scores_{get_current_time()}.csv", print_log=print_log
    # ) -> None:
    #     score_df = pd.DataFrame(model.cv_results_).T
    #     if print_log:
    #         print(f"Cross-validation scores {score_df}")
    #     score_df.to_csv(path)

    clf_best = mscv.best_estimator_

    # logging(mscv)

    clf_best_score = mscv.best_score_
    print(f"Best score: {clf_best_score}")

    return clf_best


def run(
    df: DataFrame,
    grid: Dict[str, Union[List[Any], ndarray]],
    cv_n_splits: int,
    ratio: float,
    verbose: int = 10,
    n_jobs: int = -1,
    scoring: Union[str, callable] = "accuracy",
    featurized: bool = False,
    model_dir: Optional[str] = None,
    feats_dir: Optional[str] = None,
    random_state: Optional[int] = None,
    prefix: Optional[str] = None,
) -> BaseEstimator:
    """
    Start training with output info.
    """
    name_train, name_test, X_train, y_train, X_test, y_test = featurize_and_split(
        df,
        ratio=ratio,
        featurized=featurized,
        feats_dir=feats_dir,
        random_state=random_state,
        prefix=prefix,
    )

    print("Start training ... ", end="", flush=True)
    clf_best = _train(
        X_train,
        y_train,
        grid=grid,
        cv_n_splits=cv_n_splits,
        verbose=verbose,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    print("Done!")

    print(f"Best model:\n{clf_best}")

    if model_dir is None:
        model_dir = f"{OUTPUT}/model"
    pkl_name = f"{model_dir}/best_model_{get_current_time()}.pkl"
    with open(pkl_name, "wb") as mo:
        pickle.dump(clf_best, mo)
    print(f"Best model saved to: {pkl_name}")

    print("Evaluate model on test data ... ", end="", flush=True)
    test_score = clf_best.score(X_test, y_test)
    y_pred = clf_best.predict(X_test)
    y_score = [score[1] for score in clf_best.predict_proba(X_test)]
    print(f"Done! Score: {test_score}")

    print("Getting full score set ... ", end="", flush=True)
    test_scores = get_scoring(y_test, y_score, y_pred)
    print("Done!")
    for item, d in test_scores.items():
        print(item)
        for title, val in d.items():
            print(f"{title}: {val}")

    print("Plotting ROC for test data ... ", end="", flush=True)
    plot_roc_auc(y_test, y_score, title=f"ROC Curve, train/test={ratio}")
    print("Done!")

    return clf_best


def main() -> None:
    df = load_data(f"{DATA}/merged.csv")

    # clean, featurization, splitting
    run(df, ratio=0.8)


if __name__ == "__main__":
    main()
