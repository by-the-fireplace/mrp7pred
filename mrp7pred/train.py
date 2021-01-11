"""
Main training script
"""

import pickle
from typing import Union, Any

import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from mrp7pred.grid import grid
from mrp7pred.preprocess import featurize_and_split, load_data
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
    print_log: bool = False,
    verbose: int = 5,
    log_dir: str = OUTPUT,
    model_dir: str = f"{OUTPUT}/model",
) -> Pipeline:

    ensure_folder(log_dir)
    ensure_folder(model_dir)

    pipeline = Pipeline(
        [
            ("sclr", DummyScaler()),
            ("clf", DummyClassifier()),
        ]
    )

    mscv = GridSearchCV(
        pipeline,
        grid,
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        return_train_score=True,
        n_jobs=-1,
        verbose=verbose,
        refit="f1",
    )

    mscv.fit(X_train, y_train)

    def logging(
        model, path=f"{log_dir}/scores_{get_current_time()}.csv", print_log=print_log
    ) -> None:
        score_df = pd.DataFrame(model.cv_results_).T
        if print_log:
            print(f"Cross-validation scores {score_df}")
        score_df.to_csv(path)

    clf_best = mscv.best_estimator_
    with open(f"{model_dir}/best_model_{get_current_time()}.pkl", "wb") as mo:
        pickle.dump(clf_best, mo)

    clf_best_score = mscv.best_score_
    print(f"Best score: {clf_best_score}")

    return clf_best


def run(df: DataFrame, ratio: float = 0.8) -> Any:
    """
    Start training with output info.
    """
    name_train, name_test, X_train, y_train, X_test, y_test = featurize_and_split(
        df, ratio=ratio
    )

    print("Start training ...", end="", flush=True)
    clf_best = _train(X_train, y_train)
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
    plot_roc_auc(y_test, y_score, title=f"ROC Curve, train/test={ratio}")
    print("Done!")

    return clf_best


def main() -> None:
    df = load_data(f"{DATA}/merged.csv")

    # clean, featurization, splitting
    run(df, ratio=0.8)


if __name__ == "__main__":
    main()
