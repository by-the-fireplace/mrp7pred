"""
Scoring functions
"""

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)
from numpy import ndarray
import numpy as np
from typing import Union, Dict


def tp(y_true: ndarray, y_pred: ndarray) -> float:
    return float(confusion_matrix(y_true, y_pred)[1, 1])


def fp(y_true: ndarray, y_pred: ndarray) -> float:
    return float(confusion_matrix(y_true, y_pred)[0, 1])


def tn(y_true: ndarray, y_pred: ndarray) -> float:
    return float(confusion_matrix(y_true, y_pred)[0, 0])


def fn(y_true: ndarray, y_pred: ndarray) -> float:
    return float(confusion_matrix(y_true, y_pred)[1, 0])


def specificity(y_true: ndarray, y_pred: ndarray) -> float:
    _tn = tn(y_true, y_pred)
    _fp = fp(y_true, y_pred)
    if _tn == 0:
        return 0
    return _tn / (_tn + _fp)


def recall(y_true: ndarray, y_pred: ndarray) -> float:
    return recall_score(y_true, y_pred, average="binary")


def precision(y_true: ndarray, y_pred: ndarray) -> float:
    return precision_score(y_true, y_pred, average="binary")


def mcc(y_true: ndarray, y_pred: ndarra) -> float:
    return matthews_corrcoef(y_true, y_pred)


def f1(y_true: ndarray, y_pred: ndarray) -> float:
    return f1_score(y_true, y_pred, average="binary")


def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def log_loss(y_true: ndarray, y_pred: ndarray) -> float:
    try:
        rval = log_loss(y_true, y_pred)
    except ValueError:
        print("Error: Cannot calculate log loss because Pr(y = 1) = 0")
    return rval


def roc_auc(y_true: ndarray, y_score: ndarray) -> float:
    try:
        rval = np.float(roc_auc_score(y_true, y_score, average="macro"))
    except ValueError:
        print("Error: Monoclass. Check test data.")
    return rval


def get_scoring(
    y_true: ndarray, y_score: ndarray, y_pred: ndarray
) -> Dict[str, Dict[str, Union[int, float]]]:
    return {
        "stats": {
            "tp": tp(y_true, y_pred),
            "fp": fp(y_true, y_pred),
            "tn": tn(y_true, y_pred),
            "fn": fn(y_true, y_pred),
        },
        "score": {
            "roc_auc": roc_auc(y_true, y_score),
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "specificity": specificity(y_true, y_pred),
            "mcc": mcc(y_true, y_pred),
        },
    }
