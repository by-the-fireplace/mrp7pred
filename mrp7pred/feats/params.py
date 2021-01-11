from sklearn.feature_selection import (
    GenericUnivariateSelect,
    chi2,
    f_classif,
    mutual_info_classif,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

FEATURE_SELECTION_PARAMS = [
    {
        "fslc": [GenericUnivariateSelect(mode="k_best")],
        "fslc__score_func": [chi2, f_classif, mutual_info_classif],
        "fslc__param": [5, 10, 50, 100, 500, 1000],
        "sclr": []
        "clf": [
            RandomForestClassifier(),
            SVC(probability=True, kernel="linear"),
            XGBClassifier(),
            ComplementNB(),
            MLPCLassifier(),
        ],
    },
    {
        "fslc": [SelectFromModel()],
        "fslc__estimator": [
            RandomForestClassifier(),
            SVC(probability=True, kernel="linear"),
            XGBClassifier(),
            ComplementNB(),
        ],
        "fslc__max_features": [5, 10, 50, 100, 500, 1000],
    },
    {
        "fslc": [RFECV(cv=StratifiedKFold(n_splits=5, shuffle=False))],
        "fslc__estimator": [
            RandomForestClassifier(),
            SVC(probability=True, kernel="linear"),
            XGBClassifier(),
            ComplementNB(),
        ],
        "fslc__min_features_to_select": [5, 50, 500],
    },
]