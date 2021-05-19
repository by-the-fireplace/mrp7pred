"""
Grid for training non-NN models
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


grid = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [100, 1000],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7],
        "sclr": [StandardScaler(), MinMaxScaler()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__gamma": [0, 1, 5],
        "clf__learning_rate": [0.001, 0.01, 0.1],
        "clf__n_estimators": [100, 1000],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf", "linear"],
        "clf__C": [1, 10, 100, 1000],
        "sclr": [StandardScaler(), MinMaxScaler()],
    },
]

grid_imbalance = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [10, 100, 1000],
        "clf__class_weight": [{0: 1, 1: 1000}, {0: 1, 1: 10}],
        "sclr": [StandardScaler()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__n_estimators": [10, 100, 1000],
        "clf__scale_pos_weight": [1, 10, 100, 1000],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf", "linear"],
        "clf__class_weight": [{0: 1, 1: 1000}, {0: 1, 1: 10}],
        "sclr": [StandardScaler()],
    },
]