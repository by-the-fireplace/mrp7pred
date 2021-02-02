"""
Grid for training non-NN models
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from mrp7pred.utils import NoScaler

grid_light = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [100, 1000],
        "clf__class_weight": [{0: 0.1, 1: 0.9}, {0: 0.01, 1: 0.99}],
        "sclr": [Normalizer()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__n_estimators": [100, 1000],
        "clf__scale_pos_weight": [10, 100, 1000],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf", "linear"],
        "clf__class_weight": [{0: 0.1, 1: 0.9}, {0: 0.01, 1: 0.99}],
        "sclr": [Normalizer()],
    },
    # {
    #     "clf": [MLPClassifier()],
    #     "clf__solver": ["lbfgs"],
    #     "clf__max_iter": [1000, 2000],
    #     "clf__alpha": [0.1, 1],
    #     "clf__hidden_layer_sizes": [10, 15],
    #     "sclr__scaler": [MinMaxScaler()],
    # },
]

grid_light_balance = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [100, 1000],
        # "clf__class_weight": [{0: 0.1, 1: 0.9}, {0: 0.01, 1: 0.99}],
        "sclr": [StandardScaler()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__n_estimators": [100, 1000],
        # "clf__scale_pos_weight": [1, 10, 100, 1000],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf", "linear"],
        # "clf__class_weight": [{0: 0.1, 1: 0.9}, {0: 0.01, 1: 0.99}],
        "sclr": [StandardScaler()],
    },
    # {
    #     "clf": [MLPClassifier()],
    #     "clf__solver": ["lbfgs"],
    #     "clf__max_iter": [1000, 2000],
    #     "clf__alpha": [0.1, 1],
    #     "clf__hidden_layer_sizes": [10, 15],
    #     "sclr__scaler": [StandardScaler()],
    # },
]

grid = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [10, 100, 300, 500, 700, 1000],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [10, 50, 100, 500, 1000],
        "clf__max_features": [0.1, 0.2, 0.4, 0.8, 1.0, "sqrt", "log2"],
        "sclr__scaler": [StandardScaler(), MinMaxScaler(), Normalizer()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__min_child_weight": [1, 5],
        "clf__gamma": [0, 1, 5],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.4, 0.7, 1.0],
        "clf__learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5],
        "clf__max_depth": [4, 5, 7],
        "clf__n_estimators": [100, 300, 500, 700, 1000],
        "clf__scale_pos_weight": [10, 100, 1000],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf"],
        "clf__C": [1, 10, 100, 1000],
        "clf__gamma": [1e-3, 1e-4],
        "sclr__scaler": [StandardScaler(), MinMaxScaler(), Normalizer()],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["linear"],
        "clf__C": [1, 10, 100, 1000],
        "sclr__scaler": [StandardScaler(), MinMaxScaler(), Normalizer()],
    },
    # {
    #     "clf": [MLPClassifier()],
    #     "clf__solver": ["lbfgs"],
    #     "clf__max_iter": [1000, 1200, 1400, 1600, 1800, 2000],
    #     "clf__alpha": 10.0 ** -np.arange(1, 10),
    #     "clf__hidden_layer_sizes": np.arange(10, 15),
    #     "sclr__scaler": [StandardScaler(), MinMaxScaler(), Normalizer()],
    # },
]

grid_noscaler = [
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [10, 100, 300, 500, 700, 1000],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [10, 50, 100, 500, 1000],
        "clf__max_features": [0.1, 0.2, 0.4, 0.8, 1.0, "sqrt", "log2"],
        "sclr__scaler": [NoScaler()],
    },
    {
        "clf": [XGBClassifier()],
        "clf__min_child_weight": [1, 5],
        "clf__gamma": [0, 1, 5],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.4, 0.7, 1.0],
        "clf__learning_rate": [0.001, 0.01, 0.1, 0.3, 0.5],
        "clf__max_depth": [4, 5, 7],
        "clf__n_estimators": [100, 300, 500, 700, 1000],
        "clf__scale_pos_weight": [10, 100, 1000],
        "sclr__scaler": [NoScaler()],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["rbf"],
        "clf__C": [1, 10, 100, 1000],
        "clf__gamma": [1e-3, 1e-4],
        "sclr__scaler": [NoScaler()],
    },
    {
        "clf": [SVC(probability=True)],
        "clf__kernel": ["linear"],
        "clf__C": [1, 10, 100, 1000],
        "sclr__scaler": [NoScaler()],
    },
    {
        "clf": [MLPClassifier()],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [1000, 1200, 1400, 1600, 1800, 2000],
        "clf__alpha": 10.0 ** -np.arange(1, 10),
        "clf__hidden_layer_sizes": np.arange(10, 15),
        "sclr__scaler": [NoScaler()],
    },
]