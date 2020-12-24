"""
Grid for training non-NN models
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"

grid = [
        {
            "clf": [SVC(probability=True)],
            "clf__kernel": ["rbf"], 
            "clf__C": [1, 50, 1000],
            "clf__gamma": [0.001, 0.0001],
            "sclr__scaler": [StandardScaler()],
        },
        {
            "clf": [MLPClassifier()],
            "clf__solver": ["lbfgs"],
            "clf__alpha": 10.0 ** -np.arange(1, 10, 2),
            "clf__hidden_layer_sizes": np.arange(10, 15, 2),
            "sclr__scaler": [MinMaxScaler()],
        },
        {
            "clf": [RandomForestClassifier()],
            "clf__n_estimators": [10, 100],
            "clf__max_depth": [3, 5, 7],
            "sclr__scaler": [MinMaxScaler()],
        }
       ]

grid_light = [
        {
            "clf": [SVC(probability=True)],
            "sclr__scaler": [StandardScaler()],
        },
        {
            "clf": [MLPClassifier()],
            "sclr__scaler": [MinMaxScaler()],
        },
        {
            "clf": [RandomForestClassifier()],
            "sclr__scaler": [MinMaxScaler()],
        }
       ]