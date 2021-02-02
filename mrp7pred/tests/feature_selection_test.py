import numpy as np
import pandas as pd

from mrp7pred.feats.feature_selection import (
    _remove_similar_features,
    _remove_low_variance_features,
)


def test_remove_similar_features() -> None:
    df_feats = pd.DataFrame(
        np.array(
            [
                [0.99, 0.76, 0.76, 0.34],
                [0.56, 0.99, 0.99, 0.20],
                [0.45, 0.12, 0.12, 0.91],
                [0.67, 0.99, 0.99, 0.80],
                [0.78, 0.65, 0.65, 0.68],
                [0.87, 0.83, 0.83, 0.76],
                [0.65, 0.73, 0.73, 0.66],
                [0.56, 0.19, 0.19, 0.14],
            ]
        ),
        columns=["A", "B1", "B2", "D"],
    )
    correlation_matrix = df_feats.corr().abs()
    # print(correlation_matrix.head())
    to_drop = _remove_similar_features(correlation_matrix, threshold=0.9)
    # print(f"Number of features to be removed: {len(to_drop)}")
    # print("Features to be removed:")
    # print(df_feats.columns[to_drop])
    assert to_drop == [1] or [2]


def test_remove_low_variance_features():
    X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    X_reduced = _remove_low_variance_features(X)
    assert X_reduced.shape == (3, 2)