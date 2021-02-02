import numpy as np
import pandas as pd

from mrp7pred.preprocess import _split_train_test


def test_split_train_test():
    X = np.random.rand(1100, 10)
    y_1 = np.zeros((1000, 1))
    y_0 = np.ones((100, 1))
    y = np.concatenate((y_1, y_0), axis=0)
    data = np.append(X, y, 1)
    print(data.shape)

    df = pd.DataFrame(data=data, columns=[str(i) for i in range(10)] + ["label"])
    X_train, X_test, y_train, y_test = _split_train_test(df, ratio=0.8)
    y_train_1, y_test_1 = sum(y_train), sum(y_test)
    y_train_0, y_test_0 = len(y_train) - y_train_1, len(y_test) - y_test_1
    pos_neg_train = y_train_1 / y_train_0
    pos_neg_test = y_test_1 / y_test_0
    assert abs(pos_neg_train - 0.1) <= 0.05
    assert abs(pos_neg_test - 0.1) <= 0.05


if __name__ == "__main__":
    test_split_train_test()