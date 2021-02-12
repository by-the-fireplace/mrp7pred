from mrp7pred.mrp7pred import MRP7Pred
import pandas as pd
import numpy as np


def test_m7p_predict():
    df_data = pd.read_csv(
        "./featurized_unknown__full_features_828_20210206-132138.csv", index_col=0
    )
    df_data.dropna(inplace=True)
    features = df_data.iloc[:, :-2]
    support_similar = np.load(
        "../../webserver/featureid_best_model_20210211-031248.npy"
    )
    features_remove_similar = features.iloc[:, support_similar]
    features_remove_similar[["name", "smiles"]] = df_data[["name", "smiles"]]
    m7p = MRP7Pred(clf_dir="../../webserver/best_model_20210211-031248.pkl")
    # out = m7p.predict(featurized_df=features_remove_similar, prefix="test_")
    out = m7p.predict(
        compound_csv_dir="../../data/manual/test.csv",
        selected_features=support_similar,
        prefix="test_",
    )
    assert isinstance(out, pd.DataFrame)


if __name__ == "__main__":
    test_m7p_predict()