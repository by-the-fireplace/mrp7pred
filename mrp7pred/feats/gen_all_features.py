"""
Generate full features from given compound (smiles) list
"""

from mrp7pred.feats.rdk_features import _rdk_features
from mrp7pred.feats.chemopy_features import _chemopy_features
from mrp7pred.utils import get_current_time, ensure_folder, standardize_smiles
from mrp7pred.feats.selection import _remove_similar_features

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy import ndarray
from typing import List, Dict, Tuple, Union, Optional

# from tqdm import tqdm
import time
import datetime
import signal
import pickle
import os

import warnings

warnings.filterwarnings("ignore")

##############################################
#
# Another way to control execution time
#
##############################################
from multiprocessing import Pool, TimeoutError
from time import sleep


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def _timeout_handler(signum, frame):  # raises exception when signal sent
    raise TimeoutException


def _gen_all_features(smi: str) -> Dict[str, Union[int, float]]:
    """
    Generate all features from a given smiles string

    Current number of features (descriptors):
        632 (chemopy) + 196 (rdk) = 828

    PubChem fingerprint is not included here for now
    """
    rdk_features_d = _rdk_features(smi)
    chemopy_features_d = _chemopy_features(smi)
    return {**rdk_features_d, **chemopy_features_d}


def _load_feats(pickle_file: str = "./df_feats.pkl") -> DataFrame:
    """
    Load previously featurized data
    """
    if not os.path.exists(pickle_file):
        return pd.DataFrame()
    with open(pickle_file, "rb") as fi:
        df_feats = pickle.load(fi)
    return df_feats


def featurize(
    X: DataFrame,
    time_limit: int = 30,
    remove_similar: bool = True,
    feats_dir: Optional[str] = None,
    smiles_col_name: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Tuple[ndarray, DataFrame]:
    """
    Batch feature generation from a series of smiles

    Parameters
    --------
    X: DataFrame
        Dataframe with two columns: "name" and "smiles", ("label" if training new model)
    time_limit: int
        Maximum time (s) to featurize one smiles
    remove_similar: bool
        true if use feats.selection._remove_similar_features to select features
    feats_dir: str
        Directory to store featurized data, default "./"
    smiles_col_name: Optional[str]
        The column name of the one with smiles, if df is not None
    prefix: Optional[str]
        The featurized data will be saved to "{feats_dir}/{prefix}_full_features_{feats_len}_{ts}.csv"

    Returns
    ------
    selected_feature_id, df: Tuple[ndarray, DataFrame]
        Featurized data and selected feature id, if remove_similar is False, selected_feature_id will be full columns
    """

    df_feats = DataFrame()

    if smiles_col_name is not None and smiles_col_name not in X.columns:
        raise ValueError(f"column {smiles_col_name} is not in input data")

    # In production environment, don't use the option of "smile_col_name"
    # Ask users to rename their columns instead
    # Comment for development environment
    if "name" not in X.columns or "smiles" not in X.columns:
        raise ValueError("Column 'name' and 'smiles' are required in input data.")

    # drop duplicates
    X = X.drop_duplicates(
        subset=["smiles" if smiles_col_name is None else smiles_col_name]
    )
    total_time = 0.0
    denom = 1
    failed = []
    for index, row in enumerate(X.itertuples()):

        smi = getattr(row, "smiles")
        name = getattr(row, "name")
        try:
            label = getattr(row, "label")
        except:
            label = None

        try:
            smi = standardize_smiles(smi)
        except TypeError as e:
            print(f"{name} featurization failed\nSmiles: {smi}\nError: {e}\n")
            failed.append(name)
            continue

        time_start = datetime.datetime.now()

        # set timer and terminate if exceed time limit
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(time_limit)

        # pool = Pool(processes=1)
        # result = pool.apply_async(_gen_all_features, (smi,))

        try:
            smi_feats_d = _gen_all_features(smi)
            # smi_feats_d = result.get(timeout=time_limit)
        except KeyError as e:  # elements not supported by featurizers
            # smi_feats = np.nan
            print(f"{name} featurization failed\nSmiles: {smi}\nError: {e}\n")
            failed.append(name)
            # df_feats = pd.concat([df_feats, smi_series.to_frame().T])
            continue
        except TypeError as e:
            print(f"{name} featurization failed\nSmiles: {smi}\nError: {e}\n")
            failed.append(name)
            continue
        except TimeoutException:
            print(
                f"{name} Featurization failed\nSmiles: {smi}\nError: Time out ({time_limit}s)\n"
            )
            failed.append(name)
            continue
        # except TimeoutError:
        #     print(
        #         f"{name} Featurization failed\nSmiles: {smi}\nError: Time out ({time_limit}s)\n"
        #     )
        #     failed.append(name)
        #     continue

        elapsed = (datetime.datetime.now() - time_start).total_seconds()
        denom += 1
        total_time += elapsed

        smi_feats_d["name"] = name
        smi_feats_d["smiles"] = smi
        if label is not None:
            smi_feats_d["label"] = label
        # new_row: smiles   features
        smi_feats = pd.Series(smi_feats_d)
        # new_row = smi_series.append(smi_feats)

        # Append generated features to df_feats
        df_feats = pd.concat([df_feats, smi_feats.to_frame().T], ignore_index=True)

        print(
            f"Featurized {index+1}. {name}\nSMILES: {smi}\nTime cost: {round(elapsed, 3)}s\n"
        )

    signal.alarm(0)
    selected_features_id = np.arange(len(df_feats.columns))
    if remove_similar:
        if label is None:
            df_feats_num = df_feats.drop(["name", "smiles"])
        else:
            df_feats_num = df_feats.drop(["name", "smiles", "label"], axis=1)
        df_feats_num = df_feats_num.astype("float64")
        selected_features_id, df_feats_processed = _remove_similar_features(
            df_feats_num, threshold=0.9
        )

        # Add name, smiles, label (if training) back to the dataframe
        df_feats_processed["name"] = df_feats["name"]
        df_feats_processed["smiles"] = df_feats["smiles"]
        if label is not None:
            df_feats_processed["label"] = df_feats["label"]

    if feats_dir:
        ensure_folder(feats_dir)
        ts = get_current_time()
        out_dir = f"{feats_dir}/{prefix}_full_features_{len(df_feats.columns)}_{ts}.csv"
        df_feats.to_csv(out_dir)
        print(f"Featurized data saved to {out_dir}. df_feats.shape: {df_feats.shape}")
    else:
        print(
            f"Featurization finished! Average time: {round(total_time/(denom-1), 3)}s"
        )

    return selected_features_id, df_feats


# if __name__ == "__main__":
#     DATA_DIR = "../../data/all_compounds_with_std_smiles.csv"
#     df = pd.read_csv(DATA_DIR, index_col=0).reset_index(drop=True)

#     df_test = df.loc[155:167, :]
#     df_test_feats = featurize(
#         X=df_test["std_smiles"].values.tolist(),
#         df=df_test,
#         smiles_col_name="std_smiles",
#         out_folder="./all_features_test",
#         prefix="test",
#     )

#     df_man = df.loc[:116, :]
#     df_man_feats = featurize(
#         X=df_man["std_smiles"].values.tolist(),
#         df=df_man,
#         smiles_col_name="std_smiles",
#         out_folder="./all_features_man",
#         prefix="man",
#     )

#     df_cc = df.loc[116:, :]
#     df_cc_feats = featurize(
#         X=df_cc["std_smiles"].values.tolist(),
#         time_limit=10,
#         df=df_cc,
#         smiles_col_name="std_smiles",
#         out_folder="./all_features_cc",
#         prefix="cc",
#     )
