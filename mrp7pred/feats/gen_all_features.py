"""
Generate full features from given compound (smiles) list
"""

from mrp7pred.feats.rdk_features import rdk_feature_list, _rdk_features
from mrp7pred.feats.chemopy_features import _chemopy_features
from mrp7pred.utils import get_current_time, ensure_folder

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy import ndarray
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
import time
import datetime
import signal
import pickle
import os

import warnings

warnings.filterwarnings("ignore")


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
    if not os.path.exists(pickle_file):
        return pd.DataFrame()
    with open(pickle_file, "rb") as fi:
        df_feats = pickle.load(fi)
    return df_feats


def featurize(
    X: List[str],
    out_folder: str = ".",
    time_limit: int = 10,
    df: Optional[DataFrame] = None,
    smiles_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Batch feature generation from a series of smiles

    Parameters
    --------
    X: ndarray
        Smiles stored in ndarray
    out_folder: str
        Directory to store featurized data, default "./"
    time_limit: int
        Maximum time (s) to featurize one smiles
    df: Optional[DataFrame]
        Input could be a column of a dataframe, the function will keep other columns
    smiles_col_name: Optional[str]
        The column name of the one with smiles, if df is not None

    Returns
    ------
    df: DataFrame
        Featurized data
    """
    ensure_folder(out_folder)
    df_feats = _load_feats()

    if df is None:
        # remove duplicates
        ori_len = len(X)
        X = list(set(X))
        if ori_len > len(X):
            print(f"Removed {ori_len - len(X)} duplicates")
    else:
        if X != df[smiles_col_name].values.tolist():
            raise ValueError("Smiles in df does not match input smiles list")
        if smiles_col_name is None:
            raise ValueError("smiles_col_name is not defined")
        ori_len = len(df)
        df = df.drop_duplicates(subset=[smiles_col_name])
        if ori_len > len(df):
            print(f"Removed {ori_len - len(df)} duplicates")
        X = df[smiles_col_name].values.tolist()

    # set smiles as index
    for index, smi in enumerate(X):
        # skip featurization if already done
        if df_feats.isin([smi]).any().any():
            print(f"(Loaded) {index}. {name}\n")
            continue
        smi_series = pd.Series([smi], index=["smiles"])
        time_start = datetime.datetime.now()

        # set timer and terminate if exceed time limit
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(time_limit)
        try:
            smi_feats_d = _gen_all_features(smi)

        except KeyError as e:  # elements not supported by featurizers
            # smi_feats = np.nan
            print(f"Featurization failed\nSmiles: {smi}\nError: {e}")
            df_feats = pd.concat([df_feats, smi_series.to_frame().T])
            continue
        except TimeoutException:
            print(
                f"Featurization failed\nSmiles: {smi}\nError: Time out ({time_limit}s)"
            )
            df_feats = pd.concat([df_feats, smi_series.to_frame().T])
            continue

        smi_feats = pd.Series(smi_feats_d)
        new_row = smi_series.append(smi_feats)
        df_feats = pd.concat([df_feats, new_row.to_frame().T], ignore_index=True)

        # monitor output
        name = ""
        if df is not None:
            name = df["name"].values[index]
        print(f"{index}. {name}\nSMILES: {smi}\n")

    with open("./df_feats.pkl", "wb") as fo:
        pickle.dump(df_feats, fo)

    if df is not None:
        df_feats = pd.concat(
            [df.reset_index(drop=True).iloc[:, :2], df_feats.reset_index(drop=True)],
            axis=1,
        )

    ts = get_current_time()
    out_dir = f"{out_folder}/full_features_828_{ts}.csv"
    df_feats.to_csv(out_dir)
    print(f"Featurized data saved to {out_dir}")
    return df_feats


if __name__ == "__main__":
    DATA_DIR = "./data/all_compounds_with_std_smiles.csv"
    df = pd.read_csv(DATA_DIR, index_col=0).reset_index(drop=True)

    df_test = df.loc[155:167, :]
    df_test_feats = featurize(
        X=df_test["std_smiles"].values.tolist(),
        df=df_test,
        smiles_col_name="std_smiles",
        out_folder="./all_features_test",
    )

    # df_man = df.loc[:116, :]
    # df_man_feats = featurize(
    #     X=df_man["std_smiles"].values.tolist(),
    #     df=df_man,
    #     smiles_col_name="std_smiles",
    #     out_folder="./all_features_man"
    # )

    df_cc = df.loc[116:, :]
    df_cc_feats = featurize(
        X=df_cc["std_smiles"].values.tolist(),
        time_limit=60,
        df=df_cc,
        smiles_col_name="std_smiles",
        out_folder="./all_features_cc",
    )
