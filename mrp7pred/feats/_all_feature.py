"""
Generate full features from given compound (smiles) list
"""

from mrp7pred.feats.rdk_features import rdk_feature_list, _rdk_features
from mrp7pred.feats.chemopy_features import _chemopy_features
from mrp7pred.utils import get_current_time

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy import ndarray
from typing import List, Dict, Tuple, Union
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def gen_all_features(smi: str) -> Series:
    """
    Generate all features from a given smiles string
    Notice that _chemopy_features(smi) returns a tuple (feature_list, feats)
    
    Current number of features (descriptors): 
        632 (chemopy) + 196 (rdk) = 828
    
    PubChem fingerprint is not included here for now
    """
    rdk_features = _rdk_features(smi)
    chemopy_feature_list, chemopy_features = _chemopy_features(smi)
    return pd.Series(
        data=rdk_features + chemopy_features,
        index=rdk_feature_list + chemopy_feature_list
    )
    
def featurize(X: List[str], out_folder: str=".") -> DataFrame:
    """
    Batch feature generation from a series of smiles
    
    Parameters
    --------
    X: ndarray
        Smiles stored in ndarray
    out_folder: str
        Directory to store featurized data, default "./"
    
    Returns
    ------
    df: DataFrame
        Featurized data
    """
    df = pd.DataFrame()
    
    # remove duplicates
    ori_len = len(X)
    X = list(set(X))
    if ori_len > len(X):
        print(f"Removed {ori_len - len(X)} duplicates")
    
    
    # set smiles as index
    for index, smi in tqdm(enumerate(X), total=len(X)):
        smi_feats = gen_all_features(smi)
        # try:
        #     smi_feats = gen_all_features(smi)
        # except Exception as e:
        #     smi_feats = np.nan
        #     print(f"Featurization failed\nSmiles: {smi}\nError: {e}")
        #     df = pd.concat([df, pd.Series().to_frame().T])
        #     continue
        smi_series = pd.Series([smi], index=["smiles"])
        new_row = smi_series.append(smi_feats)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    ts = get_current_time()
    out_dir = f"{out_folder}/full_features_828_{ts}.csv"
    df.to_csv(out_dir)
    print(f"Featurized data saved to {out_dir}")
    return df
    

if __name__ == "__main__":
    test_smis = [
        "Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1",
        "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC(OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C",
        "CCC1(O)CC2CN(CCc3c([nH]c4ccccc34)C(C(=O)OC)(c3cc4c(cc3OC)N(C=O)C3C(O)(C(=O)OC)C(OC(C)=O)C5(CC)C=CCN6CCC43C65)C2)C1",
        "CCCCCC=CCC=CC=CC=CC(SCC(NC(=O)CCC(N)C(=O)O)C(=O)NCC(=O)O)C(O)CCCC(=O)O",
        "CC12CCC3c4ccc(O)cc4CCC3C1CCC2OC1OC(C(=O)O)C(O)C(O)C1O",
        "CC12CCC3c4ccc(O)cc4CCC3C1CCC2OC1OC(C(=O)O)C(O)C(O)C1O"
    ]
    df = featurize(test_smis)
    print(f"df.shape: {df.shape}")
    print(df.head())
    