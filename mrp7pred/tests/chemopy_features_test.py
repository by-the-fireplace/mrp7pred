import numpy as np
import pandas as pd

from mrp7pred.feats.chemopy_features import _chemopy_features
from mrp7pred.utils import standardize_smiles
from rdkit import Chem


def test_chemopy_features():
    smi = standardize_smiles("Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1")  # gemcitabine
    # print(f"Test SMILES (std): {smi}")
    mol = Chem.MolFromSmiles(smi)
    feats_dict = _chemopy_features(smi)
    # print("all feats: ", feats_dict)
    # print("length: ", len(feats_dict))
    assert isinstance(feats_dict, dict)
    assert len(feats_dict) == 632