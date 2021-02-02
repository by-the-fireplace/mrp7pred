"""
Something wierd happened here: if renaming this file as "cdk_pubchem_fp_test.py", pytest will fail.
"""

from mrp7pred.feats.cdk_pubchem_fingerprint import _pubchem_fingerprint
from mrp7pred.utils import standardize_smiles

import warnings

warnings.filterwarnings("ignore")


def test_cdk_pubchem_fingerprint():
    smiles = standardize_smiles("CCCCN")
    pubchem_fp = _pubchem_fingerprint(smiles)
    assert pubchem_fp.shape == (881,)