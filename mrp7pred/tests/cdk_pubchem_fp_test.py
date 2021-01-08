from mrp7pred.feats.cdk_pubchem_fingerprint import _pubchem_fingerprint
from mrp7pred.utils import standardize_smiles

import warnings

warnings.filterwarnings("ignore")


def cdk_pubchem_fingerprint_test():
    smiles = standardize_smiles("CCCCN")
    pubchem_fp = _pubchem_fingerprint(smiles)
    assert pubchem_fp.shape == (881,)


if __name__ == "__main__":
    cdk_pubchem_fingerprint_test()