from mrp7pred.feats.cdk_features import _cdk_features
from mrp7pred.utils import standardize_smiles


def test_cdk_features():
    smiles = standardize_smiles("CCC")
    assert _cdk_features(smiles) == 44