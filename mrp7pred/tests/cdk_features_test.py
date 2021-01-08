from mrp7pred.feats.cdk_features import _cdk_features
from mrp7pred.utils import standardize_smiles


def cdk_features_test():
    smiles = standardize_smiles("CCC")
    assert _cdk_features(smiles) == 44.0956192017


if __name__ == "__main__":
    cdk_features_test()
