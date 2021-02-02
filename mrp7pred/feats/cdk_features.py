from mrp7pred.cinfony_py3 import cdk
from mrp7pred.utils import standardize_smiles


def _cdk_features(smiles):
    mol = cdk.readstring("smi", smiles)
    return int(mol.molwt)
