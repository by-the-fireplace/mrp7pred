"""
Generate ChemoPy features from SMILE strings

ChemoPy: https://github.com/salotz/chemopy

Number of descriptors: 309
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
from mrp7pred.pychem_py3 import (
    constitution,
    topology,
    connectivity,
    kappa,
    bcut,
    basak,
    estate,
    moran,
    geary,
    molproperty,
    charge,
    moe,
    moreaubroto
)

from rdkit import Chem
from mrp7pred.utils import standardize_smiles

def _chemopy_features(smi: str):
    """
    Generate chemopy features from smiles strings
    
    Parameters
    --------
    smi: str
        Smiles string to be featurized. Should be standardized by 
        mrp7pred.utils.standardize_smiles()
    
    Returns
    --------
    feats: List[Union[float, int]]
        List of generated features
    """
    mol = Chem.MolFromSmiles(smi)
    feat_constitution = constitution.GetConstitutional(mol) # 30
    feat_topo = topology.GetTopology(mol) # 35
    feat_connect = connectivity.GetConnectivity(mol) # 44
    feat_kappa = kappa.GetKappa(mol) # 7
    feat_burden1 = bcut.CalculateBurdenVDW(mol) # 16
    feat_burden2 = bcut.CalculateBurdenPolarizability(mol) # 16
    feat_basak = basak.GetBasak(mol) # 21 
    feat_estate = estate.GetEstate(mol) # 89
    feats_moran = moran.GetMoranAuto(mol) # 32
    feats_geary = geary.GetGearyAuto(mol) # 32
    feats_molproperty = molproperty.GetMolecularProtey(mol) # 6
    feat_charge = charge.GetCharge(mol) # 25
    feat_moe = moe.GetMOE(mol) # 60
    feat_moreau_broto_auto = moreaubroto.GetMoreauBrotoAuto(mol) # 32
    
    feats_dict = (
        feat_constitution
        & feat_topo
        & feat_connect
        & feat_kappa
        & feat_burden1
        & feat_burden2
        & feat_basak
        & feat_estate
        & feats_moran
        & feats_geary
        & feats_molproperty
        & feat_charge
        & feat_moe
        & feat_moreau_broto_auto
    )
    
    feature_list, feats = [], []
    for k, v in feats_dict.items():
        feature_list.append(f"pychem_{k}")
        feats.append(v)
    
    return feats
    
    
if __name__ == "__main__":
    smi = standardize_smiles("Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1") # gemcitabine
    print(f"Test SMILES (std): {smi}")
    mol = Chem.MolFromSmiles(smi)
    feats = _chemopy_features(smi)
    print("all feats: ", feats)
    print("length: ", len(feats))