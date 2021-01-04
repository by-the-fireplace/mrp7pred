"""
Generate RDK features from SMILE strings

1D: 
    Constitutional descriptors (106):
        
2D:
    Molecular property descriptors (5):
        ExactMolWt
        MolLogP
        MolMR
        MolWt
        TPSA        
    Connectivity descriptors (12): 
        Chi0, Chi1
        Chi0v - 4v
        chi0n - 4n
    MOE-type descriptors (58):
        EState_VSA1 - 11
        LabuteASA
        PEOE_VSA1 - 14
        SMR_VSA1 - 10
        SlogP_VSA1 - 12
        VSA_EState1 - 10
    Topological descriptors (15):
        BalabanJ
        BertzCT
        HallKierAlpha
        Ipc
        Kappa1 - 3
        MaxAbsEStateIndex
        MaxAbsPartialCharge
        MaxEStateIndex
        MaxPartialCharge
        MinAbsEStateIndex
        MinAbsPartialCharge
        MinEStateIndex
        MinPartialCharge
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from mrp7pred.utils import standardize_smiles
from mrp7pred.cinfony import rdk


feature_list = (
    [
        "atomic_mass_high",
        "atomic_mass_low",
        "gasteiger_charge_high",
        "gasteiger_charge_low",
        "crippen_logp_high",
        "crippen_logp_low",
        "crippen_mr_high",
        "crippen_mr_low",
        "chi0n",
        "chi1n",
        "chi2n",
        "chi3n",
        "chi4n",
        "chi0v",
        "chi1v",
        "chi2v",
        "chi3v",
        "chi4v",
        "MolLogP",
        "MolMR",
        "ExactMolWt",
        "FractionCSP3",
        "HallKierAlpha",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        "LabuteASA",
        "NumHeterocycles",
        "NumAromaticHeterocycles",
        "NumSaturatedHeterocycles",
        "NumAliphaticHeterocycles",
        "NumAromaticCarbocycles",
        "NumSaturatedCarbocycles",
        "NumAliphaticCarbocycles",
        "NumRings",
        "NumAromaticRings",
        "NumSaturatedRings",
        "NumAliphaticRings",
        "NumAtomStereoCenters",
        "NumBridgeheadAtoms",
        "NumHBA",
        "NumHBD",
        "NumHeteroatoms",
        "NumLipinskiHBA",
        "NumLipinskiHBD",
        "NumRotatableBonds",
        "NumSpiroAtoms",
        "TPSA",
    ]
    + [f"PEOE_VSA{i}" for i in range(1, 15)]
    + [f"SMR_VSA{i}" for i in range(1, 11)]
    + [f"SlogP_VSA{i}" for i in range(1, 13)]
    + [f"MQN{i}" for i in range(1, 43)]
)


def _rdk_features(smi: str) -> list:
    """
    Featurization single smiles
    """
    mol = Chem.MolFromSmiles(smi)
    bcut2d = _rdMolDescriptors.BCUT2D(mol)  # rdkit.rdBase._vectd, len=8
    chiNn = [
        _rdMolDescriptors.CalcChiNn(mol, i) for i in range(5)
    ]  # list, len=5
    chiNv = [
        _rdMolDescriptors.CalcChiNv(mol, i) for i in range(5)
    ]  # list, len=5
    mollogp, molmr = _rdMolDescriptors.CalcCrippenDescriptors(mol)  # floats
    molwt = _rdMolDescriptors.CalcExactMolWt(mol)  # float
    csp3 = _rdMolDescriptors.CalcFractionCSP3(mol)  # float
    hka = _rdMolDescriptors.CalcHallKierAlpha(mol)  # float
    kappa = [
        _rdMolDescriptors.CalcKappa1(mol),
        _rdMolDescriptors.CalcKappa2(mol),
        _rdMolDescriptors.CalcKappa3(mol),
    ]  # list, len=3
    labuteasa = _rdMolDescriptors.CalcLabuteASA(mol)  # float
    num_hetero_cycles = [
        _rdMolDescriptors.CalcNumHeterocycles(mol),
        _rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        _rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        _rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
    ]  # list[int], [total, aromatic, saturated, aliphatic], len=4
    num_carbo_cycles = [
        _rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        _rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        _rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
    ]  # list[int], [total, aromatic, saturated, aliphatic], len=3
    num_rings = [
        _rdMolDescriptors.CalcNumRings(mol),
        _rdMolDescriptors.CalcNumAromaticRings(mol),
        _rdMolDescriptors.CalcNumSaturatedRings(mol),
        _rdMolDescriptors.CalcNumAliphaticRings(mol),
    ]  # list[int], [total, aromatic, saturated, aliphatic], len=4
    num_stereo_centers = _rdMolDescriptors.CalcNumAtomStereoCenters(mol)  # int
    num_bridgehead_atoms = _rdMolDescriptors.CalcNumBridgeheadAtoms(mol)  # int
    num_hba = _rdMolDescriptors.CalcNumHBA(mol)  # int
    num_hbd = _rdMolDescriptors.CalcNumHBD(mol)  # int
    num_hetero_atom = _rdMolDescriptors.CalcNumHeteroatoms(mol)  # int
    num_lipinski_hba = _rdMolDescriptors.CalcNumLipinskiHBA(mol)  # int
    num_lipinski_hbd = _rdMolDescriptors.CalcNumLipinskiHBD(mol)  # int
    num_rot_bonds = _rdMolDescriptors.CalcNumRotatableBonds(mol)  # int
    num_spiro_atoms = _rdMolDescriptors.CalcNumSpiroAtoms(mol)  # int
    tpsa = _rdMolDescriptors.CalcTPSA(mol)  # float
    peoe_vsa = _rdMolDescriptors.PEOE_VSA_(mol)  # list, len=14
    smr_vsa = _rdMolDescriptors.SMR_VSA_(mol)  # list, len=10
    slogp_vsa = _rdMolDescriptors.SlogP_VSA_(mol)  # list, len=12
    mqn = _rdMolDescriptors.MQNs_(mol)  # list, len=42

    feats = (
        [ele for ele in bcut2d]
        + chiNn
        + chiNv
        + [mollogp, molmr, molwt, csp3, hka]
        + kappa
        + [labuteasa]
        + num_hetero_cycles
        + num_carbo_cycles
        + num_rings
        + [num_stereo_centers, num_bridgehead_atoms, num_hba, num_hbd]
        + [num_hetero_atom, num_lipinski_hba, num_lipinski_hbd]
        + [num_rot_bonds, num_spiro_atoms, tpsa]
        + peoe_vsa
        + smr_vsa
        + slogp_vsa
        + mqn
    )

    return feats


if __name__ == "__main__":
    test_smi = standardize_smiles("Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1") # gemcitabine
    mol = rdk.readstring("smi", test_smi)
    test_rdk_feats = _rdk_features(test_smi)
    print("Length RDK feats: ", len(test_rdk_feats))
    print("Previous RDK feats: ", test_rdk_feats)
    
    print(f"Test SMILES (std): {test_smi}")
    print("HeavyAtomCount: ", mol.calcdesc(['HeavyAtomCount'])['HeavyAtomCount'])
    