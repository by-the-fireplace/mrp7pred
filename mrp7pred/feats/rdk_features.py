"""
Generate RDK features from SMILE strings

1D:
    - Constitutional descriptors (106):

2D:
    - Molecular property descriptors (5):
        ExactMolWt
        MolLogP
        MolMR
        MolWt
        TPSA
    - Connectivity descriptors (12):
        Chi0, Chi1
        Chi0v - 4v
        chi0n - 4n
    - MOE-type descriptors (58):
        EState_VSA1 - 11
        LabuteASA
        PEOE_VSA1 - 14
        SMR_VSA1 - 10
        SlogP_VSA1 - 12
        VSA_EState1 - 10
    - Topological descriptors (15):
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

Total number of descriptors: 196
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

warnings.filterwarnings("ignore")

from typing import List, Union, Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from mrp7pred.utils import standardize_smiles
from mrp7pred.cinfony_py3 import rdk


_feature_list = [
    "FractionCSP3",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "NHOHCount",
    "NOCount",
    "RingCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi1",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Chi4v",
    "Chi0n",
    "Chi1n",
    "Chi2n",
    "Chi3n",
    "Chi4n",
    "EState_VSA1",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "EState_VSA10",
    "EState_VSA11",
    "ExactMolWt",
    "HallKierAlpha",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MolLogP",
    "MolMR",
    "MolWt",
    "PEOE_VSA1",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "SMR_VSA1",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SMR_VSA10",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "TPSA",
    "VSA_EState1",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "VSA_EState10",
    "MaxAbsEStateIndex",
    "MaxAbsPartialCharge",
    "MaxEStateIndex",
    "MaxPartialCharge",
    "MinAbsEStateIndex",
    "MinAbsPartialCharge",
    "MinEStateIndex",
    "MinPartialCharge",
]
# rdk_feature_list = [f"rdk_{feat}" for feat in _feature_list]


def _rdk_features(smi: str) -> Dict[str, Union[float, int]]:
    """
    Generate rdk features from smiles strings

    Parameters
    --------
    smi: str
        Smiles string to be featurized. Should be standardized by
        mrp7pred.utils.standardize_smiles()

    Returns
    --------
    feats: Dict[str, Union[float, int]]
        Dict of generated features
        Feature name as key
    """
    mol_cinfony = rdk.readstring("smi", smi)
    feats = []
    for feat_name in _feature_list:
        feats.append(mol_cinfony.calcdesc([feat_name])[feat_name])
    return dict(zip(_feature_list, feats))


if __name__ == "__main__":
    test_smi = standardize_smiles("Nc1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1")  # gemcitabine
    print(f"Test SMILES (std): {test_smi}")
    print(dict(zip(_feature_list, _rdk_features(test_smi))))
