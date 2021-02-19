from mrp7pred.feats.gen_all_features import featurize
import pandas as pd
import numpy as np


def test_gen_feature():
    X1 = pd.DataFrame(
        columns=["name", "smiles"], data=[["Disulfiram", "CCN(CC)C(=S)SSC(=S)N(CC)CC"]]
    )
    X2 = pd.DataFrame(
        columns=["name", "smiles"],
        data=[
            ["Disulfiram", "CCN(CC)C(=S)SSC(=S)N(CC)CC"],
            [
                "Ixazomib Citrate",
                "B1(OC(=O)C(O1)(CC(=O)O)CC(=O)O)C(CC(C)C)NC(=O)CNC(=O)C2=C(C=CC(=C2)Cl)Cl",
            ],
        ],
    )
    X3 = pd.DataFrame(
        columns=["name", "smiles"],
        data=[
            [
                "Paclitaxel",
                "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
            ],
            [
                "Vincristine",
                "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C=O)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O",
            ],
        ],
    )
    # df_feats_X1 = featurize(X1, prefix="test_")
    # df_feats_X2 = featurize(X2, prefix="test_")
    df_feats_X3 = featurize(X3, prefix="test_")
    # print(df_feats_X1)
    # print(df_feats_X2)
    print(df_feats_X3)


if __name__ == "__main__":
    test_gen_feature()