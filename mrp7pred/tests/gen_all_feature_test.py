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
    df_feats_X1 = featurize(X1, prefix="test_")
    df_feats_X2 = featurize(X2, prefix="test_")
    print(df_feats_X1)
    print(df_feats_X2)


if __name__ == "__main__":
    test_gen_feature()