from mrp7pred.utils import draw_molecule
import pandas as pd


def test_draw_molecules():
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
    for index, row in enumerate(X3.itertuples()):
        smi = getattr(row, "smiles")
        drawing = draw_molecule(smi, subImgSize=(300, 300))
        # print(drawing)
        break


if __name__ == "__main__":
    test_draw_molecules()