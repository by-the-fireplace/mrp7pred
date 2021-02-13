"""
Helper functions
"""

import os
from datetime import datetime
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from rdkit import Chem
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

# from tqdm import tqdm

# tqdm.pandas()
sns.set()

# import warnings

# warnings.filterwarnings("ignore")

__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"

DATA = "../data"
OUTPUT = "../output"
MODEL_DIR = f"{OUTPUT}/model"
FIG_DIR = f"{OUTPUT}/fig"


def standardize_smiles(smiles: str) -> str:
    """
    Convert smiles to rdk CanonSmiles
    """
    try:
        mol = Chem.CanonSmiles(smiles)
    except Exception:
        return "error"
    return mol


def draw_molecule(
    smiles: str, highlight: str = None, subImgSize: Tuple[int, int] = (300, 300)
) -> Any:
    """
    Draw 2D structure given a smiles string
    Highlight given substrings
    """
    mol = Chem.MolFromSmiles(smiles)
    if highlight:
        query = Chem.MolFromSmarts(highlight)
        # list of atom groups
        substructures = list(mol.GetSubstructMatches(query))
        atoms_l = [atom for atoms in substructures for atom in atoms]

        bonds_l = []
        for substructure in substructures:
            for bond in query.GetBonds():
                aid1 = substructure[bond.GetBeginAtomIdx()]
                aid2 = substructure[bond.GetEndAtomIdx()]
                bonds_l.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())

        return Chem.Draw.MolToImage(
            mol=mol, size=subImgSize, highlightAtoms=atoms_l, highlightBonds=bonds_l
        )
    else:
        return Chem.Draw.MolToImage(mol=mol, size=subImgSize)


def get_current_time() -> str:
    """
    Generate a timestamp as filename suffix
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def ensure_folder(path: str) -> None:
    """
    Make sure dir exists, if not, create one
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def plot_precision_recall(
    y_test: ndarray,
    y_score: ndarray,
    title: str = "P-R Curve",
    out_dir: str = f"{OUTPUT}/fig",
) -> None:

    ensure_folder(out_dir)
    fig = plt.figure(figsize=(8, 8))

    average_precision = average_precision_score(y_test, y_score)
    pr_curve = precision_recall_curve(y_test, y_score)
    plt.plot(
        pr_curve[0],
        pr_curve[1],
        label="AUC={:.3f}".format(average_precision),
    )

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title(title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 13}, loc="lower right")
    fig.savefig(f"{out_dir}/PR_{get_current_time()}.png")
    plt.show()


def plot_roc_auc(
    y_test: ndarray,
    y_score: ndarray,
    title: str = "ROC Curve",
    out_dir: str = f"{OUTPUT}/fig",
) -> None:

    ensure_folder(out_dir)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, label="AUC={:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="orange", linestyle="--")

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title(title, fontweight="bold", fontsize=15)
    plt.legend(prop={"size": 13}, loc="lower right")
    fig.savefig(f"{out_dir}/ROC_{get_current_time()}.png")
    plt.show()


class DummyClassifier(BaseEstimator):
    def __init__(self, estimator=RandomForestClassifier()):
        self.estimator = estimator

    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X: ndarray, y=None) -> ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: ndarray, y=None) -> ndarray:
        return self.estimator.predict_proba(X)

    def score(self, X: ndarray, y: ndarray) -> float:
        """
        Mean accuracy
        """
        return self.estimator.score(X, y)


class NoScaler(BaseEstimator, TransformerMixin):
    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        return self

    def transform(self, X: ndarray) -> ndarray:
        return X


class DummyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=NoScaler()):
        self.scaler = scaler

    def fit(self, X: ndarray, y: ndarray, **kwargs) -> object:
        return self.scaler.fit(X, y, **kwargs)

    def transform(self, X: ndarray, **kwargs) -> Union[None, ndarray]:
        return self.scaler.transform(X, **kwargs)
