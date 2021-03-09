"""
Helper functions
"""

import os
from datetime import datetime
from typing import Any, Tuple, Union
from pandas import DataFrame
from numpy import ndarray

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import re
import scipy
import scipy.cluster.hierarchy as sch


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
        return Chem.MolFromSmiles(smiles)
    return mol


def draw_molecule(
    smiles: str,
    highlight: str = None,
    subImgSize: Tuple[int, int] = (300, 300),
    transparentBackground: bool = True,
) -> Any:
    """
    Draw 2D structure given a smiles string
    Highlight given substrings
    """
    mol = Chem.MolFromSmiles(smiles)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(subImgSize[0], subImgSize[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # if highlight:
    #     query = Chem.MolFromSmarts(highlight)
    #     # list of atom groups
    #     substructures = list(mol.GetSubstructMatches(query))
    #     atoms_l = [atom for atoms in substructures for atom in atoms]

    #     bonds_l = []
    #     for substructure in substructures:
    #         for bond in query.GetBonds():
    #             aid1 = substructure[bond.GetBeginAtomIdx()]
    #             aid2 = substructure[bond.GetEndAtomIdx()]
    #             bonds_l.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())

    #     return Draw.MolToImage(
    #         mol=mol, size=subImgSize, highlightAtoms=atoms_l, highlightBonds=bonds_l
    #     )
    # else:
    #     return Draw.MolToImage(mol=mol, size=subImgSize)
    if transparentBackground:
        new_svg = ""
        svg = re.sub("<rect .+>", "", svg)
    return svg


def get_molweight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    mw = Chem.Descriptors.ExactMolWt(mol)
    return round(mw, 2)


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


def plot_multicollinearity(
    data: DataFrame,
    figsize: Tuple[int, int],
    dir_path: str = ".",
    filename: str = "multicollinearity",
) -> None:
    """
    Plots multicollinearity matrix, using Pearson's correlation scores, and saves resulting figure as .png file.

    Parameters
    --------
    data: DataFrame
        training set to avoid data leakage.
    dir_path: str
        directory of figure to be saved (default='').
    filename: str
        filename of figure to be saved (default='multicollinearity_matrix').

    Returns
    --------
    None. Displays multicollinearity matrix and saves figure as .png file.
    """
    sns.set_style("whitegrid")
    corr_matrix = data.corr(method="pearson")
    clustered_corr = cluster_corr(corr_matrix)
    plt.figure(figsize=figsize)
    plt.title("Multicollinearity Matrix: Pearson Correlation")
    # mask = (corr_matrix, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig = sns.heatmap(
        corr_matrix,
        # mask=mask,
        cmap=cmap,
        center=0,
        vmax=1,
        vmin=-1,
        annot=False,
        square=True,
        linewidth=0,
        cbar_kws={"shrink": 0.5},
    )
    fig = fig.get_figure()
    fig.savefig(dir_path + "/" + filename + ".png", bbox_inches="tight")


def cluster_corr(corr_array, inplace=False):
    """
    From: https://wil.yegelwel.com/cluster-correlation-matrix/

    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(
        linkage, cluster_distance_threshold, criterion="distance"
    )
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


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


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Apply selected feature id to featurized data
    """

    def __init__(self, selected_feature_id):
        self.selected_feature_id = selected_feature_id

    def fit(self, X: Union[DataFrame, ndarray], y) -> object:
        return self

    def transform(self, X: Union[DataFrame, ndarray]) -> Union[DataFrame, ndarray]:
        return X.iloc[:, self.selected_feature_id]
