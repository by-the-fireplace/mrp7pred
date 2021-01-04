"""
MRP7Pred class
"""

import pickle
import warnings

import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.pipeline import Pipeline
from typing import Optional, Union, Dict, Any, List

from featurization import featurize
from preprocess import featurize_and_split
from train import DummyClassifier, DummyScaler, NoScaler, run
from utils import DATA, MODEL_DIR, OUTPUT, get_current_time
from grid import grid

warnings.filterwarnings("ignore")


class MRP7Pred(object):
    def __init__(
        self,
        clf_dir: str = f"{MODEL_DIR}/best_model_20201224-082918.pkl",
        train_new: bool = False        
    ) -> None:
        """
        Parameters
        --------
        clf_dir: str
        train_new: bool
            Set train_new as True if want to train new model
        """
        self.train_new = train_new
        if not train_new:
            print("Loading trained model ... ", end="", flush=True)
            with open(clf_dir, "rb") as ci:
                self.clf = pickle.load(ci)
            print("Done!")
        
    def run_train(
        self,
        df: DataFrame,
        train_test_ratio: float=0.8,
        grid: Dict[str, Union[List[Any], ndarray]]=grid,
        ):
        """
        Featurize and train models
        
        Parameters
        --------
        df: pandas.DataFrame
            A dataframe containing all data.
            Must have columns: "name", "smiles", "label"
        train_test_ratio: float
            The ratio of training data : test data
        grid: Dict
            Grid for GridSearchCV(), defined in MRP7Pred.grid
        
        Returns
        --------
        self.clf: sklearn.pipeline.Pipeline
            Best model
        """
        self.clf_best = run(df, ratio=train_test_ratio)
        
    
    def predict(self, compound_csv_dir: str) -> DataFrame:
        """
        Featurize data and make predictions

        Parameters
        --------
        compound_csv_dir: str
            The directory of unknown compound data
            with columns "name" and "smiles"

        Returns
        --------
        pred: ndarray
        """
        
        df_all = pd.read_csv(compound_csv_dir)
        if "name" not in df_all.columns or "smiles" not in df_all.columns:
            raise ValueError(
                'The input csv should have these two columns: ["name", "smiles"]'
            )

        df = df_all[["name", "smiles"]]

        print("Generating features ... ")
        df_feat = featurize(df)
        print("Done!")

        print("Start predicting ...", end="", flush=True)
        feats = df_feat.iloc[:, 2:]  # features start from 3rd column
        preds = self.clf.predict(feats)
        scores = [score[1] for score in self.clf.predict_proba(feats)]
        print("Done!")

        print("Writing output ...", end="", flush=True)
        df_out = pd.DataFrame(columns=["name", "smiles", "pred", "score"])
        df_out["name"] = df["name"]
        df_out["smiles"] = df["smiles"]
        df_out["pred"] = preds
        df_out["score"] = scores
        df_out.to_csv(f"{OUTPUT}/predicted_{get_current_time()}.csv")
        print(f"Done! Results saved to: {OUTPUT}/predicted_{get_current_time()}.csv")
        return df_out


def main() -> None:
    m7p = MRP7Pred()
    m7p.predict(f"{DATA}/unknown.csv")


if __name__ == "__main__":
    main()
