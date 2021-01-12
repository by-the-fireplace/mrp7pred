"""
MRP7Pred class
"""

import pickle
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.pipeline import Pipeline
from typing import Optional, Union, Dict, Any, List

# from mrp7pred.featurization import featurize
from mrp7pred.feats.gen_all_features import featurize
from mrp7pred.preprocess import split_data
from mrp7pred.train import run
from mrp7pred.utils import (
    DATA,
    MODEL_DIR,
    OUTPUT,
    get_current_time,
    DummyClassifier,
    DummyScaler,
    NoScaler,
)

# import warnings

# warnings.filterwarnings("ignore")


class MRP7Pred(object):
    def __init__(
        self,
        clf_dir: str = f"{MODEL_DIR}/best_model_20210111-233521.pkl",
        train_new: bool = False,
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
        grid: Dict[str, Union[List[Any], ndarray]],
        train_test_ratio: float = 0.8,
        featurized: bool = False,
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
        featurized: bool
            True if data has been featurized else False
        grid: Dict
            Grid for GridSearchCV(), defined in MRP7Pred.grid

        Returns
        --------
        self.clf: sklearn.pipeline.Pipeline
            Best model
        """
        if not self.train_new:
            raise ValueError(
                "MRP7Pred was instantiated with train_new=False, execute training process will overwrite the previous model!"
            )

        self.clf_best = run(
            df,
            grid=grid,
            ratio=train_test_ratio,
            featurized=featurized,
        )

    def predict(
        self, compound_csv_dir: Optional[str] = None, df_all: Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Featurize data and make predictions

        Parameters
        --------
        compound_csv_dir: Optional[str]
            The directory of unknown compound data
            with columns "name" and "smiles"
        df_all: Optional[DataFrame]
            Featurized data in dataframe

        Returns
        --------
        pred: ndarray
        """
        if compound_csv_dir is None and df_all is None:
            raise ValueError(
                "'Must provide the path to csv file containing compound smiles 'compound_csv_dir' or a pandas dataframe 'df_all' which stores all compound smiles with compound names."
            )

        if df_all is None:
            df_all = pd.read_csv(compound_csv_dir)

        if "name" not in df_all.columns or "smiles" not in df_all.columns:
            raise ValueError(
                'The input csv should have these two columns: ["name", "smiles"]'
            )

        # only extract name and smiles
        df = df_all[["name", "smiles"]]

        print("Generating features ... ")
        df_feat = featurize(df["smiles"], df=df, smiles_col_name="smiles")
        print("Done!")

        print("Start predicting ...", end="", flush=True)
        df_feat = df_feat.dropna()
        feats = df_feat.drop(columns=["name", "smiles"])
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
