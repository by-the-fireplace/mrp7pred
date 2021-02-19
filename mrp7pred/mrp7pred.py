"""
MRP7Pred class
"""

import pickle
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.pipeline import Pipeline
from typing import Optional, Union, Dict, Any, List, Callable

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
    ensure_folder,
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

    def auto_train_test(
        self,
        df: DataFrame,
        grid: Dict[str, Union[List[Any], ndarray]],
        cv_n_splits: int = 5,
        verbose: int = 10,
        n_jobs: int = -1,
        train_test_ratio: float = 0.8,
        scoring: Union[str, callable] = "accuracy",
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
            cv_n_splits=cv_n_splits,
            ratio=train_test_ratio,
            featurized=featurized,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def predict(
        self,
        selected_features_arr: Optional[ndarray] = None,
        compound_csv_dir: Optional[str] = None,
        compound_df: Optional[DataFrame] = None,
        featurized_df: Optional[DataFrame] = None,
        prefix: Optional[str] = None,
    ) -> DataFrame:
        """
        Featurize data and make predictions

        Parameters
        --------
        compound_csv_dir: Optional[str]
            The directory of unknown compound data
            with columns "name" and "smiles"
        selected_features: Optional[ndarray]
            index of selected features
        featurized_df: Optional[DataFrame]
            Featurized data in dataframe
        prefix: Optional[str]
            Prediction results output filename prefix

        Returns
        --------
        pred: ndarray
        """
        if compound_csv_dir is None and compound_df is None:
            raise ValueError(
                "Must pass either the path to csv file containing compound smiles to 'compound_csv_dir' or a dataframe with columns 'name' and 'smiles' to 'compound_df"
            )

        if featurized_df is None:
            if compound_csv_dir:
                df = pd.read_csv(compound_csv_dir)
            elif compound_df is not None:
                df = compound_df

        else:
            if (
                "name" not in featurized_df.columns
                or "smiles" not in featurized_df.columns
            ):
                raise ValueError(
                    'The input csv should have these two columns: ["name", "smiles"]'
                )

            # only extract name and smiles
            df = featurized_df[["name", "smiles"]]
            df_feat = featurized_df.drop(["name", "smiles"], axis=1)

        if featurized_df is None:
            print("Generating features ... ")
            # df_feats should be purely numeric
            df = featurize(df, prefix=prefix)
            df_feat = df.drop(["name", "smiles"], axis=1)
            # print("Done!")
        print("Start predicting ... ", end="", flush=True)
        if selected_features_arr is not None:
            df_feat = df_feat.dropna().iloc[:, selected_features_arr]
        preds = self.clf.predict(df_feat)
        scores = [score[1] for score in self.clf.predict_proba(df_feat)]
        print("Done!")

        # print("Writing output ... ", end="", flush=True)
        df_out = pd.DataFrame(columns=["name", "smiles", "pred", "score"])
        df_out["name"] = df["name"]
        df_out["smiles"] = df["smiles"]
        df_out["pred"] = preds
        df_out["score"] = scores
        # ensure_folder(OUTPUT)
        # df_out.to_csv(f"{OUTPUT}/{prefix}predicted_{get_current_time()}.csv")
        # print(
        #     f"Done! Results saved to: {OUTPUT}/{prefix}predicted_{get_current_time()}.csv"
        # )
        return df_out


def main() -> None:
    m7p = MRP7Pred()
    m7p.predict(f"{DATA}/unknown.csv")


if __name__ == "__main__":
    main()
