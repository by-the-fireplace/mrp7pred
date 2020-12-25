"""
MRP7Pred class
"""

from numpy import ndarray
import pickle
import pandas as pd
from pandas import DataFrame

from utils import MODEL_DIR, OUTPUT, get_current_time
from feature_engineer import featurize
from train import DummyClassifier, NoScaler, DummyScaler

class MRP7Pred(object):
    def __init__(
        self,
        clf_dir: str=f"{MODEL_DIR}/best_model_20201224-082918.pkl",
        train: bool=False
    ) -> None:
        """
        Parameters
        --------
        clf_dir: str
        train: bool
        """
        
        print("Loading trained model ... ", end="", flush=True)
        with open(clf_dir, "rb") as ci:
            self.clf = pickle.load(ci)
        print("Done!")
        
        
    def predict(self, compound_csv_dir: str) -> DataFrame:
        """
        Make predictions
        
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
            raise ValueError("The input csv should have these two columns: [\"name\", \"smiles\"]")
        
        df = df_all[["name", "smiles"]]
        
        print("Generating features ... ")
        df_feat = featurize(df)
        print("Done!")
        
        print("Start predicting ...", end="", flush=True)
        feats = df_feat.iloc[:, 2:] # features start from 3rd column
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
        print("Done!")
        return df_out
    
    
def main() -> None:
    m7p = MRP7Pred()
    m7p.predict("./data/unknown.csv")
    

if __name__ == "__main__":
    main()