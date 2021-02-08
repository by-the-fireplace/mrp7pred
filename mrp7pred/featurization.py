"""
Generate selected features

Input: cleaned dataframe with columns:
    name
    smiles
    label
Output: featurized dataframe

Number of features: 828

feature selection details see notebook/feature_engineering.ipynb
"""

__author__ = "Jingquan Wang"
__email__ = "jq.wang1214@gmail.com"

from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from mrp7pred.feats.gen_all_features import featurize
from mrp7pred.utils import DATA


# def featurize(df: DataFrame) -> DataFrame:
#     """
#     Generate featurized dataframe
#     df must have columns named "name" and "smiles"
#     """
#     # add new feature columns
#     df = df.reindex(columns=df.columns.tolist() + feature_list).reset_index()

#     # populate features
#     for index, row in tqdm(enumerate(df.itertuples()), total=len(df)):
#         try:
#             feats = _rdk_features(getattr(row, "smiles"))
#         except Exception as e:
#             print(f"Error found when processing compound {getattr(row, 'name')}")
#             print(f"Smiles: {getattr(row, 'smiles')}")
#             print(e)
#             continue
#         df.loc[index, "atomic_mass_high":"MQN42"] = feats

#     input_len = len(df)
#     df = df.dropna()
#     output_len = len(df)
#     if input_len != output_len:
#         print(f"Dropped {input_len-output_len} unfeaturizable compounds.")
#     return df


def featurize_unknown():
    df_unknown = pd.read_csv(f"{DATA}/unknown.csv")[["nane", "smiles"]]
    df_feats = featurize(df_unknown, prefix="featurized_unknown_")


if __name__ == "__main__":
    featurize_unknown()
