from datetime import datetime
import string
import random
import os
import pandas as pd
from pandas import DataFrame
from mrp7pred.mrp7pred import MRP7Pred

## uploading specs ##
UPLOAD_FOLDER = "./data/"


def ensure_folder(path: str) -> None:
    if not os.path.isdir(path):
        os.mkdir(path)


def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d")


def random_string(N: int) -> str:
    return "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(N)
    )


def get_predictions(df: DataFrame, clf_dir: str):
    m7p = MRP7Pred(clf_dir=clf_dir)
    out = m7p.predict(prefix=f"{get_current_time()}")
    out = out.sort_values(by=["score"], ascending=False)
    return out
