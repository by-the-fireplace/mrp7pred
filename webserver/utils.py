from datetime import datetime
import string
import random
import os
import pandas as pd
from pandas import DataFrame

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


def predict(df: DataFrame):
    return