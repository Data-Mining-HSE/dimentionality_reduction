from pathlib import Path

import pandas as pd

from src.constants import DATASET_COLUMNS


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path, header=None, names=DATASET_COLUMNS)
