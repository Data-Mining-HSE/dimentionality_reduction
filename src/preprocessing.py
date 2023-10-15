from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.constants import DATASET_COLUMNS


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path, header=None, names=DATASET_COLUMNS)


def fill_empty_values(dataset: pd.DataFrame) -> pd.DataFrame:
    for column in dataset.columns:
        fill_value = column.mean()
        dataset[column] = dataset[column].fillna(fill_value)
    return dataset


def get_statistics(dataset: pd.DataFrame) -> pd.DataFrame:
    statistics = []
    statistics.append(['Mean', *list(dataset.mean())])
    statistics.append(['Median', *list(dataset.median())])
    statistics.append(['Std', *list(dataset.std())])
    statistics.append(['Quantile (25%)', *list(dataset.quantile(0.25))])
    statistics.append(['Quantile (50%)', *list(dataset.quantile(0.5))])
    statistics.append(['Quantile (75%)', *list(dataset.quantile(0.75))])
    return pd.DataFrame(statistics, columns=['Statistic', *dataset.columns])


def get_histograms(dataset: pd.DataFrame) -> plt.Figure:
    dataset.hist(figsize=(8, 8))

