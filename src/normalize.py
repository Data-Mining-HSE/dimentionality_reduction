import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import normalize as _normalize


def normalize(dataset: pd.DataFrame) -> pd.DataFrame:
    normalized_matrix = _normalize(dataset, norm='l2', axis=0) 
    dataset = pd.DataFrame(normalized_matrix, columns=dataset.columns)
    return dataset


def normalize(dataset: pd.DataFrame) -> pd.DataFrame:
    for column in dataset.columns:
        euclidean_column_norm = np.sqrt(dataset[column].abs().pow(2).sum())
        dataset[column] /= euclidean_column_norm
    return dataset


def centering(dataset: pd.DataFrame) -> pd.DataFrame:
    for column in dataset.columns:
        dataset[column] -= dataset[column].mean()
    return dataset


def check_norm(dataset: pd.DataFrame) -> pd.DataFrame:
    data = []
    for column in dataset.columns:
        data.append(round(np.linalg.norm(dataset[column], ord=2), 1))
    return pd.DataFrame({'Column_name': dataset.columns, 'Euclidean norm': data})
