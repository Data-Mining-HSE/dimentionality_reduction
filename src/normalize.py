import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as _normalize


def normalize(dataset: pd.DataFrame) -> pd.DataFrame:
    normalized_matrix = _normalize(dataset, norm='l2', axis=0) 
    dataset = pd.DataFrame(normalized_matrix, columns=dataset.columns)
    return dataset


def check_norm(dataset: pd.DataFrame) -> pd.DataFrame:
    data = []
    for column in dataset.columns:
        data.append(round(np.linalg.norm(dataset[column], ord=2), 1))
    return pd.DataFrame({'Column_name': dataset.columns, 'Euclidean norm': data})
