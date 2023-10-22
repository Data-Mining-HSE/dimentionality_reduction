import numpy as np
import pandas as pd
from numpy import float32
from numpy.typing import NDArray


def svd(dataset: pd.DataFrame) -> pd.DataFrame:
    matrix_u, singular_values, matrix_v = np.linalg.svd(dataset,
                                                  full_matrices=False, 
                                                  compute_uv=True)
    return matrix_u, singular_values, matrix_v


def get_variance(singular_values: NDArray[float32]) -> float32:
    sqr_singular_values = singular_values ** 2
    variance = np.sum(sqr_singular_values)
    return variance


def get_explained_variances(singular_values: NDArray[float32]) -> NDArray[float32]:
    variance = get_variance(singular_values)
    explained_variances = singular_values ** 2 / variance
    return explained_variances


def get_svd_info(singular_values: NDArray[float32]) -> pd.DataFrame:
    explained_variances = get_explained_variances(singular_values)
    data = {
        'Singular values': [np.around(singular_values, 2)],
        'Total variance': [np.around(get_variance(singular_values), 2)],
        'Explained variances': [np.around(explained_variances, 2)],
    }

    for num_singular_values in range(1, len(singular_values)):

        data[f'Explained variance (via {num_singular_values} components)'] = [
                    np.around(np.sum(explained_variances[:num_singular_values]), 2)]
    return pd.DataFrame(data).T


def covert(matrix_u: NDArray[float32], singular_values: NDArray[float32], matrix_v: NDArray[float32],
           precision: int) ->  NDArray[float32]:
    singular_values_matrix = np.diag(singular_values)
    matrix_z = matrix_u.dot(singular_values_matrix)[:,:precision].dot(matrix_v[:precision, :])
    return matrix_z



def get_errors(original_matrix: NDArray[float32], approximated_matrix: NDArray[float32]) -> pd.DataFrame:

    errors = original_matrix - approximated_matrix

    error_frobenius_norm = np.linalg.norm(errors, ord='fro') ** 2
    error_euclidean_norm = np.linalg.norm(errors, ord=2)

    relative_error_frobenius_norm = error_frobenius_norm / np.linalg.norm(original_matrix, ord='fro') ** 2

    data = [
        ('Frobenius norm error', error_frobenius_norm),
        ('Spectral norm error', error_euclidean_norm),
        ('Relative Frobenius norm error', relative_error_frobenius_norm),
    ]

    return pd.DataFrame(data, columns=['Error', 'Result'])
