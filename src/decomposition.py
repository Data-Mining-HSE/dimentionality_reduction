import numpy as np
import pandas as pd
from numpy import float32
from numpy.typing import NDArray
from sklearn.manifold import MDS


def svd(dataset: pd.DataFrame) -> tuple[NDArray[float32], NDArray[float32], NDArray[float32]]:
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


def low_rank_approx(matrix_u: NDArray[float32], singular_values: NDArray[float32],
                    matrix_v: NDArray[float32], precision: int) ->  NDArray[float32]:

    approximated_matrix = np.zeros((len(matrix_u), len(matrix_v)))
    for precision_id in range(precision):
        approximated_matrix += singular_values[precision_id] * np.outer(matrix_u.T[precision_id],
                                                                        matrix_v[precision_id])
    return approximated_matrix


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


def get_coefficient_matrix(matrix_z: NDArray[float32], singular_values: NDArray[float32],
                           matrix_v: NDArray[float32], decomposition_rank: int) -> NDArray[float32]:
    coefficient_matrix = []
    for j in range(matrix_z.shape[1]):
        row_coefficients = []
        for i in range(decomposition_rank):
            row_coefficients.append(singular_values[i] * matrix_v[j, i])
        coefficient_matrix.append(row_coefficients)
    return np.array(coefficient_matrix).T



def multidimensional_scaling(dataset: NDArray[float32], verbose: bool = False):
    mds = MDS(n_components=2, max_iter=300, dissimilarity='euclidean', normalized_stress='auto')
    decomposed_data = mds.fit_transform(dataset)
    if verbose:
        print(f'Stress value: {round(mds.stress_, 2)}')
    return decomposed_data

def get_gram_column_matrix(matrix_z: NDArray[float32]) -> NDArray[float32]:
    return matrix_z.T @ matrix_z

def is_nonegative_define(gram_matrix: NDArray[float32]) -> NDArray[float32]:
    return np.all(np.linalg.eigvals(gram_matrix) >= 0)

def get_diagonal_eigenvalues_matrix(gram_matrix: NDArray[float32]) -> NDArray[float32]:
    if not is_nonegative_define(gram_matrix):
        print("Gram matrix are negative define. So it is impossible to put our space to Euclidean")
        return np.zeros_like(gram_matrix.shape)
    return np.diag(np.linalg.eigvals(gram_matrix))

def get_embbeding_spaces_error(diagonal_matrix: NDArray[float32]) -> float32:
    return np.sum(diagonal_matrix.diagonal()[2:7] ** 2)

