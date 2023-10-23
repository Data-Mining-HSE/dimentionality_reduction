import numpy as np
from numpy import float32
from numpy.typing import NDArray


def minkowski_distance(point_a: NDArray[float32], point_b: NDArray[float32], p: float) -> NDArray[float32]:
    return np.power(np.sum(np.power(np.abs(point_a - point_b), p)), 1/p)


def pairwise_minkowski_distance(matrix: NDArray[float32], p: float) -> NDArray[float32]:
    matrix_size = matrix.shape[0]
    distances = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            distances[i,j] = minkowski_distance(matrix[i], matrix[j], p)
    return distances
