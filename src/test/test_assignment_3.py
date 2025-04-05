import numpy as np
from src.main.assignment_3 import (
    gaussian_elimination,
    lu_decomposition,
    is_diagonally_dominant,
    is_positive_definite
)

def test_gaussian_elimination():
    matrix = [
        [3, 2, -4, 3],
        [2, 3, 3, 15],
        [5, -3, 1, 14]
    ]
    expected = [2.0, -1.0, 1.0]
    result = gaussian_elimination([row[:] for row in matrix])
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6

def test_lu_decomposition():
    matrix = [
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ]
    L_expected = [
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [3.0, 4.0, 1.0, 0.0],
        [-1.0, -3.0, 0.0, 1.0]
    ]
    U_expected = [
        [1.0, 1.0, 0.0, 3.0],
        [0.0, -1.0, -1.0, -5.0],
        [0.0, 0.0, 3.0, 13.0],
        [0.0, 0.0, 0.0, -13.0]
    ]
    L, U = lu_decomposition(matrix)
    assert np.allclose(L, L_expected, atol=1e-6)
    assert np.allclose(U, U_expected, atol=1e-6)
    assert abs(np.prod(np.diag(U)) - 39.0) < 1e-6

def test_diagonal_dominance():
    matrix = [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ]
    assert is_diagonally_dominant(matrix) is True

def test_positive_definiteness():
    matrix = [
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ]
    assert is_positive_definite(matrix) is True
