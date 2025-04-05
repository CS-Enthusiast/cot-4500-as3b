# src/main/assignment_3.py
import numpy as np

def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix)) if i != j)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True


def lu_decomposition(matrix):
    n = len(matrix)
    L = np.identity(n)
    U = np.array(matrix, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] -= factor * U[i]

    return L, U

def gaussian_elimination(matrix):
    n = len(matrix)

    # Forward elimination
    for i in range(n):
        # Make the diagonal contain all non-zero values
        if matrix[i][i] == 0:
            for j in range(i + 1, n):
                if matrix[j][i] != 0:
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    break
        for j in range(i + 1, n):
            ratio = matrix[j][i] / matrix[i][i]
            for k in range(i, n + 1):
                matrix[j][k] -= ratio * matrix[i][k]

    # Backward substitution
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]

    return x


def main():
    # Problem 1: Gaussian Elimination
    matrix = [
        [3, 2, -4, 3],
        [2, 3, 3, 15],
        [5, -3, 1, 14]
    ]
    result = gaussian_elimination(matrix)
    print()
    print(result)
    print()

    # Problem 2: LU Factorization
    A = [
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ]
    L, U = lu_decomposition(A)
    determinant = np.prod(np.diag(U))

    print(determinant)
    print()
    print(L.tolist())
    print()
    print(U.tolist())
    print()

    # Problem 3: Diagonal Dominance
    diag_matrix = [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ]
    print(is_diagonally_dominant(diag_matrix))
    print()

if __name__ == "__main__":
    main()
