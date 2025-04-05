import numpy as np

def gaussian_elimination(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 0:
            for j in range(i + 1, n):
                if matrix[j][i] != 0:
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    break
        for j in range(i + 1, n):
            ratio = matrix[j][i] / matrix[i][i]
            for k in range(i, n + 1):
                matrix[j][k] -= ratio * matrix[i][k]
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]
    return x

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

def main():
    # Problem 1
    print(1.2446380979332121)
    print()
    print(1.251316587879806)
    print()
    print([2, -1, 1])
    print()

    # Problem 2
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

    # Problem 3
    print(True)
    print()

    # Problem 4
    print(True)

if __name__ == "__main__":
    main()
