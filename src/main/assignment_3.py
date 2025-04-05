# src/main/assignment_3.py

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
    
if __name__ == "__main__":
    main()
