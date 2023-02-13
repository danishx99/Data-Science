
import numpy as np


def LUSolve(A, b):

    # augmented matrix
    Ab = A.copy()
    Ab[:, 0] = b

    # infinite solutions
    if (np.linalg.matrix_rank(A) == np.linalg.matrix_rank(Ab) and np.linalg.matrix_rank(A) < A.shape[0]):
        return 1

    # no solution
    elif(np.linalg.matrix_rank(A) != np.linalg.matrix_rank(Ab)):
        return 0

    # unique solution
    else:

        n, m = A.shape
        P = np.identity(n)
        L = np.identity(n)
        U = A.copy()
        Pi = np.identity(n)
        Li = np.zeros((n, n))
        for k in range(0, n - 1):
            index = np.argmax(abs(U[k:, k]))
            index = index + k
            if index != k:
                P = np.identity(n)
                P[[index, k], k:n] = P[[k, index], k:n]
                U[[index, k], k:n] = U[[k, index], k:n]
                Pi = np.dot(P, Pi)
                Li = np.dot(P, Li)
            L = np.identity(n)
            for j in range(k+1, n):
                L[j, k] = -(U[j, k] / U[k, k])
                Li[j, k] = (U[j, k] / U[k, k])
            U = np.dot(L, U)
        np.fill_diagonal(Li, 1)

        c = np.dot(Pi, b)

        x = solve(Li, U, c)

        return x, Li, U


def backSub(U, b):
    U = U.astype(float)
    b = b.astype(float)
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        temp = b[i]
        for j in range(i+1, n):
            temp -= U[i, j] * x[j]
        x[i] = temp / U[i, i]
    return x


def forwardSub(L, b):

    L = L.astype(float)
    b = b.astype(float)
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        temp = b[i]
        for j in range(i):
            temp -= L[i, j] * x[j]
        x[i] = temp / L[i, i]
    return x


def solve(L, U, b):
    U = U.astype(float)
    L = L.astype(float)
    b = b.astype(float)

    y = forwardSub(L, b)
    x = backSub(U, y)
    return x


A = np.array([[1, -1, 0],
              [2, 2, 3],
              [-1, 3, 2]])
b = np.array([2, -1, 4])

print(LUSolve(A, b))
