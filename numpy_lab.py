import numpy as np

arr_numpy = np.array([1, 2, 3, 4])
print(arr_numpy)

arr_one = np.ones((3,4))
arr_zero = np.zeros((4,3))
print("Array of Ones:\n", arr_one)
print("Array of Zeros:\n", arr_zero)

A = np.array([[2, 7, 3],[5, 9, 1]])
B = np.array([[6, 4, 3, 7],[3, 5, 7, 9],[8, 4, 2, 1]])
print("Array A:\n", A)
print("Array B:\n", B)
ans = A @ B
print("Matrix Multiplication of A and B:\n", ans)

matrix = np.array([[3, 1],[1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)


