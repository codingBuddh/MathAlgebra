"""
Integration tests for the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.utils import parse_system
from linearalgebra.core import Matrix, Vector
from linearalgebra.solvers import solve_linear_system, gaussian_elimination
from linearalgebra.decompositions import lu_decomposition, qr_decomposition, svd_decomposition

def test_end_to_end_equation_solving():
    # Define a system of equations
    equations = [
        "2x + y - z = 5",
        "-3x + y + 2z = -5",
        "x + y + z = 3"
    ]
    
    # Solve using our library
    solution = solve_linear_system(equations)
    
    # Expected solution: x=2, y=1, z=0
    assert pytest.approx(solution["x"], 0.001) == 2.0
    assert pytest.approx(solution["y"], 0.001) == 1.0
    assert pytest.approx(solution["z"], 0.001) == 0.0
    
    # Verify the solution directly
    x, y, z = solution["x"], solution["y"], solution["z"]
    
    # Check each equation
    assert pytest.approx(2*x + y - z, 0.001) == 5.0
    assert pytest.approx(-3*x + y + 2*z, 0.001) == -5.0
    assert pytest.approx(x + y + z, 0.001) == 3.0

def test_matrix_decomposition_workflow():
    # Create a matrix
    A = Matrix([
        [4, 2, 1],
        [16, 4, 1],
        [64, 8, 1]
    ])
    
    # Perform LU decomposition
    L, U = lu_decomposition(A)
    
    # Verify A = L * U
    LU = L * U
    assert np.allclose(A.data, LU.data)
    
    # Perform QR decomposition
    Q, R = qr_decomposition(A)
    
    # Verify A = Q * R
    QR = Q * R
    assert np.allclose(A.data, QR.data)
    
    # Solve a linear system using the decomposition
    b = Vector([1, 2, 3])
    
    # Using LU decomposition to solve Ax = b
    # First solve Ly = b for y
    y_data = np.zeros(3)
    for i in range(3):
        y_data[i] = b.data[i]
        for j in range(i):
            y_data[i] -= L.data[i, j] * y_data[j]
    
    y = Vector(y_data)
    
    # Then solve Ux = y for x
    x_data = np.zeros(3)
    for i in range(2, -1, -1):
        x_data[i] = y.data[i]
        for j in range(i+1, 3):
            x_data[i] -= U.data[i, j] * x_data[j]
        x_data[i] /= U.data[i, i]
    
    x_lu = Vector(x_data)
    
    # Compare with direct solution
    x_direct = A.inverse() * b
    
    assert np.allclose(x_lu.data, x_direct.data)

def test_combined_operations():
    # Create matrices and vectors
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    v = Vector([1, 2])
    
    # Perform a series of operations
    C = A * B + 2 * A - B
    w = A * v + 3 * v
    
    # Verify results manually
    expected_C_data = np.array([[19, 22], [43, 50]]) + 2 * np.array([[1, 2], [3, 4]]) - np.array([[5, 6], [7, 8]])
    expected_w_data = np.array([5, 11]) + 3 * np.array([1, 2])
    
    assert np.allclose(C.data, expected_C_data)
    assert np.allclose(w.data, expected_w_data)

def test_svd_reconstruction():
    # Create a matrix with known rank deficiency
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],  # This row is 2 * first row
        [3, 6, 9]   # This row is 3 * first row
    ])
    
    # Perform SVD
    U, S, VT = svd_decomposition(A)
    
    # Reconstruct A from its SVD
    A_reconstructed = U * S * VT
    
    # Verify reconstruction
    assert np.allclose(A.data, A_reconstructed.data)
    
    # This matrix has rank 1, so zeroing out small singular values should reduce rank
    S_approx = Matrix(S.data.copy())
    s_values = np.diag(S.data)
    threshold = max(s_values) * 0.1  # Use relative threshold
    
    for i in range(len(s_values)):
        if s_values[i] < threshold:
            S_approx.data[i, i] = 0
    
    # Reconstruct using the approximation
    A_approx = U * S_approx * VT
    
    # Verify rank is at most 1
    assert np.linalg.matrix_rank(A_approx.data) <= 1 