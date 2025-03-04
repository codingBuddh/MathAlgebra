"""
Tests for the Matrix class in the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.core import Matrix, Vector

def test_matrix_creation():
    # Create from list
    m1 = Matrix([[1, 2], [3, 4]])
    assert m1.shape == (2, 2)
    assert np.array_equal(m1.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    # Create from numpy array
    m2 = Matrix(np.array([[5, 6], [7, 8]]))
    assert m2.shape == (2, 2)
    assert np.array_equal(m2.data, np.array([[5.0, 6.0], [7.0, 8.0]]))

def test_matrix_addition():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    result = m1 + m2
    assert np.array_equal(result.data, np.array([[6.0, 8.0], [10.0, 12.0]]))
    
    # Test error for incompatible dimensions
    m3 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m1 + m3

def test_matrix_subtraction():
    m1 = Matrix([[5, 6], [7, 8]])
    m2 = Matrix([[1, 2], [3, 4]])
    result = m1 - m2
    assert np.array_equal(result.data, np.array([[4.0, 4.0], [4.0, 4.0]]))
    
    # Test error for incompatible dimensions
    m3 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m1 - m3

def test_matrix_scalar_multiplication():
    m = Matrix([[1, 2], [3, 4]])
    
    # Matrix * scalar
    result1 = m * 2
    assert np.array_equal(result1.data, np.array([[2.0, 4.0], [6.0, 8.0]]))
    
    # scalar * Matrix
    result2 = 3 * m
    assert np.array_equal(result2.data, np.array([[3.0, 6.0], [9.0, 12.0]]))

def test_matrix_matrix_multiplication():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    result = m1 * m2
    assert np.array_equal(result.data, np.array([[19.0, 22.0], [43.0, 50.0]]))
    
    # Test error for incompatible dimensions
    m3 = Matrix([[1, 2], [3, 4]])  # 2x2 matrix
    m4 = Matrix([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
    with pytest.raises(ValueError):
        m3 * m4  # This should raise an error (2x2 * 3x2 is incompatible)

def test_matrix_vector_multiplication():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 2, 3])
    result = m * v
    assert np.array_equal(result.data, np.array([14.0, 32.0]))
    
    # Test error for incompatible dimensions
    v2 = Vector([1, 2])
    with pytest.raises(ValueError):
        m * v2

def test_matrix_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    result = m.transpose()
    assert result.shape == (3, 2)
    assert np.array_equal(result.data, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))

def test_matrix_determinant():
    m = Matrix([[1, 2], [3, 4]])
    assert pytest.approx(m.determinant(), 0.001) == -2.0
    
    # Test error for non-square matrix
    m2 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m2.determinant()

def test_matrix_inverse():
    m = Matrix([[1, 2], [3, 4]])
    inv = m.inverse()
    expected = np.array([[-2.0, 1.0], [1.5, -0.5]])
    assert np.allclose(inv.data, expected)
    
    # Test error for non-square matrix
    m2 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m2.inverse()
    
    # Test error for singular matrix
    m3 = Matrix([[1, 2], [2, 4]])  # Linearly dependent rows
    with pytest.raises(ValueError):
        m3.inverse()

def test_matrix_identity():
    m = Matrix.identity(3)
    assert m.shape == (3, 3)
    assert np.array_equal(m.data, np.eye(3))

def test_matrix_from_coefficients():
    coeffs = [
        {"x": 3, "y": 4, "z": -5, "constant": 12},
        {"x": 2, "y": -1, "z": 3, "constant": 7},
        {"x": 1, "y": 1, "z": 1, "constant": 6}
    ]
    variables = ["x", "y", "z"]
    
    A, b = Matrix.from_coefficients(coeffs, variables)
    
    expected_A = np.array([
        [3.0, 4.0, -5.0],
        [2.0, -1.0, 3.0],
        [1.0, 1.0, 1.0]
    ])
    expected_b = np.array([12.0, 7.0, 6.0])
    
    assert np.array_equal(A.data, expected_A)
    assert np.array_equal(b.data, expected_b) 