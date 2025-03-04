"""
Tests for the core module (Matrix and Vector classes) in the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.core import Matrix, Vector

# ===== Matrix Tests =====

def test_matrix_creation():
    # Create from list
    m1 = Matrix([[1, 2], [3, 4]])
    assert m1.shape == (2, 2)
    assert np.array_equal(m1.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    # Create from numpy array
    m2 = Matrix(np.array([[5, 6], [7, 8]]))
    assert m2.shape == (2, 2)
    assert np.array_equal(m2.data, np.array([[5.0, 6.0], [7.0, 8.0]]))
    
    # Test properties
    assert m1.rows == 2
    assert m1.cols == 2

def test_matrix_string_representation():
    m = Matrix([[1, 2], [3, 4]])
    assert str(m) == "[[1. 2.]\n [3. 4.]]"
    assert repr(m) == "Matrix([[1.0, 2.0], [3.0, 4.0]])"

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
    m1 = Matrix([[1, 2], [3, 4]])  # 2x2 matrix
    m2 = Matrix([[5, 6], [7, 8]])  # 2x2 matrix
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
    # 2x2 matrix
    m1 = Matrix([[1, 2], [3, 4]])
    assert pytest.approx(m1.determinant(), 0.001) == -2.0
    
    # 3x3 matrix
    m2 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert pytest.approx(m2.determinant(), 0.001) == 0.0  # Singular matrix
    
    # 3x3 non-singular matrix
    m3 = Matrix([[2, 3, 4], [1, 2, 3], [4, 5, 7]])
    assert pytest.approx(m3.determinant(), 0.001) == 1.0
    
    # Test error for non-square matrix
    m4 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m4.determinant()

def test_matrix_inverse():
    # 2x2 matrix
    m1 = Matrix([[1, 2], [3, 4]])
    inv1 = m1.inverse()
    expected1 = np.array([[-2.0, 1.0], [1.5, -0.5]])
    assert np.allclose(inv1.data, expected1)
    
    # 3x3 matrix
    m2 = Matrix([[2, 3, 4], [1, 2, 3], [4, 5, 7]])
    inv2 = m2.inverse()
    # Verify A * A^-1 = I
    product = m2 * inv2
    assert np.allclose(product.data, np.eye(3))
    
    # Test error for non-square matrix
    m3 = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        m3.inverse()
    
    # Test error for singular matrix
    m4 = Matrix([[1, 2], [2, 4]])  # Linearly dependent rows
    with pytest.raises(ValueError):
        m4.inverse()

def test_matrix_identity():
    # 2x2 identity
    m1 = Matrix.identity(2)
    assert m1.shape == (2, 2)
    assert np.array_equal(m1.data, np.eye(2))
    
    # 3x3 identity
    m2 = Matrix.identity(3)
    assert m2.shape == (3, 3)
    assert np.array_equal(m2.data, np.eye(3))
    
    # Test with invalid size
    with pytest.raises(ValueError):
        Matrix.identity(-1)

def test_matrix_from_coefficients():
    # Basic test
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
    
    # Test with missing variables
    coeffs2 = [
        {"x": 3, "z": -5, "constant": 12},
        {"y": -1, "z": 3, "constant": 7},
        {"x": 1, "y": 1, "constant": 6}
    ]
    
    A2, b2 = Matrix.from_coefficients(coeffs2, variables)
    
    expected_A2 = np.array([
        [3.0, 0.0, -5.0],
        [0.0, -1.0, 3.0],
        [1.0, 1.0, 0.0]
    ])
    expected_b2 = np.array([12.0, 7.0, 6.0])
    
    assert np.array_equal(A2.data, expected_A2)
    assert np.array_equal(b2.data, expected_b2)

# ===== Vector Tests =====

def test_vector_creation():
    # Create from list
    v1 = Vector([1, 2, 3])
    assert v1.size == 3
    assert np.array_equal(v1.data, np.array([1.0, 2.0, 3.0]))
    
    # Create from numpy array
    v2 = Vector(np.array([4, 5, 6]))
    assert v2.size == 3
    assert np.array_equal(v2.data, np.array([4.0, 5.0, 6.0]))

def test_vector_string_representation():
    v = Vector([1, 2, 3])
    assert str(v) == "[1. 2. 3.]"
    assert repr(v) == "Vector([1.0, 2.0, 3.0])"

def test_vector_addition():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    result = v1 + v2
    assert np.array_equal(result.data, np.array([5.0, 7.0, 9.0]))
    
    # Test error for incompatible dimensions
    v3 = Vector([1, 2])
    with pytest.raises(ValueError):
        v1 + v3

def test_vector_subtraction():
    v1 = Vector([4, 5, 6])
    v2 = Vector([1, 2, 3])
    result = v1 - v2
    assert np.array_equal(result.data, np.array([3.0, 3.0, 3.0]))
    
    # Test error for incompatible dimensions
    v3 = Vector([1, 2])
    with pytest.raises(ValueError):
        v1 - v3

def test_vector_scalar_multiplication():
    v = Vector([1, 2, 3])
    
    # Vector * scalar
    result1 = v * 2
    assert np.array_equal(result1.data, np.array([2.0, 4.0, 6.0]))
    
    # scalar * Vector
    result2 = 3 * v
    assert np.array_equal(result2.data, np.array([3.0, 6.0, 9.0]))

def test_vector_dot_product():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    result = v1.dot(v2)
    assert pytest.approx(result, 0.001) == 32.0
    
    # Test error for incompatible dimensions
    v3 = Vector([1, 2])
    with pytest.raises(ValueError):
        v1.dot(v3)

def test_vector_norm():
    v = Vector([3, 4])
    assert pytest.approx(v.norm(), 0.001) == 5.0
    
    v2 = Vector([1, 1, 1])
    assert pytest.approx(v2.norm(), 0.001) == np.sqrt(3)

def test_vector_normalize():
    v = Vector([3, 4])
    normalized = v.normalize()
    assert pytest.approx(normalized.data[0], 0.001) == 0.6
    assert pytest.approx(normalized.data[1], 0.001) == 0.8
    assert pytest.approx(normalized.norm(), 0.001) == 1.0
    
    # Test zero vector
    v2 = Vector([0, 0, 0])
    with pytest.raises(ValueError):
        v2.normalize() 