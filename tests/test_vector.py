"""
Tests for the Vector class in the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.core import Vector

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

def test_vector_cross_product():
    v1 = Vector([1, 0, 0])
    v2 = Vector([0, 1, 0])
    result = v1.cross(v2)
    assert np.array_equal(result.data, np.array([0.0, 0.0, 1.0]))
    
    # Test error for non-3D vectors
    v3 = Vector([1, 2])
    with pytest.raises(ValueError):
        v1.cross(v3)
    
    with pytest.raises(ValueError):
        v3.cross(v1)

def test_vector_angle():
    v1 = Vector([1, 0, 0])
    v2 = Vector([0, 1, 0])
    angle = v1.angle(v2)
    assert pytest.approx(angle, 0.001) == np.pi/2
    
    v3 = Vector([1, 1, 0])
    angle2 = v1.angle(v3)
    assert pytest.approx(angle2, 0.001) == np.pi/4
    
    # Test error for incompatible dimensions
    v4 = Vector([1, 2])
    with pytest.raises(ValueError):
        v1.angle(v4)

def test_vector_projection():
    v1 = Vector([3, 4])
    v2 = Vector([1, 0])
    proj = v1.project(v2)
    assert np.array_equal(proj.data, np.array([3.0, 0.0]))
    
    # Test error for incompatible dimensions
    v3 = Vector([1, 2, 3])
    with pytest.raises(ValueError):
        v1.project(v3) 