"""
Core linear algebra functionality.
"""

import numpy as np
from typing import List, Union, Tuple, Dict, Optional

class Vector:
    """
    A class representing a mathematical vector.
    """
    
    def __init__(self, data: Union[List[float], np.ndarray]):
        """
        Initialize a vector with given data.
        
        Args:
            data: List or numpy array of values
        """
        if isinstance(data, list):
            self.data = np.array(data, dtype=float)
        else:
            self.data = data.astype(float)
        
        self.size = len(self.data)
    
    def __repr__(self):
        """Formal string representation of the vector."""
        # Convert numpy array to list with proper formatting
        data_list = [float(val) for val in self.data]
        return f"Vector({data_list})"
    
    def __str__(self):
        """String representation of the vector."""
        return str(self.data)  # Just return the numpy array string representation
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        if self.size != other.size:
            raise ValueError("Vectors must have the same dimension for addition")
        return Vector(self.data + other.data)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors."""
        if self.size != other.size:
            raise ValueError("Vectors must have the same dimension for subtraction")
        return Vector(self.data - other.data)
    
    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply vector by scalar."""
        return Vector(self.data * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        """Multiply scalar by vector."""
        return self.__mul__(scalar)
    
    def dot(self, other: 'Vector') -> float:
        """Compute dot product with another vector."""
        if self.size != other.size:
            raise ValueError("Vectors must have the same dimension for dot product")
        return float(np.dot(self.data, other.data))
    
    def norm(self) -> float:
        """Compute the Euclidean norm (magnitude) of the vector."""
        return float(np.linalg.norm(self.data))
    
    def normalize(self) -> 'Vector':
        """Return a normalized (unit) vector."""
        norm_value = self.norm()
        if norm_value == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector(self.data / norm_value)

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Compute the cross product with another vector.
        
        Args:
            other (Vector): Another vector
        
        Returns:
            Vector: Cross product vector
        
        Raises:
            ValueError: If either vector is not 3D
        """
        if self.size != 3 or other.size != 3:
            raise ValueError("Cross product is only defined for 3D vectors")
        
        result = np.cross(self.data, other.data)
        return Vector(result)

    def angle(self, other: 'Vector') -> float:
        """
        Compute the angle between this vector and another vector.
        
        Args:
            other (Vector): Another vector
        
        Returns:
            float: Angle in radians
        
        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.size != other.size:
            raise ValueError("Vectors must have the same dimension")
        
        dot_product = self.dot(other)
        norm_product = self.norm() * other.norm()
        
        # Handle numerical issues
        cos_angle = min(1.0, max(-1.0, dot_product / norm_product))
        
        return np.arccos(cos_angle)

    def project(self, other: 'Vector') -> 'Vector':
        """
        Project this vector onto another vector.
        
        Args:
            other (Vector): Vector to project onto
        
        Returns:
            Vector: Projection vector
        
        Raises:
            ValueError: If vectors have different dimensions
        """
        if self.size != other.size:
            raise ValueError("Vectors must have the same dimension")
        
        # Projection formula: (v·w / |w|²) * w
        scalar = self.dot(other) / other.dot(other)
        return scalar * other


class Matrix:
    """
    A class representing a mathematical matrix.
    """
    
    def __init__(self, data: Union[List[List[float]], np.ndarray]):
        """
        Initialize a matrix with given data.
        
        Args:
            data: 2D list or numpy array of values
        """
        if isinstance(data, list):
            self.data = np.array(data, dtype=float)
        else:
            self.data = data.astype(float)
        
        self.shape = self.data.shape
        self.rows, self.cols = self.shape
    
    def __repr__(self):
        """Formal string representation of the matrix."""
        # Convert numpy array to list of lists with proper formatting
        data_list = [[float(val) for val in row] for row in self.data]
        return f"Matrix({data_list})"
    
    def __str__(self):
        """String representation of the matrix."""
        return str(self.data)  # Just return the numpy array string representation
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices."""
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape for addition")
        return Matrix(self.data + other.data)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices."""
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape for subtraction")
        return Matrix(self.data - other.data)
    
    def __mul__(self, other: Union['Matrix', Vector, float, int]) -> Union['Matrix', Vector]:
        """Multiply matrix with another matrix, vector, or scalar."""
        if isinstance(other, (int, float)):
            return Matrix(self.data * other)
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Matrix dimensions incompatible for multiplication: {self.shape} and {other.shape}")
            return Matrix(np.matmul(self.data, other.data))
        elif isinstance(other, Vector):
            if self.cols != other.size:
                raise ValueError(f"Matrix and vector dimensions incompatible for multiplication: {self.shape} and {other.size}")
            result = np.matmul(self.data, other.data)
            return Vector(result)
        else:
            raise TypeError("Unsupported operand type for *")
    
    def __rmul__(self, scalar: float) -> 'Matrix':
        """Multiply scalar by matrix."""
        if isinstance(scalar, (int, float)):
            return Matrix(self.data * scalar)
        else:
            raise TypeError("Unsupported operand type")
    
    def transpose(self) -> 'Matrix':
        """Return the transpose of the matrix."""
        return Matrix(self.data.T)
    
    def determinant(self) -> float:
        """Calculate the determinant of the matrix."""
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices")
        return float(np.linalg.det(self.data))
    
    def inverse(self) -> 'Matrix':
        """Calculate the inverse of the matrix."""
        if self.rows != self.cols:
            raise ValueError("Inverse can only be calculated for square matrices")
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular, cannot compute inverse")
        return Matrix(np.linalg.inv(self.data))
    
    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        """Create an identity matrix of given size."""
        return cls(np.eye(size))
    
    @classmethod
    def from_coefficients(cls, coeffs: List[Dict[str, float]], variables: List[str]) -> Tuple['Matrix', Vector]:
        """
        Create a coefficient matrix and constant vector from a list of coefficient dictionaries.
        
        Args:
            coeffs: List of coefficient dictionaries from parsed equations
            variables: List of variable names
            
        Returns:
            Tuple of coefficient Matrix and constant Vector
        """
        n = len(coeffs)  # Number of equations
        m = len(variables)  # Number of variables
        
        A_data = np.zeros((n, m))
        b_data = np.zeros(n)
        
        for i, coeff_dict in enumerate(coeffs):
            for j, var in enumerate(variables):
                A_data[i, j] = coeff_dict.get(var, 0)
            b_data[i] = coeff_dict.get("constant", 0)
        
        return cls(A_data), Vector(b_data) 