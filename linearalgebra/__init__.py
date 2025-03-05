"""
MathAlgebra Linear Algebra Library
"""

from .core import Matrix, Vector
from .matrix_ops import (
    naive_multiply,
    strassen_multiply,
    block_multiply,
    sparse_multiply,
    smart_multiply
)

__all__ = [
    'Matrix',
    'Vector',
    'naive_multiply',
    'strassen_multiply',
    'block_multiply',
    'sparse_multiply',
    'smart_multiply'
]

__version__ = '0.1.0'


#Explicit Imports
from .utils import extract_coefficients
from .solvers import solve_linear_system 