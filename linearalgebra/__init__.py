
"""
mathalgebra - A linear algebra library for mathematical operations.

This library provides tools for working with linear algebra concepts,
including equation parsing, matrix operations, and linear system solvers.
"""

__version__ = '0.1.0'


#Explicit Imports
from .utils import extract_coefficients
from .core import Matrix, Vector
from .solvers import solve_linear_system 