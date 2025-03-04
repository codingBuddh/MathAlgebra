"""
Matrix decomposition methods for the mathalgebra library.
"""

import numpy as np
from typing import Tuple
from .core import Matrix

def lu_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Compute the LU decomposition of a matrix.
    
    Args:
        A (Matrix): Input matrix
        
    Returns:
        Tuple[Matrix, Matrix]: L and U matrices such that A = L * U
    """
    if A.rows != A.cols:
        raise ValueError("LU decomposition requires a square matrix")
    
    # Check if matrix is singular by computing determinant
    if abs(A.determinant()) < 1e-10:
        raise ValueError("Matrix is singular, cannot perform LU decomposition")
    
    n = A.rows
    L = np.eye(n)
    U = A.data.copy()
    
    for i in range(n):
        if abs(U[i, i]) < 1e-10:
            raise ValueError("Matrix is singular, cannot perform LU decomposition")
            
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    
    return Matrix(L), Matrix(U)

def qr_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Compute the QR decomposition of a matrix using Gram-Schmidt process.
    
    Args:
        A (Matrix): Input matrix
        
    Returns:
        Tuple[Matrix, Matrix]: Q and R matrices such that A = Q * R
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A.data[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A.data[:, j])
            v -= R[i, j] * Q[:, i]
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            raise ValueError("Matrix columns are linearly dependent")
            
        Q[:, j] = v / norm_v
        R[j, j] = np.dot(Q[:, j], A.data[:, j])
    
    return Matrix(Q), Matrix(R)

def svd_decomposition(A: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Compute the Singular Value Decomposition (SVD) of a matrix.
    
    Args:
        A (Matrix): Input matrix
        
    Returns:
        Tuple[Matrix, Matrix, Matrix]: U, Sigma, and V^T matrices such that A = U * Sigma * V^T
    """
    # Use full_matrices=False to get the "economy" SVD which matches our test expectations
    U, s, Vt = np.linalg.svd(A.data, full_matrices=False)
    
    # Create diagonal matrix Sigma from singular values
    m, n = A.shape
    min_dim = min(m, n)
    Sigma = np.zeros((U.shape[1], Vt.shape[0]))
    
    # Fill diagonal with singular values
    for i in range(min_dim):
        Sigma[i, i] = s[i]
    
    return Matrix(U), Matrix(Sigma), Matrix(Vt) 