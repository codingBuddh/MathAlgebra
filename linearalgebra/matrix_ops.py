"""
Advanced matrix multiplication implementations for different scenarios.
Includes various algorithms optimized for different matrix characteristics.
"""

import numpy as np
from typing import Union, List, Optional, TYPE_CHECKING, Tuple
import scipy.sparse as sp

if TYPE_CHECKING:
    from .core import Matrix

def naive_multiply(A: 'Matrix', B: 'Matrix') -> 'Matrix':
    """
    Standard matrix multiplication using the naive O(n³) algorithm.
    Suitable for small matrices (typically < 64x64).
    
    Args:
        A (Matrix): First matrix
        B (Matrix): Second matrix
        
    Returns:
        Matrix: Result of multiplication
    """
    if A.cols != B.rows:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    result = np.zeros((A.rows, B.cols))
    for i in range(A.rows):
        for j in range(B.cols):
            for k in range(A.cols):
                result[i, j] += A.data[i, k] * B.data[k, j]
    
    from .core import Matrix
    return Matrix(result)

def strassen_multiply(A: 'Matrix', B: 'Matrix', threshold: int = 64) -> 'Matrix':
    """
    Strassen's matrix multiplication algorithm.
    Time complexity: O(n^2.807)
    
    Args:
        A (Matrix): First matrix
        B (Matrix): Second matrix
        threshold (int): Size threshold below which to use standard multiplication
        
    Returns:
        Matrix: Result of multiplication
    """
    if A.cols != B.rows:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    A_data = A.data
    B_data = B.data
    
    def _strassen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        
        if n <= threshold:
            return np.matmul(A, B)
        
        # Pad matrices if necessary
        if n % 2 != 0:
            pad_size = ((0, 1), (0, 1))
            A = np.pad(A, pad_size, mode='constant')
            B = np.pad(B, pad_size, mode='constant')
            n += 1
        
        # Split matrices
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        
        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]
        
        # Compute the seven products
        M1 = _strassen(A11 + A22, B11 + B22)
        M2 = _strassen(A21 + A22, B11)
        M3 = _strassen(A11, B12 - B22)
        M4 = _strassen(A22, B21 - B11)
        M5 = _strassen(A11 + A12, B22)
        M6 = _strassen(A21 - A11, B11 + B12)
        M7 = _strassen(A12 - A22, B21 + B22)
        
        # Combine results
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        # Combine quadrants
        C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
        
        # Remove padding if added
        if A.shape[0] > A_data.shape[0]:
            C = C[:A_data.shape[0], :B_data.shape[1]]
        
        return C
    
    result = _strassen(A_data, B_data)
    from .core import Matrix
    return Matrix(result)

def block_multiply(A: 'Matrix', B: 'Matrix', block_size: int = 32) -> 'Matrix':
    """
    Block matrix multiplication optimized for cache efficiency.
    Uses tiling to improve memory access patterns.
    
    Args:
        A (Matrix): First matrix
        B (Matrix): Second matrix
        block_size (int): Size of blocks for tiling
        
    Returns:
        Matrix: Result of multiplication
    """
    if A.cols != B.rows:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    m, n = A.shape
    n, p = B.shape
    
    # Initialize result matrix
    result = np.zeros((m, p))
    
    # Iterate over blocks
    for i in range(0, m, block_size):
        for j in range(0, p, block_size):
            for k in range(0, n, block_size):
                # Get block boundaries
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, p)
                k_end = min(k + block_size, n)
                
                # Multiply blocks
                result[i:i_end, j:j_end] += np.matmul(
                    A.data[i:i_end, k:k_end],
                    B.data[k:k_end, j:j_end]
                )
    
    from .core import Matrix
    return Matrix(result)

def sparse_multiply(A: 'Matrix', B: 'Matrix', format: str = 'csr') -> 'Matrix':
    """
    Specialized multiplication for sparse matrices using SciPy's sparse matrix operations.
    
    Args:
        A (Matrix): First matrix
        B (Matrix): Second matrix
        format (str): Sparse matrix format ('csr', 'csc', or 'coo')
        
    Returns:
        Matrix: Result of multiplication
    """
    if A.cols != B.rows:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    # Convert to sparse matrices
    A_sparse = sp.csr_matrix(A.data) if format == 'csr' else \
               sp.csc_matrix(A.data) if format == 'csc' else \
               sp.coo_matrix(A.data)
    
    B_sparse = sp.csr_matrix(B.data) if format == 'csr' else \
               sp.csc_matrix(B.data) if format == 'csc' else \
               sp.coo_matrix(B.data)
    
    result = A_sparse.dot(B_sparse)
    from .core import Matrix
    return Matrix(result.toarray())

def smart_multiply(A: 'Matrix', B: 'Matrix', **kwargs) -> 'Matrix':
    """
    Automatically choose the best multiplication method based on matrix characteristics.
    
    Args:
        A (Matrix): First matrix
        B (Matrix): Second matrix
        **kwargs: Additional arguments for specific methods
            - threshold: Size threshold for Strassen's algorithm
            - block_size: Block size for block multiplication
            - sparsity_threshold: Threshold for considering a matrix sparse
        
    Returns:
        Matrix: Result of multiplication
    """
    def calculate_sparsity(mat: 'Matrix') -> float:
        return np.count_nonzero(mat.data == 0) / mat.data.size
    
    size = max(A.rows, A.cols, B.rows, B.cols)
    A_sparsity = calculate_sparsity(A)
    B_sparsity = calculate_sparsity(B)
    
    # For very sparse matrices
    if A_sparsity > 0.8 and B_sparsity > 0.8:
        return sparse_multiply(A, B)
    
    # For small matrices
    if size <= 64:
        return naive_multiply(A, B)
    
    # For large matrices
    if size >= 512:
        return block_multiply(A, B, kwargs.get('block_size', 32))
    
    # For medium-sized matrices
    return strassen_multiply(A, B, kwargs.get('threshold', 64)) 