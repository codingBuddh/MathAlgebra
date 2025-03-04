"""
Tests for the decompositions module in the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.core import Matrix
from linearalgebra.decompositions import lu_decomposition, qr_decomposition, svd_decomposition

def test_lu_decomposition():
    A = Matrix([
        [2, 3, 1],
        [4, 7, 5],
        [6, 5, 3]
    ])
    
    L, U = lu_decomposition(A)
    
    # Verify L is lower triangular with 1s on diagonal
    assert np.allclose(np.tril(L.data), L.data)
    assert np.allclose(np.diag(L.data), np.ones(3))
    
    # Verify U is upper triangular
    assert np.allclose(np.triu(U.data), U.data)
    
    # Verify A = L * U
    LU = L * U
    assert np.allclose(A.data, LU.data)

def test_lu_decomposition_non_square():
    A = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    with pytest.raises(ValueError, match="square"):
        lu_decomposition(A)

def test_lu_decomposition_singular():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    
    with pytest.raises(ValueError, match="singular"):
        lu_decomposition(A)

def test_qr_decomposition():
    A = Matrix([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ])
    
    Q, R = qr_decomposition(A)
    
    # Verify Q is orthogonal (Q^T * Q = I)
    QT = Q.transpose()
    QTQ = QT * Q
    assert np.allclose(QTQ.data, np.eye(3), atol=1e-10)
    
    # Verify R is upper triangular
    assert np.allclose(np.triu(R.data), R.data)
    
    # Verify A = Q * R
    QR = Q * R
    assert np.allclose(A.data, QR.data)

def test_qr_decomposition_linearly_dependent():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    
    with pytest.raises(ValueError, match="linearly dependent"):
        qr_decomposition(A)

def test_qr_decomposition_rectangular():
    # Test with a rectangular matrix with linearly independent columns
    A = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])  # 3x2 matrix with linearly independent columns
    
    Q, R = qr_decomposition(A)
    
    # Verify Q has orthogonal columns
    QT = Q.transpose()
    QTQ = QT * Q
    assert np.allclose(QTQ.data, np.eye(min(A.rows, A.cols)), atol=1e-10)
    
    # Verify R is upper triangular
    assert np.allclose(np.triu(R.data), R.data)
    
    # Verify A = Q * R
    QR = Q * R
    assert np.allclose(A.data, QR.data)

def test_svd_decomposition():
    A = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    
    U, S, VT = svd_decomposition(A)
    
    # Verify U has orthogonal columns
    UT = U.transpose()
    UTU = UT * U
    assert np.allclose(UTU.data, np.eye(min(A.rows, A.cols)), atol=1e-10)
    
    # Verify V has orthogonal columns
    V = VT.transpose()
    VTV = V.transpose() * V
    assert np.allclose(VTV.data, np.eye(min(A.rows, A.cols)), atol=1e-10)
    
    # Verify S is diagonal with non-negative entries
    assert np.allclose(np.diag(np.diag(S.data)), S.data)
    assert np.all(np.diag(S.data) >= 0)
    
    # Verify A = U * S * VT
    USV = U * S * VT
    assert np.allclose(A.data, USV.data)

def test_svd_decomposition_square():
    A = Matrix([
        [1, 2],
        [3, 4]
    ])
    
    U, S, VT = svd_decomposition(A)
    
    # Verify U is orthogonal
    UT = U.transpose()
    UTU = UT * U
    assert np.allclose(UTU.data, np.eye(A.rows), atol=1e-10)
    
    # Verify V is orthogonal
    V = VT.transpose()
    VTV = V.transpose() * V
    assert np.allclose(VTV.data, np.eye(A.cols), atol=1e-10)
    
    # Verify S is diagonal with non-negative entries
    assert np.allclose(np.diag(np.diag(S.data)), S.data)
    assert np.all(np.diag(S.data) >= 0)
    
    # Verify A = U * S * VT
    USV = U * S * VT
    assert np.allclose(A.data, USV.data)

def test_svd_decomposition_singular():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    
    U, S, VT = svd_decomposition(A)
    
    # Verify U has orthogonal columns
    UT = U.transpose()
    UTU = UT * U
    assert np.allclose(UTU.data, np.eye(min(A.rows, A.cols)), atol=1e-10)
    
    # Verify V has orthogonal columns
    V = VT.transpose()
    VTV = V.transpose() * V
    assert np.allclose(VTV.data, np.eye(min(A.rows, A.cols)), atol=1e-10)
    
    # Verify S is diagonal with non-negative entries
    assert np.allclose(np.diag(np.diag(S.data)), S.data)
    assert np.all(np.diag(S.data) >= 0)
    
    # Verify A = U * S * VT
    USV = U * S * VT
    assert np.allclose(A.data, USV.data)
    
    # Verify rank deficiency
    s_values = np.diag(S.data)
    assert np.sum(s_values > 1e-10) < min(A.rows, A.cols) 