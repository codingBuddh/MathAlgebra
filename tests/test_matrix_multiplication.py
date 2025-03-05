"""
Test cases for different matrix multiplication methods with timing information.
"""

import numpy as np
import time
from linearalgebra import Matrix, basic_multiply, strassen_multiply, sparse_multiply, smart_multiply
import pytest

def time_operation(func, *args, runs=3):
    """Helper function to time operations."""
    times = []
    for _ in range(runs):
        start_time = time.perf_counter()
        result = func(*args)
        elapsed_time = time.perf_counter() - start_time
        times.append(elapsed_time)
    return result, min(times)  # Return result and best time

class TestMatrixMultiplication:
    @pytest.fixture
    def small_matrices(self):
        """Small dense matrices (4x4)"""
        A = Matrix(np.random.rand(4, 4))
        B = Matrix(np.random.rand(4, 4))
        return A, B
    
    @pytest.fixture
    def medium_matrices(self):
        """Medium dense matrices (64x64)"""
        A = Matrix(np.random.rand(64, 64))
        B = Matrix(np.random.rand(64, 64))
        return A, B
    
    @pytest.fixture
    def large_matrices(self):
        """Large dense matrices (256x256)"""
        A = Matrix(np.random.rand(256, 256))
        B = Matrix(np.random.rand(256, 256))
        return A, B
    
    @pytest.fixture
    def sparse_matrices(self):
        """Large sparse matrices (1000x1000 with 95% sparsity)"""
        def create_sparse(shape, sparsity):
            matrix = np.random.rand(*shape)
            mask = np.random.random(shape) < sparsity
            matrix[mask] = 0
            return Matrix(matrix)
        
        A = create_sparse((1000, 1000), 0.95)
        B = create_sparse((1000, 1000), 0.95)
        return A, B
    
    def test_small_matrices(self, small_matrices, capsys):
        """Test multiplication methods with small matrices."""
        A, B = small_matrices
        print("\nSmall Matrix Multiplication Tests (4x4)")
        print("=" * 50)
        
        # Test each method
        methods = [
            ('Basic', lambda x, y: basic_multiply(x, y)),
            ('Strassen', lambda x, y: strassen_multiply(x, y)),
            ('Sparse', lambda x, y: sparse_multiply(x, y)),
            ('Smart', lambda x, y: smart_multiply(x, y))
        ]
        
        results = {}
        for name, method in methods:
            result, elapsed_time = time_operation(method, A, B)
            results[name] = (result, elapsed_time)
            print(f"{name} Method: {elapsed_time:.6f} seconds")
        
        # Verify all methods give same result
        base_result = results['Basic'][0]
        for name, (result, _) in results.items():
            if name != 'Basic':
                np.testing.assert_array_almost_equal(
                    base_result.data, 
                    result.data,
                    decimal=10,
                    err_msg=f"{name} method gave different result"
                )
    
    def test_medium_matrices(self, medium_matrices, capsys):
        """Test multiplication methods with medium matrices."""
        A, B = medium_matrices
        print("\nMedium Matrix Multiplication Tests (64x64)")
        print("=" * 50)
        
        methods = [
            ('Basic', lambda x, y: basic_multiply(x, y)),
            ('Strassen', lambda x, y: strassen_multiply(x, y)),
            ('Sparse', lambda x, y: sparse_multiply(x, y)),
            ('Smart', lambda x, y: smart_multiply(x, y))
        ]
        
        for name, method in methods:
            _, elapsed_time = time_operation(method, A, B)
            print(f"{name} Method: {elapsed_time:.6f} seconds")
    
    def test_large_matrices(self, large_matrices, capsys):
        """Test multiplication methods with large matrices."""
        A, B = large_matrices
        print("\nLarge Matrix Multiplication Tests (256x256)")
        print("=" * 50)
        
        methods = [
            ('Basic', lambda x, y: basic_multiply(x, y)),
            ('Strassen', lambda x, y: strassen_multiply(x, y)),
            ('Sparse', lambda x, y: sparse_multiply(x, y)),
            ('Smart', lambda x, y: smart_multiply(x, y))
        ]
        
        for name, method in methods:
            _, elapsed_time = time_operation(method, A, B)
            print(f"{name} Method: {elapsed_time:.6f} seconds")
    
    def test_sparse_matrices(self, sparse_matrices, capsys):
        """Test multiplication methods with sparse matrices."""
        A, B = sparse_matrices
        print("\nSparse Matrix Multiplication Tests (1000x1000, 95% sparse)")
        print("=" * 50)
        
        methods = [
            ('Basic', lambda x, y: basic_multiply(x, y)),
            ('Sparse', lambda x, y: sparse_multiply(x, y)),
            ('Smart', lambda x, y: smart_multiply(x, y))
        ]
        
        for name, method in methods:
            _, elapsed_time = time_operation(method, A, B)
            print(f"{name} Method: {elapsed_time:.6f} seconds")
    
    def test_different_formats(self, sparse_matrices, capsys):
        """Test sparse multiplication with different formats."""
        A, B = sparse_matrices
        print("\nSparse Matrix Format Tests (1000x1000, 95% sparse)")
        print("=" * 50)
        
        formats = ['csr', 'csc', 'coo']
        for format in formats:
            _, elapsed_time = time_operation(
                lambda x, y: sparse_multiply(x, y, format=format),
                A, B
            )
            print(f"Sparse ({format.upper()}) Method: {elapsed_time:.6f} seconds")
    
    def test_edge_cases(self, capsys):
        """Test edge cases and special matrices."""
        print("\nEdge Cases Tests")
        print("=" * 50)
        
        # Identity matrix multiplication
        size = 100
        A = Matrix(np.random.rand(size, size))
        I = Matrix.identity(size)
        
        _, elapsed_time = time_operation(lambda: smart_multiply(A, I))
        print(f"Identity Matrix Multiplication: {elapsed_time:.6f} seconds")
        
        # Zero matrix multiplication
        Z = Matrix(np.zeros((size, size)))
        _, elapsed_time = time_operation(lambda: smart_multiply(A, Z))
        print(f"Zero Matrix Multiplication: {elapsed_time:.6f} seconds")
        
        # Diagonal matrix multiplication
        D = Matrix(np.diag(np.random.rand(size)))
        _, elapsed_time = time_operation(lambda: smart_multiply(A, D))
        print(f"Diagonal Matrix Multiplication: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 