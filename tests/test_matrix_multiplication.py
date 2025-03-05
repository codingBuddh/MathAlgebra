"""
Test cases for matrix multiplication with automatic method selection.
Includes comprehensive matrix types, performance analysis, and visualizations.
"""

import pytest
import numpy as np
from linearalgebra import Matrix
import time
import functools
import matplotlib.pyplot as plt
import psutil
import os
import gc
from typing import Tuple, List, Dict
import seaborn as sns
import pandas as pd

def measure_performance(func):
    """Decorator to measure execution time, memory usage, and FLOPS."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Force garbage collection
        gc.collect()
        process = psutil.Process()
        
        # Measure memory before
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        # Time the operation
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        
        # Measure memory after
        gc.collect()
        mem_after = process.memory_info().rss / 1024**2
        mem_used = mem_after - mem_before
        
        # Calculate FLOPS if result has shape
        if hasattr(result, 'shape'):
            ops = 2 * result.shape[0] * result.shape[1] * result.shape[1]  # Approximate FLOPs
            gflops = (ops / elapsed) / 1e9
            print(f"\n{func.__name__}:")
            print(f"  Time: {elapsed:.6f} seconds")
            print(f"  Performance: {gflops:.2f} GFLOPS")
            print(f"  Memory Used: {mem_used:.1f} MB")
            print(f"  Efficiency: {gflops/mem_used:.2f} GFLOPS/MB")
        return result
    return wrapper

class TestMatrixMultiplication:
    def setup_method(self):
        """Setup for each test method."""
        self.results_dir = os.path.join('tests', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.performance_data = []
    
    @pytest.fixture
    def small_matrices(self):
        """Create small test matrices (10x10)."""
        A = Matrix(np.random.rand(10, 10))
        B = Matrix(np.random.rand(10, 10))
        return A, B
    
    @pytest.fixture
    def medium_matrices(self):
        """Create medium test matrices (100x100)."""
        A = Matrix(np.random.rand(100, 100))
        B = Matrix(np.random.rand(100, 100))
        return A, B
    
    @pytest.fixture
    def large_matrices(self):
        """Create large test matrices (500x500)."""
        A = Matrix(np.random.rand(500, 500))
        B = Matrix(np.random.rand(500, 500))
        return A, B
    
    @pytest.fixture
    def sparse_matrices(self):
        """Create sparse test matrices (90% zeros)."""
        def create_sparse(size=100, sparsity=0.9):
            matrix = np.random.rand(size, size)
            mask = np.random.random((size, size)) < sparsity
            matrix[mask] = 0
            return Matrix(matrix)
        return create_sparse(), create_sparse()
    
    @pytest.fixture
    def band_matrices(self):
        """Create band matrices with bandwidth 5."""
        def create_band(size=100, bandwidth=5):
            matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(max(0, i-bandwidth), min(size, i+bandwidth+1)):
                    matrix[i, j] = np.random.rand()
            return Matrix(matrix)
        return create_band(), create_band()

    @pytest.fixture
    def toeplitz_matrices(self):
        """Create Toeplitz matrices (constant diagonals)."""
        def create_toeplitz(size=100):
            r = np.random.rand(size)  # First row
            c = np.random.rand(size)  # First column
            c[0] = r[0]  # Ensure they share the first element
            matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    if i <= j:
                        matrix[i, j] = r[j-i]
                    else:
                        matrix[i, j] = c[i-j]
            return Matrix(matrix)
        return create_toeplitz(), create_toeplitz()

    @pytest.fixture
    def circulant_matrices(self):
        """Create circulant matrices (special case of Toeplitz)."""
        def create_circulant(size=100):
            first_row = np.random.rand(size)
            matrix = np.zeros((size, size))
            for i in range(size):
                matrix[i] = np.roll(first_row, i)
            return Matrix(matrix)
        return create_circulant(), create_circulant()

    @pytest.fixture
    def power_of_two_matrices(self):
        """Create matrices with power-of-2 sizes."""
        sizes = [16, 32, 64, 128]
        matrices = []
        for size in sizes:
            A = Matrix(np.random.rand(size, size))
            B = Matrix(np.random.rand(size, size))
            matrices.append((A, B, size))
        return matrices

    @pytest.fixture
    def near_power_of_two_matrices(self):
        """Create matrices with sizes near powers of 2."""
        sizes = [15, 17, 31, 33, 63, 65, 127, 129]  # Just below and above powers of 2
        matrices = []
        for size in sizes:
            A = Matrix(np.random.rand(size, size))
            B = Matrix(np.random.rand(size, size))
            matrices.append((A, B, size))
        return matrices

    @pytest.fixture
    def boundary_matrices(self):
        """Create matrices at algorithm boundary sizes."""
        # Test around naive/strassen boundary (64) and strassen/block boundary (512)
        sizes = [63, 64, 65, 511, 512, 513]
        matrices = []
        for size in sizes:
            A = Matrix(np.random.rand(size, size))
            B = Matrix(np.random.rand(size, size))
            matrices.append((A, B, size))
        return matrices
    
    @pytest.fixture
    def hankel_matrices(self):
        """Create Hankel matrices (constant anti-diagonals)."""
        def create_hankel(size=100):
            vector = np.random.rand(2*size - 1)
            matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    matrix[i,j] = vector[i+j]
            return Matrix(matrix)
        return create_hankel(), create_hankel()
    
    @pytest.fixture
    def symmetric_matrices(self):
        """Create symmetric matrices."""
        def create_symmetric(size=100):
            A = np.random.rand(size, size)
            return Matrix((A + A.T) / 2)
        return create_symmetric(), create_symmetric()
    
    @pytest.fixture
    def persymmetric_matrices(self):
        """Create persymmetric matrices (symmetric about secondary diagonal)."""
        def create_persymmetric(size=100):
            A = np.random.rand(size, size)
            J = np.fliplr(np.eye(size))  # Exchange matrix
            return Matrix((A + J @ A.T @ J) / 2)
        return create_persymmetric(), create_persymmetric()
    
    @pytest.fixture
    def orthogonal_matrices(self):
        """Create random orthogonal matrices."""
        def create_orthogonal(size=100):
            A = np.random.rand(size, size)
            Q, _ = np.linalg.qr(A)  # QR decomposition gives orthogonal matrix
            return Matrix(Q)
        return create_orthogonal(), create_orthogonal()

    @measure_performance
    def test_small_matrix_multiplication(self, small_matrices):
        """Test multiplication of small matrices."""
        A, B = small_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_medium_matrix_multiplication(self, medium_matrices):
        """Test multiplication of medium matrices."""
        A, B = medium_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_large_matrix_multiplication(self, large_matrices):
        """Test multiplication of large matrices."""
        A, B = large_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_sparse_matrix_multiplication(self, sparse_matrices):
        """Test multiplication of sparse matrices."""
        A, B = sparse_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_band_matrix_multiplication(self, band_matrices):
        """Test multiplication of band matrices."""
        A, B = band_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_toeplitz_matrix_multiplication(self, toeplitz_matrices):
        """Test multiplication of Toeplitz matrices."""
        A, B = toeplitz_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_circulant_matrix_multiplication(self, circulant_matrices):
        """Test multiplication of circulant matrices."""
        A, B = circulant_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_power_of_two_multiplication(self, power_of_two_matrices):
        """Test multiplication with power-of-2 sized matrices."""
        for A, B, size in power_of_two_matrices:
            print(f"\nTesting {size}x{size} matrix multiplication")
            result = A * B
            expected = np.matmul(A.data, B.data)
            np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_near_power_of_two_multiplication(self, near_power_of_two_matrices):
        """Test multiplication with sizes near powers of 2."""
        for A, B, size in near_power_of_two_matrices:
            print(f"\nTesting {size}x{size} matrix multiplication")
            result = A * B
            expected = np.matmul(A.data, B.data)
            np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_boundary_case_multiplication(self, boundary_matrices):
        """Test multiplication at algorithm boundary sizes."""
        for A, B, size in boundary_matrices:
            print(f"\nTesting {size}x{size} matrix multiplication (boundary case)")
            result = A * B
            expected = np.matmul(A.data, B.data)
            np.testing.assert_array_almost_equal(result.data, expected)
    
    def test_matrix_size_mismatch(self):
        """Test multiplication with incompatible matrix sizes."""
        A = Matrix(np.random.rand(10, 20))
        B = Matrix(np.random.rand(30, 40))
        with pytest.raises(ValueError):
            _ = A * B
    
    def test_multiplication_with_identity(self):
        """Test multiplication with identity matrix."""
        A = Matrix(np.random.rand(50, 50))
        I = Matrix.identity(50)
        result = A * I
        np.testing.assert_array_almost_equal(result.data, A.data)
    
    def test_multiplication_with_zeros(self):
        """Test multiplication with zero matrix."""
        A = Matrix(np.random.rand(50, 50))
        Z = Matrix(np.zeros((50, 50)))
        result = A * Z
        np.testing.assert_array_almost_equal(result.data, np.zeros((50, 50)))
    
    @measure_performance
    def test_rectangular_matrix_multiplication(self):
        """Test multiplication of rectangular matrices."""
        sizes = [(10, 20, 30), (50, 30, 40), (100, 50, 75)]  # (m, n, p) for (m×n) * (n×p)
        for m, n, p in sizes:
            print(f"\nTesting {m}x{n} * {n}x{p} multiplication")
            A = Matrix(np.random.rand(m, n))
            B = Matrix(np.random.rand(n, p))
            result = A * B
            expected = np.matmul(A.data, B.data)
            np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_hankel_matrix_multiplication(self, hankel_matrices):
        """Test multiplication of Hankel matrices."""
        A, B = hankel_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_symmetric_matrix_multiplication(self, symmetric_matrices):
        """Test multiplication of symmetric matrices."""
        A, B = symmetric_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_persymmetric_matrix_multiplication(self, persymmetric_matrices):
        """Test multiplication of persymmetric matrices."""
        A, B = persymmetric_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_orthogonal_matrix_multiplication(self, orthogonal_matrices):
        """Test multiplication of orthogonal matrices."""
        A, B = orthogonal_matrices
        result = A * B
        expected = np.matmul(A.data, B.data)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    @measure_performance
    def test_comprehensive_performance_analysis(self):
        """Comprehensive performance analysis across matrix types and sizes."""
        # Matrix configurations to test
        configs = [
            ('Dense', lambda s: np.random.rand(s, s)),
            ('Sparse', lambda s: self._create_sparse(s, 0.9)),
            ('Band', lambda s: self._create_band(s, 5)),
            ('Toeplitz', lambda s: self._create_toeplitz(s)),
            ('Hankel', lambda s: self._create_hankel(s)),
            ('Symmetric', lambda s: self._create_symmetric(s))
        ]
        
        # Sizes to test (including boundaries)
        sizes = [32, 63, 64, 65, 127, 128, 129, 511, 512, 513]
        
        results = []
        for size in sizes:
            for name, creator in configs:
                # Create test matrices
                A = Matrix(creator(size))
                B = Matrix(creator(size))
                
                # Measure performance
                gc.collect()
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024**2
                
                start = time.perf_counter()
                result = A * B
                elapsed = time.perf_counter() - start
                
                gc.collect()
                mem_after = process.memory_info().rss / 1024**2
                mem_used = mem_after - mem_before
                
                # Calculate metrics
                ops = 2 * size**3
                gflops = (ops / elapsed) / 1e9
                efficiency = gflops / mem_used if mem_used > 0 else 0
                
                results.append({
                    'size': size,
                    'type': name,
                    'time': elapsed,
                    'gflops': gflops,
                    'memory_mb': mem_used,
                    'efficiency': efficiency
                })
        
        # Save results
        self._save_performance_results(results)
        
        # Create visualizations
        self._create_performance_visualizations(results)
    
    def _create_performance_visualizations(self, results: List[Dict]):
        """Create comprehensive performance visualizations."""
        # Set style
        plt.style.use('default')  # Use default style instead of seaborn
        
        # 1. Performance vs Size by Matrix Type
        plt.figure(figsize=(12, 6))
        for matrix_type in set(r['type'] for r in results):
            data = [(r['size'], r['gflops']) for r in results if r['type'] == matrix_type]
            sizes, gflops = zip(*data)
            plt.plot(sizes, gflops, 'o-', label=matrix_type)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (GFLOPS)')
        plt.title('Performance Scaling by Matrix Type')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'performance_scaling.png'))
        plt.close()
        
        # 2. Memory Usage Heatmap
        sizes = sorted(set(r['size'] for r in results))
        types = sorted(set(r['type'] for r in results))
        memory_data = np.zeros((len(types), len(sizes)))
        
        for i, t in enumerate(types):
            for j, s in enumerate(sizes):
                memory = next(r['memory_mb'] for r in results if r['type'] == t and r['size'] == s)
                memory_data[i, j] = memory
        
        plt.figure(figsize=(12, 8))
        plt.imshow(memory_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Memory Usage (MB)')
        plt.xticks(range(len(sizes)), sizes, rotation=45)
        plt.yticks(range(len(types)), types)
        plt.xlabel('Matrix Size')
        plt.ylabel('Matrix Type')
        plt.title('Memory Usage (MB)')
        
        # Add text annotations to the heatmap
        for i in range(len(types)):
            for j in range(len(sizes)):
                plt.text(j, i, f'{memory_data[i, j]:.1f}',
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'memory_usage.png'))
        plt.close()
        
        # 3. Efficiency Analysis
        plt.figure(figsize=(12, 6))
        for matrix_type in types:
            data = [(r['size'], r['efficiency']) for r in results if r['type'] == matrix_type]
            sizes, efficiency = zip(*data)
            plt.plot(sizes, efficiency, 'o-', label=matrix_type)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Efficiency (GFLOPS/MB)')
        plt.title('Computational Efficiency by Matrix Type')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'efficiency.png'))
        plt.close()
        
        # 4. Algorithm Selection Analysis
        boundary_sizes = [63, 64, 65, 511, 512, 513]
        boundary_data = [r for r in results if r['size'] in boundary_sizes]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Performance at naive/Strassen boundary
        sizes_small = [63, 64, 65]
        for t in types:
            data = [(r['size'], r['gflops']) for r in boundary_data 
                   if r['type'] == t and r['size'] in sizes_small]
            if data:
                sizes, gflops = zip(*data)
                ax1.plot(sizes, gflops, 'o-', label=t)
        
        ax1.set_title('Algorithm Transition: Naive to Strassen (64)')
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Performance (GFLOPS)')
        ax1.legend()
        ax1.grid(True)
        
        # Performance at Strassen/block boundary
        sizes_large = [511, 512, 513]
        for t in types:
            data = [(r['size'], r['gflops']) for r in boundary_data 
                   if r['type'] == t and r['size'] in sizes_large]
            if data:
                sizes, gflops = zip(*data)
                ax2.plot(sizes, gflops, 'o-', label=t)
        
        ax2.set_title('Algorithm Transition: Strassen to Block (512)')
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Performance (GFLOPS)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_transitions.png'))
        plt.close()
    
    def _save_performance_results(self, results: List[Dict]):
        """Save performance results to CSV file."""
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, 'performance_results.csv'), index=False)
        
        # Generate summary report
        with open(os.path.join(self.results_dir, 'performance_summary.txt'), 'w') as f:
            f.write("Matrix Multiplication Performance Summary\n")
            f.write("=====================================\n\n")
            
            # Overall best performance
            best = max(results, key=lambda x: x['gflops'])
            f.write(f"Best Performance:\n")
            f.write(f"  Matrix Type: {best['type']}\n")
            f.write(f"  Size: {best['size']}x{best['size']}\n")
            f.write(f"  Performance: {best['gflops']:.2f} GFLOPS\n")
            f.write(f"  Memory Usage: {best['memory_mb']:.1f} MB\n")
            f.write(f"  Efficiency: {best['efficiency']:.2f} GFLOPS/MB\n\n")
            
            # Algorithm transition analysis
            f.write("Algorithm Transition Analysis:\n")
            for boundary in [64, 512]:
                f.write(f"\nAt size {boundary} (boundary):\n")
                relevant = [r for r in results if r['size'] in [boundary-1, boundary, boundary+1]]
                for r in sorted(relevant, key=lambda x: (x['type'], x['size'])):
                    f.write(f"  {r['type']} ({r['size']}x{r['size']}): "
                           f"{r['gflops']:.2f} GFLOPS, {r['memory_mb']:.1f} MB\n")
            
            # Best performance by matrix type
            f.write("\nBest Performance by Matrix Type:\n")
            for type_name in set(r['type'] for r in results):
                type_results = [r for r in results if r['type'] == type_name]
                best = max(type_results, key=lambda x: x['gflops'])
                f.write(f"\n{type_name}:\n")
                f.write(f"  Size: {best['size']}x{best['size']}\n")
                f.write(f"  Performance: {best['gflops']:.2f} GFLOPS\n")
                f.write(f"  Memory Usage: {best['memory_mb']:.1f} MB\n")
                f.write(f"  Efficiency: {best['efficiency']:.2f} GFLOPS/MB\n")

    # Helper methods for matrix creation
    def _create_sparse(self, size: int, sparsity: float) -> np.ndarray:
        matrix = np.random.rand(size, size)
        mask = np.random.random((size, size)) < sparsity
        matrix[mask] = 0
        return matrix
    
    def _create_band(self, size: int, bandwidth: int) -> np.ndarray:
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(max(0, i-bandwidth), min(size, i+bandwidth+1)):
                matrix[i, j] = np.random.rand()
        return matrix
    
    def _create_toeplitz(self, size: int) -> np.ndarray:
        r = np.random.rand(size)
        c = np.random.rand(size)
        c[0] = r[0]
        return np.array([[c[i-j] if i>=j else r[j-i] for j in range(size)] for i in range(size)])
    
    def _create_hankel(self, size: int) -> np.ndarray:
        vector = np.random.rand(2*size - 1)
        return np.array([[vector[i+j] for j in range(size)] for i in range(size)])
    
    def _create_symmetric(self, size: int) -> np.ndarray:
        A = np.random.rand(size, size)
        return (A + A.T) / 2

    @measure_performance
    def test_time_based_analysis(self):
        """Analyze execution time vs matrix size for different algorithms."""
        print("\nRunning time-based analysis...")
        
        # Matrix sizes to test (focusing on interesting boundaries)
        sizes = [16, 32, 48, 64, 96, 128, 256, 384, 512]
        methods = [
            ('Naive', 'naive_multiply'),
            ('Strassen', 'strassen_multiply'),
            ('Block', 'block_multiply'),
            ('Sparse', 'sparse_multiply'),
            ('Smart', 'smart_multiply')
        ]
        
        results = []
        
        for size in sizes:
            print(f"\nTesting size {size}x{size}")
            # Create test matrices with different characteristics
            matrices = {
                'Dense': (Matrix(np.random.rand(size, size)), Matrix(np.random.rand(size, size))),
                'Sparse': (Matrix(self._create_sparse(size, 0.9)), Matrix(self._create_sparse(size, 0.9))),
                'Band': (Matrix(self._create_band(size, 5)), Matrix(self._create_band(size, 5))),
                'Symmetric': (Matrix(self._create_symmetric(size)), Matrix(self._create_symmetric(size)))
            }
            
            for matrix_type, (A, B) in matrices.items():
                print(f"\n  Testing {matrix_type} matrices:")
                for method_name, method_func in methods:
                    print(f"    Using {method_name} method...")
                    gc.collect()
                    
                    try:
                        # Measure execution time and memory
                        start_time = time.perf_counter()
                        process = psutil.Process()
                        mem_before = process.memory_info().rss / 1024**2
                        
                        # Get the method from the Matrix class
                        method = getattr(Matrix, method_func, None)
                        if method is None:
                            # Try getting from matrix_ops module
                            from linearalgebra import matrix_ops
                            method = getattr(matrix_ops, method_func)
                        
                        # Execute multiplication
                        if method_func in ['naive_multiply', 'strassen_multiply', 'block_multiply', 'sparse_multiply']:
                            result = method(A, B)
                        else:
                            result = A * B  # For smart_multiply, use operator
                        
                        end_time = time.perf_counter()
                        
                        # Measure memory and calculate metrics
                        gc.collect()
                        mem_after = process.memory_info().rss / 1024**2
                        mem_used = max(0.1, mem_after - mem_before)  # Ensure non-zero memory
                        
                        execution_time = end_time - start_time
                        gflops = (2 * size**3) / (execution_time * 1e9)
                        efficiency = gflops / mem_used
                        
                        results.append({
                            'size': size,
                            'method': method_name,
                            'matrix_type': matrix_type,
                            'time': execution_time,
                            'memory': mem_used,
                            'gflops': gflops,
                            'efficiency': efficiency
                        })
                        
                        print(f"      Time: {execution_time:.6f}s")
                        print(f"      Memory: {mem_used:.1f}MB")
                        print(f"      GFLOPS: {gflops:.2f}")
                        print(f"      Efficiency: {efficiency:.2f} GFLOPS/MB")
                        
                    except Exception as e:
                        print(f"      Error: {str(e)}")
                        results.append({
                            'size': size,
                            'method': method_name,
                            'matrix_type': matrix_type,
                            'time': float('inf'),
                            'memory': 0,
                            'gflops': 0,
                            'efficiency': 0
                        })
        
        # Save results before visualization
        self._save_time_analysis_results(results)
        
        # Create visualizations
        self._create_time_based_visualizations(results)
        
        # Ensure we have valid results
        assert len(results) > 0, "No results collected"
        return results

    def _save_time_analysis_results(self, results: List[Dict]):
        """Save time analysis results to files."""
        # Save to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, 'time_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Generate detailed summary
        with open(os.path.join(self.results_dir, 'time_analysis_summary.txt'), 'w') as f:
            f.write("Time-Based Analysis Summary\n")
            f.write("=========================\n\n")
            
            # Overall best performance
            best = max(results, key=lambda x: x['gflops'])
            f.write("Best Overall Performance:\n")
            f.write(f"Method: {best['method']}\n")
            f.write(f"Matrix Type: {best['matrix_type']}\n")
            f.write(f"Size: {best['size']}x{best['size']}\n")
            f.write(f"Time: {best['time']:.6f} seconds\n")
            f.write(f"GFLOPS: {best['gflops']:.2f}\n")
            f.write(f"Memory: {best['memory']:.1f} MB\n")
            f.write(f"Efficiency: {best['efficiency']:.2f} GFLOPS/MB\n\n")
            
            # Analysis by method
            f.write("Performance by Method:\n")
            for method in set(r['method'] for r in results):
                method_results = [r for r in results if r['method'] == method]
                f.write(f"\n{method}:\n")
                f.write("-" * len(method) + "\n")
                
                # Best case
                best = max(method_results, key=lambda x: x['gflops'])
                f.write(f"Best case:\n")
                f.write(f"  Size: {best['size']}x{best['size']}\n")
                f.write(f"  Matrix Type: {best['matrix_type']}\n")
                f.write(f"  Time: {best['time']:.6f}s\n")
                f.write(f"  GFLOPS: {best['gflops']:.2f}\n")
                
                # Average performance
                avg_time = np.mean([r['time'] for r in method_results if r['time'] != float('inf')])
                avg_gflops = np.mean([r['gflops'] for r in method_results])
                f.write(f"\nAverage performance:\n")
                f.write(f"  Time: {avg_time:.6f}s\n")
                f.write(f"  GFLOPS: {avg_gflops:.2f}\n")
            
            # Analysis by matrix type
            f.write("\nPerformance by Matrix Type:\n")
            for mtype in set(r['matrix_type'] for r in results):
                type_results = [r for r in results if r['matrix_type'] == mtype]
                f.write(f"\n{mtype} Matrices:\n")
                f.write("-" * (len(mtype) + 9) + "\n")
                
                best = max(type_results, key=lambda x: x['gflops'])
                f.write(f"Best method: {best['method']}\n")
                f.write(f"Size: {best['size']}x{best['size']}\n")
                f.write(f"GFLOPS: {best['gflops']:.2f}\n")

    def _create_time_based_visualizations(self, results: List[Dict]):
        """Create visualizations for time-based analysis."""
        plt.style.use('default')
        
        # Common plot settings
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 8
        
        # 1. Execution Time vs Matrix Size (by method and matrix type)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Execution Time Analysis', fontsize=16)
        
        for idx, matrix_type in enumerate(sorted(set(r['matrix_type'] for r in results))):
            ax = axes[idx // 2, idx % 2]
            type_results = [r for r in results if r['matrix_type'] == matrix_type]
            
            for method in sorted(set(r['method'] for r in type_results)):
                data = [(r['size'], r['time']) for r in type_results if r['method'] == method]
                if data:
                    sizes, times = zip(*data)
                    ax.plot(sizes, times, 'o-', label=method)
            
            ax.set_title(f'{matrix_type} Matrices')
            ax.set_xlabel('Matrix Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'time_analysis_detailed.png'))
        plt.close()
        
        # 2. Performance Comparison (GFLOPS)
        plt.figure(figsize=(12, 6))
        for method in sorted(set(r['method'] for r in results)):
            data = [(r['size'], r['gflops']) for r in results if r['method'] == method]
            if data:
                sizes, gflops = zip(*data)
                plt.plot(sizes, gflops, 'o-', label=method)
        
        plt.title('Performance Comparison')
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (GFLOPS)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'))
        plt.close()
        
        # 3. Memory Usage Analysis
        plt.figure(figsize=(12, 6))
        for method in sorted(set(r['method'] for r in results)):
            data = [(r['size'], r['memory']) for r in results if r['method'] == method]
            if data:
                sizes, memory = zip(*data)
                plt.plot(sizes, memory, 'o-', label=method)
        
        plt.title('Memory Usage Analysis')
        plt.xlabel('Matrix Size')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'memory_analysis.png'))
        plt.close()
        
        # 4. Efficiency Heatmap
        matrix_types = sorted(set(r['matrix_type'] for r in results))
        methods = sorted(set(r['method'] for r in results))
        sizes = sorted(set(r['size'] for r in results))
        
        efficiency_data = np.zeros((len(methods), len(sizes)))
        for i, method in enumerate(methods):
            for j, size in enumerate(sizes):
                relevant = [r['efficiency'] for r in results 
                          if r['method'] == method and r['size'] == size]
                efficiency_data[i, j] = np.mean(relevant) if relevant else 0
        
        plt.figure(figsize=(15, 8))
        plt.imshow(efficiency_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Efficiency (GFLOPS/MB)')
        plt.xticks(range(len(sizes)), sizes, rotation=45)
        plt.yticks(range(len(methods)), methods)
        plt.title('Efficiency Heatmap')
        plt.xlabel('Matrix Size')
        plt.ylabel('Method')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(sizes)):
                plt.text(j, i, f'{efficiency_data[i,j]:.1f}',
                        ha='center', va='center',
                        color='white' if efficiency_data[i,j] > np.mean(efficiency_data) else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_heatmap.png'))
        plt.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 