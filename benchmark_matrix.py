"""
Benchmark script for MathAlgebra matrix multiplication performance.
"""

import time
import numpy as np
from linearalgebra import Matrix

def benchmark_matrix_multiplication(size=1000, runs=5):
    """
    Benchmark matrix multiplication for given size.
    
    Args:
        size (int): Size of square matrix (default: 1000)
        runs (int): Number of test runs for averaging (default: 5)
    """
    print(f"\nBenchmarking {size}x{size} Matrix Multiplication")
    print("-" * 50)
    
    # Generate random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    matrix_A = Matrix(A)
    matrix_B = Matrix(B)
    
    # Warm-up run
    _ = matrix_A * matrix_B
    
    # Actual benchmark
    times = []
    for i in range(runs):
        start_time = time.perf_counter()
        result = matrix_A * matrix_B
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"Run {i+1}: {elapsed_time:.4f} seconds")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    
    print("\nResults:")
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"Best Time: {min_time:.4f} seconds")
    print(f"Matrix Size: {size}x{size}")
    print(f"Operations: ~{size**3:,} floating-point operations")
    print(f"GFLOPS: {(2 * size**3) / (min_time * 1e9):.2f}")

if __name__ == "__main__":
    # Test with 1000x1000 matrices
    benchmark_matrix_multiplication(size=1000)
    
    # Optionally test with other sizes
    # benchmark_matrix_multiplication(size=500)
    # benchmark_matrix_multiplication(size=2000) 