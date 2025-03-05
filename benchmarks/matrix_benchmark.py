"""
Comprehensive benchmark suite for matrix multiplication methods.
Tests different scenarios and matrix types to compare performance.
"""

import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import psutil
import platform
import cpuinfo
import gc
import os
from memory_profiler import profile
from linearalgebra import (
    Matrix,
    naive_multiply,
    strassen_multiply,
    block_multiply,
    sparse_multiply,
    smart_multiply
)

def print_separator(title: str = ""):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print(f"\n{' ' + title + ' ':=^{width}}")
    else:
        print("=" * width)

def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()
    if hasattr(os, 'sync'):  # Unix/Linux/MacOS
        os.sync()
    
    # Try to release memory back to the OS
    try:
        import ctypes
        if hasattr(ctypes, 'CDLL'):
            if platform.system() == 'Linux':
                try:
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except:
                    pass
    except:
        pass

class MatrixBenchmark:
    def __init__(self):
        self.results = []  # Store results as a list of dictionaries
        self.output_dir = os.path.join('benchmarks', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.system_info = self._get_system_info()
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system information to terminal."""
        print_separator("System Information")
        for key, value in self.system_info.items():
            print(f"{key:15}: {value}")
        print_separator()
        
    def _get_system_info(self) -> Dict:
        """Gather system information."""
        cpu_info = cpuinfo.get_cpu_info()
        return {
            'OS': platform.system() + ' ' + platform.release(),
            'CPU': cpu_info['brand_raw'],
            'CPU Cores': psutil.cpu_count(logical=False),
            'CPU Threads': psutil.cpu_count(logical=True),
            'RAM': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'Python Version': platform.python_version()
        }

    def measure_memory(self, func, *args) -> Tuple[float, float]:
        """Measure memory usage of a function."""
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2
        
        result = func(*args)
        
        gc.collect()
        mem_after = process.memory_info().rss / 1024**2
        mem_used = mem_after - mem_before
        
        return result, mem_used

    def time_operation(self, func, *args, runs: int = 3) -> Tuple[float, float, float, float]:
        """Time an operation over multiple runs."""
        times = []
        memory_usage = []
        
        print(f"\nRunning {runs} iterations:")
        for i in range(runs):
            try:
                gc.collect()
                start_time = time.perf_counter()
                _, mem_used = self.measure_memory(func, *args)
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)
                memory_usage.append(mem_used)
                print(f"  Run {i+1}: {elapsed_time:.6f}s, Memory: {mem_used:.1f}MB")
                
                clear_memory()
            except Exception as e:
                print(f"  Run {i+1}: ERROR - {str(e)}")
                if i == 0:
                    raise
        
        if not times:
            return float('inf'), float('inf'), 0, 0
            
        return np.mean(times), np.min(times), np.std(times), np.mean(memory_usage)

    def create_dense_matrix(self, size: int) -> Matrix:
        """Create a dense random matrix."""
        return Matrix(np.random.rand(size, size))

    def create_sparse_matrix(self, size: int, sparsity: float) -> Matrix:
        """Create a sparse random matrix with given sparsity."""
        matrix = np.random.rand(size, size)
        mask = np.random.random((size, size)) < sparsity
        matrix[mask] = 0
        return Matrix(matrix)

    def create_band_matrix(self, size: int, bandwidth: int) -> Matrix:
        """Create a band matrix with given bandwidth."""
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(max(0, i-bandwidth), min(size, i+bandwidth+1)):
                matrix[i, j] = np.random.rand()
        return Matrix(matrix)

    def create_triangular_matrix(self, size: int, upper: bool = True) -> Matrix:
        """Create an upper or lower triangular matrix."""
        matrix = np.random.rand(size, size)
        if upper:
            matrix = np.triu(matrix)
        else:
            matrix = np.tril(matrix)
        return Matrix(matrix)

    def benchmark_matrix_operations(self, size: int, matrix_type: str, creator_func, **kwargs):
        """Benchmark all multiplication methods for a specific matrix type and size."""
        print_separator(f"Testing {matrix_type} Matrix ({size}x{size})")
        
        try:
            print("Generating test matrices...")
            A = creator_func(size, **kwargs)
            B = creator_func(size, **kwargs)
            
            methods = [
                ('Naive', naive_multiply),
                ('Strassen', strassen_multiply),
                ('Block', lambda a, b: block_multiply(a, b, block_size=32)),
                ('Sparse', sparse_multiply),
                ('Smart', smart_multiply)
            ]
            
            for method_name, method in methods:
                print(f"\nTesting {method_name} method:")
                try:
                    avg_time, best_time, std_dev, memory = self.time_operation(method, A, B)
                    gflops = (2 * size**3) / (best_time * 1e9) if best_time != float('inf') else 0
                    efficiency = gflops / memory if memory > 0 else 0
                    
                    # Add result to the list
                    self.results.append({
                        'matrix_type': matrix_type,
                        'matrix_size': size,
                        'method': method_name,
                        'avg_time': avg_time,
                        'best_time': best_time,
                        'std_dev': std_dev,
                        'gflops': gflops,
                        'memory_mb': memory,
                        'efficiency': efficiency,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    print("\nResults:")
                    print(f"  Average Time : {avg_time:.6f}s ± {std_dev:.6f}")
                    print(f"  Best Time    : {best_time:.6f}s")
                    print(f"  Performance  : {gflops:.2f} GFLOPS")
                    print(f"  Memory Usage : {memory:.1f}MB")
                    print(f"  Efficiency   : {efficiency:.2f} GFLOPS/MB")
                except Exception as e:
                    print(f"  ERROR: Could not test {method_name} method: {str(e)}")
                    self._add_error_result(matrix_type, size, method_name)
                
                clear_memory()
            
            del A
            del B
            clear_memory()
            
        except Exception as e:
            print(f"ERROR: Could not test {matrix_type} matrix of size {size}x{size}: {str(e)}")
            for method_name, _ in methods:
                self._add_error_result(matrix_type, size, method_name)

    def _add_error_result(self, matrix_type: str, size: int, method: str):
        """Add error results to the results list."""
        self.results.append({
            'matrix_type': matrix_type,
            'matrix_size': size,
            'method': method,
            'avg_time': float('nan'),
            'best_time': float('nan'),
            'std_dev': float('nan'),
            'gflops': float('nan'),
            'memory_mb': float('nan'),
            'efficiency': float('nan'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

    def run_comprehensive_benchmark(self):
        """Run benchmarks for all matrix types and sizes."""
        sizes = [10, 50, 100, 500, 1000, 2000]
        matrix_configs = [
            ('Dense', self.create_dense_matrix, {}),
            ('Sparse-90', self.create_sparse_matrix, {'sparsity': 0.9}),
            ('Sparse-95', self.create_sparse_matrix, {'sparsity': 0.95}),
            ('Band-5', self.create_band_matrix, {'bandwidth': 5}),
            ('Band-10', self.create_band_matrix, {'bandwidth': 10}),
            ('Upper-Triangular', self.create_triangular_matrix, {'upper': True}),
            ('Lower-Triangular', self.create_triangular_matrix, {'upper': False})
        ]
        
        for size in sizes:
            for matrix_type, creator_func, kwargs in matrix_configs:
                self.benchmark_matrix_operations(size, matrix_type, creator_func, **kwargs)
                
                # Save progress after each matrix type
                self.save_results()
        
        return pd.DataFrame(self.results)

    def save_results(self):
        """Save current results to CSV file."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.output_dir, 'benchmark_results.csv'), index=False)

    def generate_summary(self):
        """Generate a summary of the benchmark results."""
        results_df = pd.DataFrame(self.results)
        
        summary = []
        summary.append("Matrix Multiplication Benchmark Summary")
        summary.append("=" * 50 + "\n")
        
        # System Information
        summary.append("System Information")
        summary.append("-" * 30)
        for key, value in self.system_info.items():
            summary.append(f"{key}: {value}")
        summary.append("")
        
        # Best performing method for each matrix type and size
        summary.append("Best Performing Methods")
        summary.append("-" * 30)
        
        for matrix_type in results_df['matrix_type'].unique():
            summary.append(f"\n{matrix_type} Matrices:")
            type_data = results_df[results_df['matrix_type'] == matrix_type]
            
            for size in sorted(type_data['matrix_size'].unique()):
                size_data = type_data[type_data['matrix_size'] == size]
                best_idx = size_data['gflops'].idxmax()
                best_result = size_data.loc[best_idx]
                
                summary.append(f"\n  Size {size}x{size}:")
                summary.append(f"    Best Method: {best_result['method']}")
                summary.append(f"    Performance: {best_result['gflops']:.2f} GFLOPS")
                summary.append(f"    Memory Usage: {best_result['memory_mb']:.1f} MB")
                summary.append(f"    Efficiency: {best_result['efficiency']:.2f} GFLOPS/MB")
        
        # Write summary to file
        summary_path = os.path.join(self.output_dir, 'benchmark_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary))

def run_benchmarks():
    """Run all benchmarks and generate reports."""
    benchmark = MatrixBenchmark()
    
    print("\nStarting comprehensive benchmark suite...")
    print("This will test matrices up to 2000x2000")
    input("Press Enter to continue...")
    
    benchmark.run_comprehensive_benchmark()
    benchmark.generate_summary()
    
    print("\nBenchmark complete! Results have been saved to:")
    print(f"  - {os.path.join('benchmarks', 'output', 'benchmark_results.csv')} (Complete results)")
    print(f"  - {os.path.join('benchmarks', 'output', 'benchmark_summary.txt')} (Summary)")

if __name__ == "__main__":
    run_benchmarks() 