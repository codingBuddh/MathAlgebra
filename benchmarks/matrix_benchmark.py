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

class MatrixBenchmark:
    def __init__(self):
        self.results = {}
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

    def time_operation(self, func, *args, runs: int = 5) -> Tuple[float, float, float, float]:
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
            except Exception as e:
                print(f"  Run {i+1}: ERROR - {str(e)}")
                if i == 0:  # If first run fails, we can't continue
                    raise
        
        if not times:  # If all runs failed
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

    def benchmark_size_scaling(self, sizes: List[int]):
        """Benchmark how different methods scale with matrix size."""
        print_separator("Size Scaling Benchmark")
        
        results = {
            'size': [],
            'method': [],
            'avg_time': [],
            'best_time': [],
            'std_dev': [],
            'gflops': [],
            'memory_mb': [],
            'efficiency': []
        }
        
        methods = [
            ('Naive', naive_multiply),
            ('Strassen', strassen_multiply),
            ('Block', lambda a, b: block_multiply(a, b, block_size=32)),
            ('Smart', smart_multiply)
        ]
        
        for size in sizes:
            print_separator(f"Matrix Size: {size}x{size}")
            
            # Skip naive for large matrices
            if size > 500:
                methods_to_test = [m for m in methods if m[0] != 'Naive']
            else:
                methods_to_test = methods
                
            try:
                print(f"Generating random matrices...")
                A = self.create_dense_matrix(size)
                B = self.create_dense_matrix(size)
                
                for name, method in methods_to_test:
                    print(f"\nTesting {name} method:")
                    try:
                        avg_time, best_time, std_dev, memory = self.time_operation(method, A, B)
                        gflops = (2 * size**3) / (best_time * 1e9) if best_time != float('inf') else 0
                        efficiency = gflops / memory if memory > 0 else 0
                        
                        results['size'].append(size)
                        results['method'].append(name)
                        results['avg_time'].append(avg_time)
                        results['best_time'].append(best_time)
                        results['std_dev'].append(std_dev)
                        results['gflops'].append(gflops)
                        results['memory_mb'].append(memory)
                        results['efficiency'].append(efficiency)
                        
                        print("\nResults:")
                        print(f"  Average Time : {avg_time:.6f}s ± {std_dev:.6f}")
                        print(f"  Best Time    : {best_time:.6f}s")
                        print(f"  Performance  : {gflops:.2f} GFLOPS")
                        print(f"  Memory Usage : {memory:.1f}MB")
                        print(f"  Efficiency   : {efficiency:.2f} GFLOPS/MB")
                    except Exception as e:
                        print(f"  ERROR: Could not test {name} method: {str(e)}")
                        # Still add to results with NaN values
                        results['size'].append(size)
                        results['method'].append(name)
                        results['avg_time'].append(float('nan'))
                        results['best_time'].append(float('nan'))
                        results['std_dev'].append(float('nan'))
                        results['gflops'].append(float('nan'))
                        results['memory_mb'].append(float('nan'))
                        results['efficiency'].append(float('nan'))
            except Exception as e:
                print(f"ERROR: Could not test size {size}x{size}: {str(e)}")
        
        self.results['size_scaling'] = pd.DataFrame(results)

    def benchmark_matrix_types(self, size: int = 1000):
        """Benchmark different matrix types."""
        print_separator("Matrix Types Benchmark")
        
        results = {
            'type': [],
            'method': [],
            'avg_time': [],
            'best_time': [],
            'std_dev': [],
            'memory_mb': [],
            'gflops': []
        }
        
        matrix_types = [
            ('Dense', lambda: self.create_dense_matrix(size)),
            ('Sparse-90', lambda: self.create_sparse_matrix(size, 0.9)),
            ('Sparse-95', lambda: self.create_sparse_matrix(size, 0.95)),
            ('Band-5', lambda: self.create_band_matrix(size, 5)),
            ('Band-10', lambda: self.create_band_matrix(size, 10)),
            ('Upper-Triangular', lambda: self.create_triangular_matrix(size, True)),
            ('Lower-Triangular', lambda: self.create_triangular_matrix(size, False))
        ]
        
        methods = [
            ('Naive', naive_multiply),
            ('Strassen', strassen_multiply),
            ('Block', lambda a, b: block_multiply(a, b, block_size=32)),
            ('Sparse', sparse_multiply),
            ('Smart', smart_multiply)
        ]
        
        for type_name, matrix_creator in matrix_types:
            print_separator(f"Testing {type_name} Matrices")
            print("Generating test matrices...")
            A = matrix_creator()
            B = matrix_creator()
            
            for method_name, method in methods:
                print(f"\nTesting {method_name} method:")
                avg_time, best_time, std_dev, memory = self.time_operation(method, A, B)
                gflops = (2 * size**3) / (best_time * 1e9)
                
                results['type'].append(type_name)
                results['method'].append(method_name)
                results['avg_time'].append(avg_time)
                results['best_time'].append(best_time)
                results['std_dev'].append(std_dev)
                results['memory_mb'].append(memory)
                results['gflops'].append(gflops)
                
                print("\nResults:")
                print(f"  Average Time : {avg_time:.6f}s ± {std_dev:.6f}")
                print(f"  Best Time    : {best_time:.6f}s")
                print(f"  Performance  : {gflops:.2f} GFLOPS")
                print(f"  Memory Usage : {memory:.1f}MB")
        
        self.results['matrix_types'] = pd.DataFrame(results)

    def plot_results(self):
        """Plot benchmark results with enhanced visualizations."""
        # Size scaling plot with memory usage
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        for method in ['Naive', 'Strassen', 'Block', 'Smart']:
            data = self.results['size_scaling'][self.results['size_scaling']['method'] == method]
            ax1.plot(data['size'], data['gflops'], marker='o', label=f"{method}")
            ax2.plot(data['size'], data['memory_mb'], marker='s', label=f"{method}")
            ax3.plot(data['size'], data['efficiency'], marker='^', label=f"{method}")
        
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('GFLOPS')
        ax1.set_title('Performance Scaling with Matrix Size')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Matrix Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage with Matrix Size')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Efficiency (GFLOPS/MB)')
        ax3.set_title('Memory Efficiency with Matrix Size')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('benchmark_scaling_combined.png')
        
        # Matrix types comparison
        plt.figure(figsize=(15, 8))
        data = self.results['matrix_types']
        types = data['type'].unique()
        methods = data['method'].unique()
        
        x = np.arange(len(types))
        width = 0.15  # Adjusted for more methods
        
        for i, method in enumerate(methods):
            method_data = data[data['method'] == method]
            times = [method_data[method_data['type'] == t]['best_time'].iloc[0] for t in types]
            plt.bar(x + i*width, times, width, label=method)
        
        plt.xlabel('Matrix Type')
        plt.ylabel('Best Time (seconds)')
        plt.title('Performance Comparison Across Matrix Types')
        plt.xticks(x + width*2.5, types, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('benchmark_matrix_types.png')

    def generate_report(self):
        """Generate a detailed benchmark report with system information."""
        report = []
        report.append("Matrix Multiplication Benchmark Report")
        report.append("=" * 50 + "\n")
        
        # System Information
        report.append("System Information")
        report.append("-" * 30)
        for key, value in self.system_info.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Size scaling summary
        report.append("1. Size Scaling Performance")
        report.append("-" * 30)
        size_data = self.results['size_scaling']
        for method in size_data['method'].unique():
            method_data = size_data[size_data['method'] == method]
            max_gflops = method_data['gflops'].max()
            best_size = method_data.loc[method_data['gflops'].idxmax(), 'size']
            avg_memory = method_data['memory_mb'].mean()
            max_efficiency = method_data['efficiency'].max()
            
            report.append(f"{method} Method:")
            report.append(f"  - Peak Performance: {max_gflops:.2f} GFLOPS")
            report.append(f"  - Best at size: {best_size}x{best_size}")
            report.append(f"  - Average Memory Usage: {avg_memory:.1f} MB")
            report.append(f"  - Peak Efficiency: {max_efficiency:.2f} GFLOPS/MB\n")
        
        # Matrix types summary
        report.append("2. Matrix Types Performance")
        report.append("-" * 30)
        type_data = self.results['matrix_types']
        for matrix_type in type_data['type'].unique():
            report.append(f"\n{matrix_type} Matrices:")
            type_results = type_data[type_data['type'] == matrix_type]
            for _, row in type_results.iterrows():
                report.append(f"  {row['method']}:")
                report.append(f"    - Best Time: {row['best_time']:.6f} seconds")
                report.append(f"    - Memory Usage: {row['memory_mb']:.1f} MB")
                report.append(f"    - GFLOPS: {row['gflops']:.2f}")
        
        # Write report to file
        with open('benchmark_report.txt', 'w') as f:
            f.write('\n'.join(report))

def run_benchmarks():
    """Run all benchmarks and generate report."""
    benchmark = MatrixBenchmark()
    
    # Test different sizes up to 10000x10000
    sizes = [10, 50, 100, 500, 1000, 2000, 5000]
    
    # Only add 10000 if we have enough memory
    try:
        # Test if we can allocate a 5000x5000 matrix
        test = np.zeros((5000, 5000))
        del test
        sizes.append(10000)
    except MemoryError:
        print("Warning: Not enough memory for 10000x10000 matrices, skipping that size")
    
    print("\nBenchmark sizes to test:", sizes)
    print("Note: Testing will take longer for larger matrices.")
    input("Press Enter to continue...")
    
    benchmark.benchmark_size_scaling(sizes)
    
    print_separator("Matrix Types Benchmark")
    print("Testing different matrix types with size 1000x1000")
    input("Press Enter to continue...")
    
    benchmark.benchmark_matrix_types(size=1000)
    
    print_separator("Generating Reports")
    benchmark.plot_results()
    benchmark.generate_report()
    
    for name, df in benchmark.results.items():
        filename = f'benchmark_{name}.csv'
        df.to_csv(filename)
        print(f"Saved results to {filename}")
    
    print("\nBenchmark complete! Results have been saved to:")
    print("  - benchmark_report.txt (Detailed analysis)")
    print("  - benchmark_scaling_combined.png (Performance plots)")
    print("  - benchmark_matrix_types.png (Matrix types comparison)")
    print("  - benchmark_*.csv (Raw data)")

if __name__ == "__main__":
    run_benchmarks() 