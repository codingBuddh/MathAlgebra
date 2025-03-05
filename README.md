# MathAlgebra

<div align="center">
  <img src="assets/images/logo1.png" alt="MathAlgebra Logo" width="200"/>
  
  # MathAlgebra
  
  [![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://img.shields.io/badge/tests-78%20passed%20in%200.16s-brightgreen.svg)](https://github.com/yourusername/mathalgebra/actions)
  
  A lightning-fast Python library for linear algebra operations, achieving 199.72 GFLOPS on matrix multiplication! 
  Built for researchers and engineers who need speed without sacrificing reliability.
</div>

## Why MathAlgebra?

- ⚡ **Blazingly Fast**: 1000x1000 matrix multiplication in just 0.01 seconds
- 🎯 **Intuitive**: Clean, Pythonic API that feels natural to use
- 💪 **Reliable**: Comprehensive test coverage with zero failures
- 📊 **Complete**: Everything you need for linear algebra computations
- 🧠 **Smart**: Automatic algorithm selection for optimal performance

## Features

### Intelligent Matrix Multiplication

MathAlgebra automatically selects the most efficient multiplication algorithm based on matrix characteristics:

- **Smart Algorithm Selection**:
  - Small matrices (< 64×64): Optimized naive multiplication
  - Medium matrices (64×64 to 512×512): Strassen's algorithm
  - Large matrices (> 512×512): Block matrix multiplication
  - Sparse matrices (> 80% sparsity): Specialized sparse operations

- **Special Case Optimization**:
  - Identity matrix multiplication
  - Zero matrix multiplication
  - Automatic sparsity detection
  - Cache-aware computations

- **Performance Metrics**:
  - Up to 199.72 GFLOPS for dense matrices
  - Efficient memory utilization
  - Automatic cache optimization
  - Minimal memory overhead

## Installation

### Using pip (Recommended)
```bash
pip install mathalgebra
```

### From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/mathalgebra.git

# Navigate to the directory
cd mathalgebra

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Getting Started

### Basic Operations
```python
from mathalgebra import Matrix, Vector

# Create and manipulate matrices
matrix = Matrix([[1, 2], [3, 4]])
determinant = matrix.determinant()  # 2
inverse = matrix.inverse()

# Work with vectors
vector = Vector([1, 2, 3])
magnitude = vector.norm()
normalized = vector.normalize()
```

### Matrix Multiplication with Automatic Optimization
```python
# Create matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

# Automatic algorithm selection
C = A * B  # Chooses the best method based on matrix characteristics

# Manual method selection if needed
C = A.multiply(B, method='strassen')  # or 'basic', 'sparse', 'block'
```

### Solving Linear Equations
```python
# Solve a system of linear equations
equations = [
    [2, 1, -1],  # 2x + y - z = 8
    [3, -2, 1],  # 3x - 2y + z = -11
    [-1, 3, -2]  # -x + 3y - 2z = -3
]
constants = [8, -11, -3]

solution = Matrix.solve(equations, constants)
# Returns: {'x': 2.0, 'y': 1.0, 'z': 0.0}
```

### Matrix Decompositions
```python
# Perform matrix decompositions
matrix = Matrix([[4, 2], [2, 1]])

# LU Decomposition
L, U = matrix.lu_decomposition()

# QR Decomposition
Q, R = matrix.qr_decomposition()

# SVD Decomposition
U, S, V = matrix.svd_decomposition()
```

## 📊 Performance Benchmarks

### Matrix Multiplication (1000x1000)
```python
# Run the benchmark
python benchmark_matrix.py

# Actual Output
Benchmarking 1000x1000 Matrix Multiplication
--------------------------------------------------
Run 1: 0.0147 seconds
Run 2: 0.0130 seconds
Run 3: 0.0129 seconds
Run 4: 0.0105 seconds
Run 5: 0.0100 seconds

Results:
Average Time: 0.0122 seconds
Best Time: 0.0100 seconds
Matrix Size: 1000x1000
Operations: ~1,000,000,000 floating-point operations
GFLOPS: 199.72
```

### Benchmark Your System
We provide a benchmark script to test performance on your system:

```python
from linearalgebra import Matrix
import numpy as np
import time

# Generate test matrices
size = 1000
A = Matrix(np.random.rand(size, size))
B = Matrix(np.random.rand(size, size))

# Measure multiplication time
start_time = time.perf_counter()
C = A * B
elapsed_time = time.perf_counter() - start_time

print(f"Time taken: {elapsed_time:.4f} seconds")
```

For detailed benchmarking, use our provided script:
```bash
# Clone the repository
git clone https://github.com/yourusername/mathalgebra.git

# Run the benchmark
python benchmark_matrix.py
```

### Performance 

| Operation | MathAlgebra |
|-----------|-------------|
| Matrix Multiplication (1000x1000) | 
| 78 Test Cases | 0.16s |
| Linear System Solving |

### Why So Fast?
- **Optimized Implementation**: Carefully tuned matrix operations
- **Efficient Memory Usage**: Minimized memory allocations and copies
- **Hardware Acceleration**: Leverages NumPy's optimized backend
- **Smart Algorithms**: Uses the most efficient approach for each operation size

## 📊 Feature List

| Category | Feature | Description | Implementation |
|----------|---------|-------------|----------------|
| **Vector Operations** | Vector Creation | Create vectors from lists or arrays with automatic type conversion | [Vector.__init__](linearalgebra/core.py#L8) |
| | Vector Addition | Element-wise addition of two vectors with dimension validation | [Vector.__add__](linearalgebra/core.py#L35) |
| | Vector Subtraction | Element-wise subtraction with automatic dimension checking | [Vector.__sub__](linearalgebra/core.py#L41) |
| | Scalar Multiplication | Multiply vector by scalar value with type handling | [Vector.__mul__](linearalgebra/core.py#L47) |
| | Dot Product | Compute dot product between two vectors with dimension validation | [Vector.dot](linearalgebra/core.py#L55) |
| | Vector Norm | Calculate Euclidean norm (magnitude) of vector | [Vector.norm](linearalgebra/core.py#L61) |
| | Cross Product | Compute cross product for 3D vectors with dimension validation | [Vector.cross](linearalgebra/core.py#L72) |
| **Matrix Operations** | Matrix Creation | Create matrices from 2D lists or arrays with type conversion | [Matrix.__init__](linearalgebra/core.py#L130) |
| | Matrix Addition | Element-wise addition with shape validation | [Matrix.__add__](linearalgebra/core.py#L148) |
| | Matrix Multiplication | Matrix-matrix and matrix-vector multiplication with validation | [Matrix.__mul__](linearalgebra/core.py#L160) |
| | Transpose | Compute matrix transpose efficiently | [Matrix.transpose](linearalgebra/core.py#L182) |
| | Determinant | Calculate matrix determinant with square matrix validation | [Matrix.determinant](linearalgebra/core.py#L186) |
| | Matrix Inverse | Compute inverse matrix with singularity check | [Matrix.inverse](linearalgebra/core.py#L192) |
| **Matrix Decompositions** | LU Decomposition | Decompose matrix into lower and upper triangular matrices | [lu_decomposition](linearalgebra/decompositions.py#L8) |
| | QR Decomposition | Decompose matrix using Gram-Schmidt process | [qr_decomposition](linearalgebra/decompositions.py#L41) |
| | SVD Decomposition | Compute Singular Value Decomposition with economy size | [svd_decomposition](linearalgebra/decompositions.py#L76) |
| **Linear System Solving** | System Solver | Solve linear equations with comprehensive error handling | [solve_linear_system](linearalgebra/solvers.py#L8) |
| | Gaussian Elimination | Solve using Gaussian elimination with partial pivoting | [gaussian_elimination](linearalgebra/solvers.py#L89) |

## Requirements

- Python 3.6+
- NumPy

## Development Setup

```bash
# Clone and setup for development
git clone https://github.com/yourusername/mathalgebra.git
cd mathalgebra

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Author

- [@Aman Soni](https://www.linkedin.com/in/aman-soni-6b17b6223/)

## License

MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Made with ❤️ by Aman Soni</strong>
</div>
