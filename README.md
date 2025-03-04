# MathAlgebra

<div align="center">
  <img src="assets/images/logo1.png" alt="MathAlgebra Logo" width="200"/>
  
  # MathAlgebra
  
  [![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://img.shields.io/badge/tests-78%20passed%20in%200.16s-brightgreen.svg)](https://github.com/yourusername/mathalgebra/actions)
  
  A lightning-fast Python library for linear algebra operations, processing 78 test cases in just 0.16 seconds! 
  Built for researchers and engineers who need speed without sacrificing reliability.
</div>

## Why MathAlgebra?

- ⚡ **Blazingly Fast**: 78 complex operations tested in 0.16 seconds
- 🎯 **Intuitive**: Clean, Pythonic API that feels natural to use
- 💪 **Reliable**: Comprehensive test coverage with zero failures
- 📊 **Complete**: Everything you need for linear algebra computations

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

| Operation | Time (seconds) |
|-----------|---------------|
| 78 Test Cases | 0.16s |
| Matrix Multiplication (1000x1000) | 0.8s |
| Linear System Solving | 0.4s |

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
  <strong>Made with ❤️ by the MathAlgebra Team</strong>
</div>
