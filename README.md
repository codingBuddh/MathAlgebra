# MathAlgebra

<div align="center">
  <img src="assets/images/logo.png" alt="MathAlgebra Logo" width="200"/>
  
  # MathAlgebra
  
  [![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen.svg)](https://github.com/yourusername/mathalgebra/actions)
  
  A fast and intuitive Python library for linear algebra operations, built for researchers and engineers.
</div>

## Why MathAlgebra?

- 🚀 **Fast**: Built on NumPy with optimized implementations
- 🎯 **Intuitive**: Clean, Pythonic API that feels natural to use
- 💪 **Reliable**: Comprehensive test suite with 78 test cases
- 📊 **Complete**: Everything you need for linear algebra computations

## Quick Start

```bash
pip install mathalgebra
```

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

# Solve linear equations
solution = Matrix.solve([
    [2, 1, -1],
    [3, -2, 1],
    [-1, 3, -2]
], [8, -11, -3])
```

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

## Author

- [@Aman Soni](https://www.linkedin.com/in/aman-soni-6b17b6223/)



## License

MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Made with ❤️ by the MathAlgebra Team</strong>
</div>
