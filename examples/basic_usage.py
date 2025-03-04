"""
Basic usage examples for the mathalgebra library.
"""

from linearalgebra.utils import extract_coefficients
from linearalgebra.solvers import solve_linear_system
from linearalgebra.core import Matrix, Vector

def example_coefficient_extraction():
    print("Example 1: Extracting coefficients from a linear equation")
    eq = "3x + 4y - 5z = 12"
    coeffs = extract_coefficients(eq)
    print(f"Equation: {eq}")
    print(f"Coefficients: {coeffs}")
    print()

def example_solving_linear_system():
    print("Example 2: Solving a system of linear equations")
    equations = [
        "3x + 4y - 5z = 12",
        "2x - y + 3z = 7",
        "x + y + z = 6"
    ]
    
    print("System of equations:")
    for eq in equations:
        print(f"  {eq}")
    
    solution = solve_linear_system(equations)
    print("Solution:")
    for var, value in solution.items():
        print(f"  {var} = {value}")
    print()

def example_matrix_operations():
    print("Example 3: Matrix operations")
    
    # Create matrices
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # Addition
    C = A + B
    print(f"A + B =\n{C}")
    
    # Multiplication
    D = A * B
    print(f"A * B =\n{D}")
    
    # Transpose
    AT = A.transpose()
    print(f"A^T =\n{AT}")
    
    # Determinant
    det_A = A.determinant()
    print(f"det(A) = {det_A}")
    print()

def example_vector_operations():
    print("Example 4: Vector operations")
    
    # Create vectors
    v = Vector([1, 2, 3])
    w = Vector([4, 5, 6])
    
    print(f"Vector v: {v}")
    print(f"Vector w: {w}")
    
    # Addition
    sum_vw = v + w
    print(f"v + w = {sum_vw}")
    
    # Dot product
    dot_vw = v.dot(w)
    print(f"v · w = {dot_vw}")
    
    # Norm
    norm_v = v.norm()
    print(f"||v|| = {norm_v}")
    
    # Normalization
    unit_v = v.normalize()
    print(f"v̂ = {unit_v}")
    print(f"||v̂|| = {unit_v.norm()}")  # Should be close to 1
    print()

if __name__ == "__main__":
    example_coefficient_extraction()
    example_solving_linear_system()
    example_matrix_operations()
    example_vector_operations() 