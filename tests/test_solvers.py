"""
Tests for the solvers module in the mathalgebra library.
"""

import pytest
import numpy as np
from linearalgebra.core import Matrix, Vector
from linearalgebra.solvers import solve_linear_system, gaussian_elimination

def test_solve_linear_system():
    equations = [
        "3x + 4y - 5z = 12",
        "2x - y + 3z = 7",
        "x + y + z = 6"
    ]
    
    solution = solve_linear_system(equations)
    
    # Expected solution: x=1, y=2, z=3
    assert pytest.approx(solution["x"], 0.001) == 1.0
    assert pytest.approx(solution["y"], 0.001) == 2.0
    assert pytest.approx(solution["z"], 0.001) == 3.0

def test_solve_linear_system_underdetermined():
    equations = [
        "x + y = 3",
        "2x + 2y = 6"
    ]
    
    with pytest.raises(ValueError, match="underdetermined"):
        solve_linear_system(equations)

def test_solve_linear_system_inconsistent():
    equations = [
        "x + y = 3",
        "x + y = 4"
    ]
    
    with pytest.raises(ValueError, match="singular"):
        solve_linear_system(equations)

def test_solve_linear_system_with_decimals():
    equations = [
        "1.5x + 2.5y = 10",
        "3.5x - 1.5y = 5"
    ]
    
    solution = solve_linear_system(equations)
    
    # Verify solution by substituting back
    x, y = solution["x"], solution["y"]
    assert pytest.approx(1.5*x + 2.5*y, 0.001) == 10.0
    assert pytest.approx(3.5*x - 1.5*y, 0.001) == 5.0

def test_gaussian_elimination():
    A = Matrix([
        [3, 4, -5],
        [2, -1, 3],
        [1, 1, 1]
    ])
    
    b = Vector([12, 7, 6])
    
    x = gaussian_elimination(A, b)
    
    # The expected solution is x=1, y=2, z=3
    assert pytest.approx(x.data[0], 0.001) == 1.0
    assert pytest.approx(x.data[1], 0.001) == 2.0
    assert pytest.approx(x.data[2], 0.001) == 3.0

def test_gaussian_elimination_singular():
    A = Matrix([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])
    
    b = Vector([6, 12, 18])
    
    with pytest.raises(ValueError, match="singular"):
        gaussian_elimination(A, b)

def test_gaussian_elimination_rectangular():
    A = Matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    b = Vector([7, 8])
    
    with pytest.raises(ValueError, match="square"):
        gaussian_elimination(A, b)

def test_gaussian_elimination_pivoting():
    # This matrix requires pivoting for numerical stability
    A = Matrix([
        [0.001, 2],
        [1, 1]
    ])
    
    b = Vector([1, 2])
    
    x = gaussian_elimination(A, b)
    
    # Verify solution: A * x = b
    result = A * x
    assert np.allclose(result.data, b.data) 