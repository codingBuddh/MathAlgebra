"""
Tests for the package structure and imports.
"""

import pytest
import importlib
import re
import numpy as np

def test_version():
    # Import the package
    import linearalgebra
    
    # Check that version is defined
    assert hasattr(linearalgebra, '__version__')
    
    # Check version format (should be like x.y.z)
    version_pattern = r'^\d+\.\d+\.\d+$'
    assert re.match(version_pattern, linearalgebra.__version__)

def test_imports():
    # Test that all modules can be imported
    modules = [
        'linearalgebra.core',
        'linearalgebra.utils',
        'linearalgebra.solvers',
        'linearalgebra.decompositions'
    ]
    
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None

def test_readme_examples():
    # Import necessary modules
    from linearalgebra.core import Matrix, Vector
    from linearalgebra.solvers import solve_linear_system
    
    # Example 1: Matrix operations
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    C = A + B
    assert np.array_equal(C.data, np.array([[6, 8], [10, 12]]))
    
    D = A * B
    assert np.array_equal(D.data, np.array([[19, 22], [43, 50]]))
    
    # Example 2: Solving a system of linear equations
    equations = [
        "3x + 4y - 5z = 12",
        "2x - y + 3z = 7",
        "x + y + z = 6"
    ]
    
    solution = solve_linear_system(equations)
    assert pytest.approx(solution["x"], 0.001) == 1.0
    assert pytest.approx(solution["y"], 0.001) == 2.0
    assert pytest.approx(solution["z"], 0.001) == 3.0 