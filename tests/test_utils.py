"""
Tests for the utils module in the mathalgebra library.
"""

import pytest
from linearalgebra.utils import extract_coefficients, parse_system

def test_extract_coefficients_basic():
    eq = "3x + 4y - 5z = 12"
    result = extract_coefficients(eq)
    expected = {"x": 3, "y": 4, "z": -5, "constant": 12}
    assert result == expected

def test_extract_coefficients_no_coefficients():
    eq = "x - y + z = 10"
    result = extract_coefficients(eq)
    expected = {"x": 1, "y": -1, "z": 1, "constant": 10}
    assert result == expected

def test_extract_coefficients_negative_constant():
    eq = "-2a + 3b - 4c = -10"
    result = extract_coefficients(eq)
    expected = {"a": -2, "b": 3, "c": -4, "constant": -10}
    assert result == expected

def test_extract_coefficients_no_constant():
    eq = "2x + 3y = 0"
    result = extract_coefficients(eq)
    expected = {"x": 2, "y": 3, "constant": 0}
    assert result == expected

def test_extract_coefficients_no_equals():
    eq = "2x + 3y - 4z"
    result = extract_coefficients(eq)
    expected = {"x": 2, "y": 3, "z": -4, "constant": 0}
    assert result == expected

def test_extract_coefficients_decimal():
    eq = "1.5x - 2.25y = 3.75"
    result = extract_coefficients(eq)
    expected = {"x": 1.5, "y": -2.25, "constant": 3.75}
    assert result == expected

def test_extract_coefficients_right_side_variables():
    eq = "2x + 3y = 4z + 5"
    result = extract_coefficients(eq)
    expected = {"x": 2, "y": 3, "z": -4, "constant": 5}
    assert result == expected

def test_extract_coefficients_all_on_one_side():
    eq = "2x + 3y - 4z - 5 = 0"
    result = extract_coefficients(eq)
    expected = {"x": 2, "y": 3, "z": -4, "constant": -5}
    assert result == expected

def test_extract_coefficients_complex_equation():
    eq = "2.5x - 3.5y + 4.5z - 5.5 = 1.5x + 2.5z + 3.5"
    result = extract_coefficients(eq)
    
    # Recalculate the expected result
    # Left side: 2.5x - 3.5y + 4.5z - 5.5
    # Right side: 1.5x + 2.5z + 3.5
    # After rearranging: (2.5-1.5)x - 3.5y + (4.5-2.5)z - 5.5 - 3.5 = 0
    # Which gives: 1.0x - 3.5y + 2.0z - 9.0 = 0
    expected = {"x": 1.0, "y": -3.5, "z": 2.0, "constant": -8.0}
    
    assert result == expected

def test_parse_system():
    equations = [
        "x + y + z = 6",
        "2x - y + z = 3",
        "x + 2y - z = 4"
    ]
    coeffs, vars = parse_system(equations)
    
    expected_coeffs = [
        {"x": 1, "y": 1, "z": 1, "constant": 6},
        {"x": 2, "y": -1, "z": 1, "constant": 3},
        {"x": 1, "y": 2, "z": -1, "constant": 4}
    ]
    expected_vars = ["x", "y", "z"]
    
    assert coeffs == expected_coeffs
    assert vars == expected_vars

def test_parse_system_different_variables():
    equations = [
        "a + b = 3",
        "c + d = 4"
    ]
    coeffs, vars = parse_system(equations)
    
    expected_coeffs = [
        {"a": 1, "b": 1, "constant": 3},
        {"c": 1, "d": 1, "constant": 4}
    ]
    expected_vars = ["a", "b", "c", "d"]
    
    assert coeffs == expected_coeffs
    assert sorted(vars) == sorted(expected_vars)

def test_parse_system_missing_variables():
    equations = [
        "x + z = 5",
        "y - z = 2"
    ]
    coeffs, vars = parse_system(equations)
    
    expected_coeffs = [
        {"x": 1, "z": 1, "constant": 5},
        {"y": 1, "z": -1, "constant": 2}
    ]
    expected_vars = ["x", "y", "z"]
    
    assert coeffs == expected_coeffs
    assert sorted(vars) == sorted(expected_vars) 