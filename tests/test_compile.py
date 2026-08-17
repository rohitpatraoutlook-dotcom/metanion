"""
Test the JIT compiler.
Run from project root: python tests/test_compile.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metanion import OpID, intern, compile_handle
from metanion.compile import StraightLineProgram


def test_compilation():
    """Test compiling expressions to functions."""
    print("\n=== Test 5: JIT Compilation ===")
    
    # Create expression: x^2 + 1
    x = intern(OpID.IDENTITY)
    square = intern(OpID.SQUARE, x)
    one = intern(OpID.CONST_ONE)
    expr = intern(OpID.ADD, square, one)
    
    print(f"Expression handle: {expr}")
    
    # Compile to function
    func = compile_handle(expr)
    print(f"Compiled function: {func}")
    
    # Test with input
    result = func([3.0])
    print(f"f(3) = {result}")
    assert abs(result - 10.0) < 0.001, "Compilation gave wrong result"
    
    # Test with multiple values
    for x_val in [0, 1, 2, 3, 4]:
        result = func([float(x_val)])
        expected = x_val ** 2 + 1
        print(f"f({x_val}) = {result}, expected: {expected}")
        assert abs(result - expected) < 0.001, f"Failed at x={x_val}"
    
    print(" Compilation tests passed!")


def test_slp():
    """Test Straight-Line Program generation."""
    print("\n=== Test 6: SLP Generation ===")
    
    # Create expression: sin(x) + cos(x)
    x = intern(OpID.IDENTITY)
    sin_x = intern(OpID.SIN, x)
    cos_x = intern(OpID.COS, x)
    expr = intern(OpID.ADD, sin_x, cos_x)
    
    program = StraightLineProgram(expr)
    print(f"SLP:\n{program}")
    
    # Evaluate
    result = program.evaluate([0.0])
    print(f"sin(0) + cos(0) = {result}")
    assert abs(result - 1.0) < 0.001, "SLP evaluation failed"
    
    print(" SLP tests passed!")


if __name__ == "__main__":
    test_compilation()
    test_slp()
    print("\n Compilation tests completed successfully!")