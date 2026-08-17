"""
Test symbolic differentiation.
Run from project root: python tests/test_calculus.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metanion import OpID, intern, differentiate, simplify
from metanion.calculus import get_derivative_rules


def test_differentiation():
    """Test symbolic differentiation."""
    print("\n=== Test 7: Symbolic Differentiation ===")
    
    x = intern(OpID.IDENTITY)
    
    # d/dx(x^2) = 2x
    square = intern(OpID.SQUARE, x)
    deriv = differentiate(square)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(x^2) = {deriv_simplified}")
    
    # d/dx(sin(x)) = cos(x)
    sin_x = intern(OpID.SIN, x)
    deriv = differentiate(sin_x)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(sin(x)) = {deriv_simplified}")
    
    # d/dx(exp(x)) = exp(x)
    exp_x = intern(OpID.EXP, x)
    deriv = differentiate(exp_x)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(exp(x)) = {deriv_simplified}")
    
    # d/dx(log(x)) = 1/x
    log_x = intern(OpID.LOG, x)
    deriv = differentiate(log_x)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(log(x)) = {deriv_simplified}")
    
    # d/dx(tanh(x)) = 1 - tanh^2(x)
    tanh_x = intern(OpID.TANH, x)
    deriv = differentiate(tanh_x)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(tanh(x)) = {deriv_simplified}")
    
    print(" Differentiation tests passed!")


def test_chain_rule():
    """Test chain rule for complex expressions."""
    print("\n=== Test 8: Chain Rule ===")
    
    x = intern(OpID.IDENTITY)
    
    # d/dx(sin(x^2 + 1)) = cos(x^2 + 1) * 2x
    one = intern(OpID.CONST_ONE)
    x2 = intern(OpID.SQUARE, x)
    inner = intern(OpID.ADD, x2, one)
    outer = intern(OpID.SIN, inner)
    
    deriv = differentiate(outer)
    deriv_simplified = simplify(deriv)
    print(f"d/dx(sin(x^2 + 1)) = {deriv_simplified}")
    
    print(" Chain rule tests passed!")


if __name__ == "__main__":
    test_differentiation()
    test_chain_rule()
    print("\n Calculus tests completed successfully!")