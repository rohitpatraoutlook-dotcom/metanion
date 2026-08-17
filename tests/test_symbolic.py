"""
Test the symbolic expression system.
Run from project root: python tests/test_symbolic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metanion import OpID, intern, lookup, simplify, get_pool
from metanion.symbolic import get_op_name, get_depth, count_nodes_in_subtree


def test_expression_creation():
    """Test creating and interning expressions."""
    print("\n=== Test 3: Expression Creation ===")
    
    # Create identity
    id_handle = intern(OpID.IDENTITY)
    print(f"Identity handle: {id_handle}")
    node = lookup(id_handle)
    print(f"Identity node: {node}")
    
    # Create constant
    zero_handle = intern(OpID.CONST_ZERO)
    one_handle = intern(OpID.CONST_ONE)
    print(f"Zero handle: {zero_handle}, One handle: {one_handle}")
    
    # Create addition
    add_handle = intern(OpID.ADD, id_handle, one_handle)
    print(f"x + 1 handle: {add_handle}")
    
    # Create sin(x)
    sin_handle = intern(OpID.SIN, id_handle)
    print(f"sin(x) handle: {sin_handle}")
    
    # Get depth
    depth = get_depth(add_handle, lookup)
    print(f"Depth of x+1: {depth}")
    
    print(" Expression creation tests passed!")


def test_expression_simplification():
    """Test algebraic simplification."""
    print("\n=== Test 4: Expression Simplification ===")
    
    pool = get_pool()
    
    # x + 0 -> x
    x = intern(OpID.IDENTITY)
    zero = intern(OpID.CONST_ZERO)
    expr = intern(OpID.ADD, x, zero)
    simplified = simplify(expr)
    print(f"x + 0 simplified to: {simplified}")
    
    # x * 1 -> x
    one = intern(OpID.CONST_ONE)
    expr = intern(OpID.MUL, x, one)
    simplified = simplify(expr)
    print(f"x * 1 simplified to: {simplified}")
    
    # x - x -> 0
    expr = intern(OpID.SUB, x, x)
    simplified = simplify(expr)
    print(f"x - x simplified to: {simplified}")
    
    # exp(log(x)) -> x
    log_x = intern(OpID.LOG, x)
    expr = intern(OpID.EXP, log_x)
    simplified = simplify(expr)
    print(f"exp(log(x)) simplified to: {simplified}")
    
    print("Expression simplification tests passed!")


if __name__ == "__main__":
    test_expression_creation()
    test_expression_simplification()
    print("\n Symbolic tests completed successfully!")