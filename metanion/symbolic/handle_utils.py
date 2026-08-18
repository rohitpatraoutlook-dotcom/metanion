"""Handle utilities for Metanion."""

from typing import Optional, Callable, List, Set, Tuple
from .expression_node import ExpressionNode


def get_depth(handle: int, lookup_func: Callable[[int], Optional[Tuple]]) -> int:
    """Get depth of expression tree."""
    node = lookup_func(handle)
    if node is None:
        return 0
    
    # node is a tuple: (op, left, right)
    if len(node) == 1:  # nullary
        return 1
    
    if len(node) == 2:  # unary
        child = node[1]
        if child is not None:
            return 1 + get_depth(child, lookup_func)
        return 1
    
    if len(node) == 3:  # binary
        left_depth = get_depth(node[1], lookup_func) if node[1] is not None else 0
        right_depth = get_depth(node[2], lookup_func) if node[2] is not None else 0
        return 1 + max(left_depth, right_depth)
    
    return 1


def count_nodes_in_subtree(handle: int, lookup_func: Callable[[int], Optional[Tuple]]) -> int:
    """Count nodes in expression tree."""
    node = lookup_func(handle)
    if node is None:
        return 0
    
    total = 1
    if len(node) >= 2 and node[1] is not None:
        total += count_nodes_in_subtree(node[1], lookup_func)
    if len(node) >= 3 and node[2] is not None:
        total += count_nodes_in_subtree(node[2], lookup_func)
    
    return total


def simplify_handle(handle: int, lookup_func: Callable, create_func: Callable) -> int:
    """Simplify an expression handle."""
    node = lookup_func(handle)
    if node is None:
        return handle
    
    # node is a tuple: (op, left, right)
    # Recursively simplify children
    left_simplified = None
    right_simplified = None
    
    if len(node) >= 2 and node[1] is not None:
        left_simplified = simplify_handle(node[1], lookup_func, create_func)
    
    if len(node) >= 3 and node[2] is not None:
        right_simplified = simplify_handle(node[2], lookup_func, create_func)
    
    # If nothing changed, return original
    if left_simplified == node[1] and right_simplified == node[2]:
        return handle
    
    # Create simplified node
    op = node[0]
    if len(node) == 1:
        return create_func(op, None, None)
    elif len(node) == 2:
        return create_func(op, left_simplified, None)
    else:
        return create_func(op, left_simplified, right_simplified)
