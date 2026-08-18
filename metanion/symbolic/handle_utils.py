"""Handle utilities for Metanion - with recursion safety limit."""

from typing import Optional, Callable, List, Set, Tuple
from .expression_node import ExpressionNode


def get_depth(handle: int, lookup_func: Callable[[int], Optional[Tuple]], max_depth: int = 50) -> int:
    """
    Get depth of expression tree with a safety limit to prevent RecursionError.
    
    Args:
        handle: The handle to get depth of.
        lookup_func: Function to lookup a handle.
        max_depth: Maximum depth to traverse (prevents infinite recursion).
    
    Returns:
        Depth of the expression tree.
    """
    if max_depth <= 0:
        return max_depth
    
    node = lookup_func(handle)
    if node is None:
        return 0
    
    # Node is a tuple: (op, left, right) or (op, value) for CONST
    if len(node) == 1:  # nullary operation
        return 1
    
    if len(node) == 2:
        # Could be unary operation or CONST/VAR
        if node[0] in [4, 2, 3]:  # CONST, CONST_ZERO, CONST_ONE (OpID values)
            return 1
        # Unary operation
        child = node[1]
        if child is not None:
            return 1 + get_depth(child, lookup_func, max_depth - 1)
        return 1
    
    if len(node) == 3:  # binary operation
        left_depth = get_depth(node[1], lookup_func, max_depth - 1) if node[1] is not None else 0
        right_depth = get_depth(node[2], lookup_func, max_depth - 1) if node[2] is not None else 0
        return 1 + max(left_depth, right_depth)
    
    return 1


def count_nodes_in_subtree(handle: int, lookup_func: Callable[[int], Optional[Tuple]], max_count: int = 100) -> int:
    """
    Count nodes in expression tree with safety limit.
    
    Args:
        handle: The handle to count nodes for.
        lookup_func: Function to lookup a handle.
        max_count: Maximum nodes to count before stopping.
    
    Returns:
        Number of nodes in the subtree (capped at max_count).
    """
    if max_count <= 0:
        return max_count
    
    node = lookup_func(handle)
    if node is None:
        return 0
    
    total = 1
    if len(node) >= 2 and node[1] is not None:
        total += count_nodes_in_subtree(node[1], lookup_func, max_count - total)
    if len(node) >= 3 and node[2] is not None and total < max_count:
        total += count_nodes_in_subtree(node[2], lookup_func, max_count - total)
    
    return min(total, max_count)


def simplify_handle(handle: int, lookup_func: Callable, create_func: Callable) -> int:
    """Simplify an expression handle."""
    node = lookup_func(handle)
    if node is None:
        return handle
    
    left_simplified = None
    right_simplified = None
    
    if len(node) >= 2 and node[1] is not None:
        left_simplified = simplify_handle(node[1], lookup_func, create_func)
    
    if len(node) >= 3 and node[2] is not None:
        right_simplified = simplify_handle(node[2], lookup_func, create_func)
    
    if left_simplified == node[1] and right_simplified == node[2]:
        return handle
    
    op = node[0]
    if len(node) == 1:
        return create_func(op, None, None)
    elif len(node) == 2:
        return create_func(op, left_simplified, None)
    else:
        return create_func(op, left_simplified, right_simplified)
