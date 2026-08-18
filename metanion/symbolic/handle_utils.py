"""
Utility functions for working with handles in the expression pool.
Provides traversal, manipulation, and analysis functions.
"""

from typing import Optional, List, Tuple, Set, Callable, Any, Dict
from collections import deque

from .op_enum import OpID, get_op_arity, get_op_name
from .expression_node import ExpressionNode
from ..exceptions import HandleNotFoundError, ExpressionError


class HandleTraversal:
    """Utility class for traversing expression trees."""
    
    @staticmethod
    def postorder_iter(start_handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> List[int]:
        """
        Iterate over handles in postorder (children before parent).
        
        Args:
            start_handle: Starting handle.
            lookup_func: Function to lookup a handle's node.
            
        Returns:
            List of handles in postorder.
        """
        result = []
        visited = set()
        stack = [(start_handle, False)]
        
        while stack:
            handle, processed = stack.pop()
            
            if processed:
                result.append(handle)
                continue
            
            if handle in visited:
                continue
            
            visited.add(handle)
            node = lookup_func(handle)
            
            if node is None:
                raise HandleNotFoundError(f"Handle {handle} not found")
            
            stack.append((handle, True))
            
            # Push children in reverse order (so they're processed in order)
            for child in reversed(node.get_children()):
                if child is not None:
                    stack.append((child, False))
        
        return result
    
    @staticmethod
    def preorder_iter(start_handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> List[int]:
        """
        Iterate over handles in preorder (parent before children).
        
        Args:
            start_handle: Starting handle.
            lookup_func: Function to lookup a handle's node.
            
        Returns:
            List of handles in preorder.
        """
        result = []
        stack = [start_handle]
        
        while stack:
            handle = stack.pop()
            result.append(handle)
            
            node = lookup_func(handle)
            if node is None:
                raise HandleNotFoundError(f"Handle {handle} not found")
            
            # Push children in reverse order
            for child in reversed(node.get_children()):
                if child is not None:
                    stack.append(child)
        
        return result
    
    @staticmethod
    def inorder_iter(start_handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> List[int]:
        """
        Iterate over handles in inorder (left, parent, right).
        Only works for binary operations.
        
        Args:
            start_handle: Starting handle.
            lookup_func: Function to lookup a handle's node.
            
        Returns:
            List of handles in inorder.
        """
        result = []
        stack = []
        current = start_handle
        
        while stack or current is not None:
            if current is not None:
                stack.append(current)
                node = lookup_func(current)
                if node is None:
                    raise HandleNotFoundError(f"Handle {current} not found")
                current = node.left if node.left is not None else None
            else:
                current = stack.pop()
                result.append(current)
                node = lookup_func(current)
                if node is None:
                    raise HandleNotFoundError(f"Handle {current} not found")
                current = node.right if node.right is not None else None
        
        return result


def get_subtree_handles(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> Set[int]:
    """
    Get all handles in the subtree rooted at handle.
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        Set of all handles in the subtree.
    """
    visited = set()
    stack = [handle]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        
        node = lookup_func(current)
        if node is None:
            continue
        
        for child in node.get_children():
            if child is not None:
                stack.append(child)
    
    return visited


def get_depth(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> int:
    """
    Get the depth of the expression tree rooted at handle.
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        Maximum depth (number of nodes along longest path).
    """
    node = lookup_func(handle)
    if node is None:
        return 0
    
    if node.arity == 0:
        return 1
    
    max_child_depth = 0
    for child in node.get_children():
        if child is not None:
            child_depth = get_depth(child, lookup_func)
            max_child_depth = max(max_child_depth, child_depth)
    
    return 1 + max_child_depth


def count_nodes_in_subtree(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> int:
    """
    Count the total number of nodes in the expression tree.
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        Total node count.
    """
    node = lookup_func(handle)
    if node is None:
        return 0
    
    total = 1
    for child in node.get_children():
        if child is not None:
            total += count_nodes_in_subtree(child, lookup_func)
    
    return total


def find_node(handle: int, target_op: OpID, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> Optional[int]:
    """
    Find the first node with the given operation in the tree.
    
    Args:
        handle: Root handle.
        target_op: Operation to find.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        Handle of the first matching node, or None if not found.
    """
    node = lookup_func(handle)
    if node is None:
        return None
    
    if node.op == target_op:
        return handle
    
    for child in node.get_children():
        if child is not None:
            result = find_node(child, target_op, lookup_func)
            if result is not None:
                return result
    
    return None


def find_all_nodes(handle: int, target_op: OpID, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> List[int]:
    """
    Find all nodes with the given operation in the tree.
    
    Args:
        handle: Root handle.
        target_op: Operation to find.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        List of handles of matching nodes.
    """
    result = []
    node = lookup_func(handle)
    if node is None:
        return result
    
    if node.op == target_op:
        result.append(handle)
    
    for child in node.get_children():
        if child is not None:
            result.extend(find_all_nodes(child, target_op, lookup_func))
    
    return result


def replace_subtree(
    root_handle: int,
    old_subtree_handle: int,
    new_subtree_handle: int,
    lookup_func: Callable[[int], Optional[ExpressionNode]],
    create_func: Callable[[OpID, Optional[int], Optional[int]], int]
) -> int:
    """
    Replace a subtree in the expression tree.
    
    Args:
        root_handle: Root of the tree.
        old_subtree_handle: Handle of the subtree to replace.
        new_subtree_handle: Handle of the new subtree.
        lookup_func: Function to lookup a handle's node.
        create_func: Function to create a new handle from (op, left, right).
        
    Returns:
        New root handle, or old root if no change.
    """
    if root_handle == old_subtree_handle:
        return new_subtree_handle
    
    node = lookup_func(root_handle)
    if node is None:
        return root_handle
    
    new_left = root_handle
    new_right = root_handle
    
    if node.left is not None:
        if node.left == old_subtree_handle:
            new_left = new_subtree_handle
        else:
            new_left = replace_subtree(node.left, old_subtree_handle, new_subtree_handle, lookup_func, create_func)
    
    if node.right is not None:
        if node.right == old_subtree_handle:
            new_right = new_subtree_handle
        else:
            new_right = replace_subtree(node.right, old_subtree_handle, new_subtree_handle, lookup_func, create_func)
    
    if new_left == node.left and new_right == node.right:
        return root_handle
    
    return create_func(node.op, new_left, new_right)


def collect_variable_handles(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> List[int]:
    """
    Collect all variable handles (IDENTITY nodes) in the expression.
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        List of variable handles.
    """
    result = []
    node = lookup_func(handle)
    if node is None:
        return result
    
    if node.op == OpID.IDENTITY:
        result.append(handle)
    
    for child in node.get_children():
        if child is not None:
            result.extend(collect_variable_handles(child, lookup_func))
    
    return result


def get_variable_count(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> int:
    """
    Count the number of variable occurrences in the expression.
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        Number of variable occurrences.
    """
    return len(collect_variable_handles(handle, lookup_func))


def is_constant_expression_tree(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]]) -> bool:
    """
    Check if the expression tree is constant (no variables).
    
    Args:
        handle: Root handle.
        lookup_func: Function to lookup a handle's node.
        
    Returns:
        True if the expression contains no variables.
    """
    return len(collect_variable_handles(handle, lookup_func)) == 0


def simplify_handle(handle: int, lookup_func: Callable[[int], Optional[ExpressionNode]], create_func: Callable) -> int:
    """
    Simplified version of handle simplification.
    Implements basic algebraic simplifications.
    
    Args:
        handle: Handle to simplify.
        lookup_func: Function to lookup a handle's node.
        create_func: Function to create a new handle from (op, left, right).
        
    Returns:
        Simplified handle.
    """
    node = lookup_func(handle)
    if node is None:
        return handle
    
    # Recursively simplify children
    left_simplified = handle
    right_simplified = handle
    
    if node.left is not None:
        left_simplified = simplify_handle(node.left, lookup_func, create_func)
    
    if node.right is not None:
        right_simplified = simplify_handle(node.right, lookup_func, create_func)
    
    # Identity: x + 0 = x, x * 1 = x
    if node.op == OpID.ADD:
        if right_simplified is not None:
            right_node = lookup_func(right_simplified)
            if right_node and right_node.op == OpID.CONST_ZERO:
                return left_simplified
        if left_simplified is not None:
            left_node = lookup_func(left_simplified)
            if left_node and left_node.op == OpID.CONST_ZERO:
                return right_simplified
    
    if node.op == OpID.MUL:
        if right_simplified is not None:
            right_node = lookup_func(right_simplified)
            if right_node and right_node.op == OpID.CONST_ONE:
                return left_simplified
        if left_simplified is not None:
            left_node = lookup_func(left_simplified)
            if left_node and left_node.op == OpID.CONST_ONE:
                return right_simplified
    
    # x - x = 0
    if node.op == OpID.SUB:
        if left_simplified == right_simplified:
            # Need to create constant zero handle
            # This is simplified - in practice, you'd get the CONST_ZERO from the pool
            return create_func(OpID.CONST_ZERO, None, None)
    
    # x / x = 1 (if x != 0, but we assume symbolic)
    if node.op == OpID.DIV:
        if left_simplified == right_simplified:
            return create_func(OpID.CONST_ONE, None, None)
    
    # exp(log(x)) = x
    if node.op == OpID.EXP:
        left_node = lookup_func(left_simplified) if left_simplified is not None else None
        if left_node and left_node.op == OpID.LOG:
            log_child = left_node.left
            if log_child is not None:
                return log_child
    
    # log(exp(x)) = x
    if node.op == OpID.LOG:
        left_node = lookup_func(left_simplified) if left_simplified is not None else None
        if left_node and left_node.op == OpID.EXP:
            exp_child = left_node.left
            if exp_child is not None:
                return exp_child
    
    # If nothing changed, return original
    if left_simplified == node.left and right_simplified == node.right:
        return handle
    
    return create_func(node.op, left_simplified, right_simplified)