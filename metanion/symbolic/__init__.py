"""Symbolic expression system for Metanion."""

from .op_enum import OpID, OpCategory, get_op_name, get_op_arity, is_binary_op, is_unary_op
from .hash_consing_pool import get_pool, reset_pool, intern, lookup
from .handle_utils import get_depth, count_nodes_in_subtree, simplify_handle
from .expression_node import ExpressionNode, ExpressionNodeFactory


def simplify(handle):
    """Simplify an expression."""
    from .hash_consing_pool import get_pool, intern as intern_fn
    
    def lookup_func(h):
        return get_pool().get_node(h)
    
    def create_func(op, left, right):
        return intern_fn(op, left, right)
    
    return simplify_handle(handle, lookup_func, create_func)


def get_all_operation_ids():
    """Get all operation IDs."""
    from .op_enum import OpID
    return list(OpID)


__all__ = [
    'OpID',
    'OpCategory',
    'get_op_name',
    'get_op_arity',
    'is_binary_op',
    'is_unary_op',
    'get_pool',
    'reset_pool',
    'intern',
    'lookup',
    'get_depth',
    'count_nodes_in_subtree',
    'simplify_handle',
    'simplify',
    'ExpressionNode',
    'ExpressionNodeFactory',
    'get_all_operation_ids',
]
