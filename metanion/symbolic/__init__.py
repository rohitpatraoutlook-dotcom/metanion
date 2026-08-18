"""Symbolic expression system for Metanion."""

from .op_enum import OpID, OpCategory, get_op_name, get_op_arity, is_binary_op, is_unary_op
from .hash_consing_pool import get_pool, reset_pool, intern, lookup
from .handle_utils import get_depth, count_nodes_in_subtree, simplify_handle

# Define simplify as alias for simplify_handle
def simplify(handle):
    from .handle_utils import simplify_handle
    return simplify_handle(handle)

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
]
