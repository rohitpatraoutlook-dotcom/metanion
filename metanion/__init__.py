"""
Metanion - A Zero-Weight Symbolic Tensor Engine
"""

__version__ = "0.1.0"
__author__ = "Metanion Team"

# Core
from .core import Tensor, DType, Shape, get_arena, reset_arena

# Symbolic
from .symbolic import (
    OpID, OpCategory, intern, lookup, simplify,
    get_pool, reset_pool, get_depth, count_nodes_in_subtree,
    get_op_name, get_op_arity, is_binary_op, is_unary_op,
    ExpressionNode, ExpressionNodeFactory, get_all_operation_ids
)

# Compile
from .compile import compile_handle, StraightLineProgram

# API - Simple interface
from .api import Metanion

__all__ = [
    '__version__', '__author__',
    'Tensor', 'DType', 'Shape',
    'get_arena', 'reset_arena',
    'OpID', 'OpCategory',
    'intern', 'lookup', 'simplify',
    'get_pool', 'reset_pool',
    'get_depth', 'count_nodes_in_subtree',
    'get_op_name', 'get_op_arity',
    'is_binary_op', 'is_unary_op',
    'ExpressionNode', 'ExpressionNodeFactory',
    'get_all_operation_ids',
    'compile_handle', 'StraightLineProgram',
    'Metanion',  # Main API
]
