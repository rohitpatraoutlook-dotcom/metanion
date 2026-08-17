"""
Symbolic expression management for the Metanion engine.
Handles operation definitions, expression pools, and handle management.
"""

from .op_enum import (
    OpID, OpCategory, OP_METADATA, OP_NAMES,
    get_op_arity, get_op_category, get_op_name,
    is_binary_op, is_unary_op, is_ternary_op,
    is_reduction_op, is_constant_op, is_arithmetic_op,
    is_trigonometric_op, is_logical_op, is_calculus_op,
    is_differentiable_op
)

from .op_metadata import (
    OpMetadata, OP_METADATA_DETAILED,
    get_op_metadata, get_op_cost_ns,
    is_op_differentiable, is_op_safe,
    get_operations_by_category, get_all_operation_ids,
    get_operation_ids_for_type_signature
)

from .expression_node import (
    ExpressionNode, ExpressionNodeFactory,
    is_constant_expression, is_identity_expression,
    get_node_depth, count_nodes, get_variable_handles,
    node_to_string
)

from .handle_utils import (
    HandleTraversal,
    get_subtree_handles,
    get_depth,
    count_nodes_in_subtree,
    find_node,
    find_all_nodes,
    replace_subtree,
    collect_variable_handles,
    get_variable_count,
    is_constant_expression_tree,
    simplify_handle
)

from .hash_consing_pool import (
    HashConsingPool,
    get_pool,
    reset_pool,
    intern,
    lookup
)

__all__ = [
    # Op Enum
    'OpID',
    'OpCategory',
    'OP_METADATA',
    'OP_NAMES',
    'get_op_arity',
    'get_op_category',
    'get_op_name',
    'is_binary_op',
    'is_unary_op',
    'is_ternary_op',
    'is_reduction_op',
    'is_constant_op',
    'is_arithmetic_op',
    'is_trigonometric_op',
    'is_logical_op',
    'is_calculus_op',
    'is_differentiable_op',
    
    # Op Metadata
    'OpMetadata',
    'OP_METADATA_DETAILED',
    'get_op_metadata',
    'get_op_cost_ns',
    'is_op_differentiable',
    'is_op_safe',
    'get_operations_by_category',
    'get_all_operation_ids',
    'get_operation_ids_for_type_signature',
    
    # Expression Node
    'ExpressionNode',
    'ExpressionNodeFactory',
    'is_constant_expression',
    'is_identity_expression',
    'get_node_depth',
    'count_nodes',
    'get_variable_handles',
    'node_to_string',
    
    # Handle Utils
    'HandleTraversal',
    'get_subtree_handles',
    'get_depth',
    'count_nodes_in_subtree',
    'find_node',
    'find_all_nodes',
    'replace_subtree',
    'collect_variable_handles',
    'get_variable_count',
    'is_constant_expression_tree',
    'simplify_handle',
    
    # Hash-Consing Pool
    'HashConsingPool',
    'get_pool',
    'reset_pool',
    'intern',
    'lookup',
]