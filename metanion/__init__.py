"""
Metanion - A Zero-Weight Symbolic Tensor Engine
"""

__version__ = "3.5.0"
__author__ = "Metanion Team"

# Core
from .core import Tensor, DType, Shape, get_arena, reset_arena

# Symbolic
from .symbolic import (
    OpID, OpCategory, intern, lookup, simplify,
    get_pool, reset_pool, get_depth, count_nodes_in_subtree,
    get_op_name, get_op_arity, is_binary_op, is_unary_op, is_constant_op,
    ExpressionNode, ExpressionNodeFactory
)

# Compile
from .compile import compile_handle, StraightLineProgram

# Algebra
from .algebra import get_rewrite_system

# Calculus
from .calculus import differentiate, get_differentiator, get_derivative_rules

# GP
from .gp import (
    GPIndividual, IndividualFactory,
    PopulationInitializer, InitializationMethod,
    TournamentSelection, RouletteSelection, RankSelection, ElitismSelection,
    SubtreeCrossover, OnePointCrossover, UniformCrossover,
    PointMutation, SubtreeMutation, ShrinkMutation,
    FitnessEvaluator, PopulationManager,
    BloatControl, BloatController,
    SymbolicRegularization,
    safe_div, safe_log, safe_log10, safe_sqrt, safe_pow,
    safe_sin, safe_cos, safe_tan, safe_exp, safe_inv,
    safe_abs, safe_square, safe_cube
)

# Research - using absolute import since research is at same level
try:
    from research import run_gp, print_expr, test_expression
except ImportError:
    try:
        import sys
        sys.path.insert(0, '.')
        from research import run_gp, print_expr, test_expression
    except ImportError:
        print("Warning: research module not found. Some features may not work.")

# API
from .api import Metanion

# Utils
from .utils import TreePrinter, get_cost_model, get_time_profiler

# IO
from .io import BinaryEncoder, BinaryDecoder, CheckpointManager

# Runtime
from .runtime import get_jit_cache, get_gc_controller

__all__ = [
    '__version__', '__author__',
    'Tensor', 'DType', 'Shape',
    'get_arena', 'reset_arena',
    'OpID', 'OpCategory',
    'intern', 'lookup', 'simplify',
    'get_pool', 'reset_pool',
    'get_depth', 'count_nodes_in_subtree',
    'get_op_name', 'get_op_arity',
    'is_binary_op', 'is_unary_op', 'is_constant_op',
    'ExpressionNode', 'ExpressionNodeFactory',
    'compile_handle', 'StraightLineProgram',
    'get_rewrite_system',
    'differentiate', 'get_differentiator', 'get_derivative_rules',
    'GPIndividual', 'IndividualFactory',
    'PopulationInitializer', 'InitializationMethod',
    'TournamentSelection', 'RouletteSelection', 'RankSelection', 'ElitismSelection',
    'SubtreeCrossover', 'OnePointCrossover', 'UniformCrossover',
    'PointMutation', 'SubtreeMutation', 'ShrinkMutation',
    'FitnessEvaluator', 'PopulationManager',
    'BloatControl', 'BloatController',
    'SymbolicRegularization',
    'safe_div', 'safe_log', 'safe_log10', 'safe_sqrt', 'safe_pow',
    'safe_sin', 'safe_cos', 'safe_tan', 'safe_exp', 'safe_inv',
    'safe_abs', 'safe_square', 'safe_cube',
    'run_gp', 'print_expr', 'test_expression',
    'Metanion',
    'TreePrinter', 'get_cost_model', 'get_time_profiler',
    'BinaryEncoder', 'BinaryDecoder', 'CheckpointManager',
    'get_jit_cache', 'get_gc_controller',
]
