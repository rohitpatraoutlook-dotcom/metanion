"""
Metanion - A Zero-Weight Symbolic Tensor Engine
"""

__version__ = "0.1.0"
__author__ = "Metanion Team"

# Core
from .core import Tensor, DType, Shape, get_arena, reset_arena, ShapeTracker, TensorBuffer

# Symbolic
from .symbolic import (
    OpID, OpCategory, intern, lookup, simplify,
    get_pool, reset_pool, get_depth, count_nodes_in_subtree,
    get_op_name, get_op_arity, is_binary_op, is_unary_op,
    ExpressionNode, ExpressionNodeFactory, get_all_operation_ids
)

# Compile
from .compile import compile_handle, StraightLineProgram

# Algebra
from .algebra import get_rewrite_system

# Calculus
from .calculus import differentiate, get_differentiator, get_derivative_rules

# Model
try:
    from .model import MetanionModel
except ImportError:
    MetanionModel = None

# GP
try:
    from .gp import (
        GPIndividual, IndividualFactory,
        PopulationInitializer, InitializationMethod,
        TournamentSelection, RouletteSelection, RankSelection, ElitismSelection,
        SubtreeCrossover, OnePointCrossover, UniformCrossover,
        PointMutation, SubtreeMutation, ShrinkMutation,
        FitnessEvaluator, PopulationManager,
        BloatControl, BloatController
    )
except ImportError:
    GPIndividual = None
    IndividualFactory = None
    PopulationInitializer = None
    InitializationMethod = None
    TournamentSelection = None
    RouletteSelection = None
    RankSelection = None
    ElitismSelection = None
    SubtreeCrossover = None
    OnePointCrossover = None
    UniformCrossover = None
    PointMutation = None
    SubtreeMutation = None
    ShrinkMutation = None
    FitnessEvaluator = None
    PopulationManager = None
    BloatControl = None
    BloatController = None

# Utils
from .utils import TreePrinter, get_cost_model, get_time_profiler

# IO
from .io import BinaryEncoder, BinaryDecoder, CheckpointManager

# Runtime
from .runtime import get_jit_cache, get_gc_controller

# Engine
from .metanion_engine import MetanionEngine, get_engine

__all__ = [
    '__version__', '__author__',
    'Tensor', 'DType', 'Shape',
    'get_arena', 'reset_arena', 'ShapeTracker', 'TensorBuffer',
    'OpID', 'OpCategory', 'intern', 'lookup', 'simplify',
    'get_pool', 'reset_pool', 'get_depth', 'count_nodes_in_subtree',
    'get_op_name', 'get_op_arity', 'is_binary_op', 'is_unary_op',
    'ExpressionNode', 'ExpressionNodeFactory', 'get_all_operation_ids',
    'compile_handle', 'StraightLineProgram',
    'get_rewrite_system',
    'differentiate', 'get_differentiator', 'get_derivative_rules',
    'MetanionModel',
    'GPIndividual', 'IndividualFactory',
    'PopulationInitializer', 'InitializationMethod',
    'TournamentSelection', 'RouletteSelection', 'RankSelection', 'ElitismSelection',
    'SubtreeCrossover', 'OnePointCrossover', 'UniformCrossover',
    'PointMutation', 'SubtreeMutation', 'ShrinkMutation',
    'FitnessEvaluator', 'PopulationManager',
    'BloatControl', 'BloatController',
    'TreePrinter', 'get_cost_model', 'get_time_profiler',
    'BinaryEncoder', 'BinaryDecoder', 'CheckpointManager',
    'get_jit_cache', 'get_gc_controller',
    'MetanionEngine', 'get_engine',
]
