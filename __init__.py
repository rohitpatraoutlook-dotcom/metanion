"""
Metanion - A Zero-Weight Symbolic Tensor Engine
Built entirely from scratch using operation sequences instead of numerical weights.
"""

__version__ = "0.1.0"
__author__ = "Rohit Patra"

# Core
from .core import Tensor, DType, Shape, get_arena, reset_arena

# Symbolic
from .symbolic import OpID, intern, lookup, simplify, get_pool, reset_pool

# Compile
from .compile import compile_handle, get_compiler

# Algebra
from .algebra import simplify as algebra_simplify, get_rewrite_system

# Calculus
from .calculus import differentiate, get_differentiator

# Model
from .model import MetanionModel, MetanionLayer, MetanionStack, ModelConfig

# Data
from .data import Dataset, DataLoader, StatisticsInjector

# Utils
from .utils import TreePrinter, get_cost_model, get_time_profiler

# IO
from .io import BinaryEncoder, BinaryDecoder, CheckpointManager

# Runtime
from .runtime import get_jit_cache, get_gc_controller, collect_garbage

# Engine
from .metanion_engine import (
    MetanionEngine,
    get_engine,
    create_model,
    train,
    predict,
)

# Config
from .config import MetanionConfig, ACTIVE_CONFIG, DEFAULT_CONFIG

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Core
    'Tensor',
    'DType',
    'Shape',
    'get_arena',
    'reset_arena',
    
    # Symbolic
    'OpID',
    'intern',
    'lookup',
    'simplify',
    'get_pool',
    'reset_pool',
    
    # Compile
    'compile_handle',
    'get_compiler',
    
    # Algebra
    'algebra_simplify',
    'get_rewrite_system',
    
    # Calculus
    'differentiate',
    'get_differentiator',
    
    # Model
    'MetanionModel',
    'MetanionLayer',
    'MetanionStack',
    'ModelConfig',
    
    # Data
    'Dataset',
    'DataLoader',
    'StatisticsInjector',
    
    # Utils
    'TreePrinter',
    'get_cost_model',
    'get_time_profiler',
    
    # IO
    'BinaryEncoder',
    'BinaryDecoder',
    'CheckpointManager',
    
    # Runtime
    'get_jit_cache',
    'get_gc_controller',
    'collect_garbage',
    
    # Engine
    'MetanionEngine',
    'get_engine',
    'create_model',
    'train',
    'predict',
    
    # Config
    'MetanionConfig',
    'ACTIVE_CONFIG',
    'DEFAULT_CONFIG',
]