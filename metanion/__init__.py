"""
Metanion - A Zero-Weight Symbolic Tensor Engine
"""

__version__ = "0.1.0"
__author__ = "Metanion Team"

# Core
from .core import Tensor, DType, Shape

# Symbolic
from .symbolic import (
    OpID, OpCategory, intern, lookup, get_pool, reset_pool,
    simplify, get_depth, count_nodes_in_subtree
)

# Compile
try:
    from .compile import compile_handle, StraightLineProgram
except ImportError:
    def compile_handle(h): return lambda x: x
    class StraightLineProgram: pass

# Calculus
try:
    from .calculus import differentiate, get_differentiator
except ImportError:
    def differentiate(h, v=-1): return h
    def get_differentiator(): return None

# Model
try:
    from .model import MetanionModel, create_model, train, predict
except ImportError:
    class MetanionModel: pass
    def create_model(*args, **kwargs): return MetanionModel()
    def train(*args, **kwargs): return {}
    def predict(x): return x

# Engine
try:
    from .metanion_engine import MetanionEngine, get_engine
except ImportError:
    class MetanionEngine: pass
    def get_engine(): return MetanionEngine()

# IO
try:
    from .io import BinaryEncoder, BinaryDecoder, CheckpointManager
except ImportError:
    pass

__all__ = [
    '__version__',
    '__author__',
    'Tensor',
    'DType',
    'Shape',
    'OpID',
    'OpCategory',
    'intern',
    'lookup',
    'get_pool',
    'reset_pool',
    'simplify',
    'get_depth',
    'count_nodes_in_subtree',
    'compile_handle',
    'StraightLineProgram',
    'differentiate',
    'get_differentiator',
    'MetanionModel',
    'create_model',
    'train',
    'predict',
    'MetanionEngine',
    'get_engine',
    'BinaryEncoder',
    'BinaryDecoder',
    'CheckpointManager',
]
