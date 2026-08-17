"""
Core tensor and memory management.
"""

from .dtype_system import DType, promote_dtypes, is_numeric_dtype
from .memory_arena import MemoryArena, get_arena, reset_arena
from .tensor_shape import (
    Shape, Strides, ShapeTracker,
    broadcast_shape, compute_strides, compute_size,
    is_scalar, is_vector, is_matrix
)
from .tensor_buffer import TensorBuffer
from .tensor import Tensor

__all__ = [
    'DType',
    'promote_dtypes',
    'is_numeric_dtype',
    'MemoryArena',
    'get_arena',
    'reset_arena',
    'Shape',
    'Strides',
    'ShapeTracker',
    'broadcast_shape',
    'compute_strides',
    'compute_size',
    'is_scalar',
    'is_vector',
    'is_matrix',
    'TensorBuffer',
    'Tensor',
]