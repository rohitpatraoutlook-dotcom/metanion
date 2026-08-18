"""Core tensor operations for Metanion."""

from .dtype_system import DType
from .memory_arena import MemoryArena, get_arena, reset_arena
from .tensor import Tensor
from .tensor_buffer import TensorBuffer
from .tensor_shape import Shape, ShapeTracker

__all__ = [
    'DType',
    'MemoryArena',
    'get_arena',
    'reset_arena',
    'Tensor',
    'TensorBuffer',
    'Shape',
    'ShapeTracker',
]

# Add promote_dtypes function
def promote_dtypes(dtype1, dtype2):
    """Promote two dtypes to the higher precision type."""
    priority = {
        DType.FLOAT64: 4,
        DType.FLOAT32: 3,
        DType.INT64: 2,
        DType.INT32: 1,
    }
    if priority[dtype1] >= priority[dtype2]:
        return dtype1
    return dtype2
