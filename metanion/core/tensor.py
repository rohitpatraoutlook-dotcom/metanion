"""
Tensor class for Metanion - main user-facing tensor operations.
"""

from typing import Optional, Union, List, Tuple, Any
import math

from .dtype_system import DType
from .memory_arena import get_arena
from .tensor_shape import Shape, ShapeTracker
from .tensor_buffer import TensorBuffer
from ..exceptions import TensorError


class Tensor:
    """Main tensor class with lazy evaluation."""
    
    def __init__(
        self,
        data: Optional[Union[int, float, List, Tuple, TensorBuffer]] = None,
        shape: Optional[Shape] = None,
        dtype: DType = DType.FLOAT64,
        _lazy_op: Optional[Any] = None
    ):
        self._dtype = dtype
        self._lazy_op = _lazy_op
        self._grad = None
        self._requires_grad = False
        
        if _lazy_op is not None:
            self._buffer = _lazy_op.buffer
            self._shape = _lazy_op.shape
            self._dtype = _lazy_op.dtype
            return
        
        if data is None:
            raise ValueError("data is required")
        
        if isinstance(data, TensorBuffer):
            self._buffer = data
            self._shape = data.shape
            self._dtype = data.dtype
            return
        
        if isinstance(data, (int, float)):
            if shape is not None and shape != ():
                raise ValueError(f"Scalar data cannot have shape {shape}")
            self._shape = ()
            self._dtype = dtype
            self._buffer = self._create_scalar_buffer(float(data))
            return
        
        if isinstance(data, (list, tuple)):
            if shape is None:
                shape = self._infer_shape(data)
            self._shape = shape
            self._dtype = dtype
            self._buffer = self._create_buffer_from_list(data, shape)
            return
        
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def _infer_shape(data: Union[List, Tuple]) -> Shape:
        """Infer shape from nested list/tuple."""
        if not isinstance(data, (list, tuple)):
            return ()
        
        if not data:
            return (0,)
        
        if isinstance(data[0], (list, tuple)):
            sub_shape = Tensor._infer_shape(data[0])
            return (len(data),) + sub_shape
        
        return (len(data),)
    
    def _create_scalar_buffer(self, value: float) -> TensorBuffer:
        """Create buffer for scalar value."""
        arena = get_arena()
        offset = arena.allocate(self._dtype.itemsize())
        buffer = TensorBuffer(offset, (), dtype=self._dtype)
        buffer.setitem((), value)
        return buffer
    
    def _create_buffer_from_list(self, data: Union[List, Tuple], shape: Shape) -> TensorBuffer:
        """Create buffer from nested list data."""
        total_size = ShapeTracker.compute_size(shape)
        arena = get_arena()
        itemsize = self._dtype.itemsize()
        offset = arena.allocate(total_size * itemsize)
        buffer = TensorBuffer(offset, shape, dtype=self._dtype)
        
        # Flatten and assign values
        flat_data = self._flatten_list(data)
        for i, val in enumerate(flat_data):
            if i < total_size:
                buffer.setitem(i, float(val))
        
        return buffer
    
    def _flatten_list(self, data: Union[List, Tuple]) -> List:
        """Flatten nested list to 1D list."""
        result = []
        for item in data:
            if isinstance(item, (list, tuple)):
                result.extend(self._flatten_list(item))
            else:
                result.append(float(item))
        return result
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    @property
    def dtype(self) -> DType:
        return self._dtype
    
    @property
    def size(self) -> int:
        return ShapeTracker.compute_size(self._shape)
    
    @property
    def ndim(self) -> int:
        return len(self._shape)
    
    def _ensure_buffer(self) -> TensorBuffer:
        """Ensure we have a valid buffer."""
        if self._lazy_op is not None:
            result = self._lazy_op.evaluate()
            self._buffer = result
            self._lazy_op = None
        return self._buffer
    
    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Addition operator."""
        if isinstance(other, (int, float)):
            other_data = [float(other)] * self.size
            other = Tensor(other_data, shape=self._shape)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type: {type(other)}")
        
        self_data = self._ensure_buffer().to_list()
        other_data = other._ensure_buffer().to_list()
        
        max_len = max(len(self_data), len(other_data))
        self_data.extend([0.0] * (max_len - len(self_data)))
        other_data.extend([0.0] * (max_len - len(other_data)))
        
        result_data = [self_data[i] + other_data[i] for i in range(max_len)]
        
        return Tensor(result_data, shape=self._shape)
    
    def __repr__(self) -> str:
        """String representation - safe for all tensor shapes."""
        try:
            if self._shape == ():
                val = self._ensure_buffer().to_list()[0] if self.size > 0 else 0
                return f"Tensor({val}, dtype={self._dtype.name})"
            else:
                # Just return shape info to avoid indexing issues
                return f"Tensor(shape={self._shape}, dtype={self._dtype.name})"
        except Exception:
            return f"Tensor(shape={self._shape}, dtype={self._dtype.name})"
    
    def to_list(self) -> List:
        """Convert tensor to flat list."""
        return self._ensure_buffer().to_list()
    
    @staticmethod
    def zeros(shape: Shape, dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create tensor filled with zeros."""
        size = ShapeTracker.compute_size(shape)
        data = [0.0] * size
        return Tensor(data, shape=shape, dtype=dtype)
    
    @staticmethod
    def ones(shape: Shape, dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create tensor filled with ones."""
        size = ShapeTracker.compute_size(shape)
        data = [1.0] * size
        return Tensor(data, shape=shape, dtype=dtype)
    
    @staticmethod
    def full(shape: Shape, fill_value: float, dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create tensor filled with constant."""
        size = ShapeTracker.compute_size(shape)
        data = [float(fill_value)] * size
        return Tensor(data, shape=shape, dtype=dtype)
