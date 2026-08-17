"""
Public Tensor class for the Metanion engine.
Provides high-level tensor operations with lazy evaluation.
"""

from typing import Optional, Union, List, Tuple, Any, Dict
import math
import copy

from ..exceptions import TensorError, ShapeMismatchError
from .dtype_system import DType, promote_dtypes, is_numeric_dtype
from .memory_arena import get_arena
from .tensor_shape import Shape, ShapeTracker, compute_size
from .tensor_buffer import TensorBuffer


class Tensor:
    """
    Main tensor class for the Metanion engine.
    Supports lazy evaluation and automatic differentiation.
    """
    
    __slots__ = (
        '_buffer', '_shape', '_dtype', '_lazy_op',
        '_requires_grad', '_grad'
    )
    
    def __init__(
        self,
        data: Optional[Union[int, float, List, Tuple, TensorBuffer]] = None,
        shape: Optional[Shape] = None,
        dtype: DType = DType.FLOAT64,
        requires_grad: bool = False,
        _lazy_op: Optional['LazyOp'] = None
    ):
        """
        Initialize a tensor.
        
        Args:
            data: Initial data (scalar, list, or TensorBuffer).
            shape: Shape of the tensor.
            dtype: Data type.
            requires_grad: Whether to track gradients.
            _lazy_op: Internal lazy operation (if any).
        """
        self._requires_grad = requires_grad
        self._grad: Optional[Tensor] = None
        self._lazy_op = _lazy_op
        
        if _lazy_op is not None:
            # Lazy tensor: use existing buffer
            self._buffer = _lazy_op.buffer
            self._shape = _lazy_op.shape
            self._dtype = _lazy_op.dtype
            return
        
        if data is None:
            raise ValueError("data is required")
        
        if isinstance(data, TensorBuffer):
            # Wrap existing buffer
            self._buffer = data
            self._shape = data.shape
            self._dtype = data.dtype
            return
        
        if isinstance(data, (int, float)):
            # Scalar
            if shape is not None and shape != ():
                raise ValueError(f"Scalar data cannot have shape {shape}")
            self._shape = ()
            self._dtype = dtype
            self._buffer = self._create_buffer_from_scalar(data)
            return
        
        if isinstance(data, (list, tuple)):
            # List data: infer shape
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
        
        # Check if first element is a list
        if isinstance(data[0], (list, tuple)):
            sub_shape = Tensor._infer_shape(data[0])
            return (len(data),) + sub_shape
        
        return (len(data),)
    
    def _create_buffer_from_scalar(self, value: Union[int, float]) -> TensorBuffer:
        """Create a buffer from a scalar value."""
        arena = get_arena()
        size = self._dtype.itemsize()
        offset = arena.allocate(size)
        buffer = TensorBuffer(offset, (), dtype=self._dtype)
        
        if isinstance(value, float):
            if self._dtype == DType.FLOAT64:
                data = value.to_bytes(8, 'little')
            elif self._dtype == DType.FLOAT32:
                data = value.to_bytes(4, 'little')
            else:
                raise TypeError(f"Cannot store float in {self._dtype}")
        elif isinstance(value, int):
            if self._dtype in (DType.FLOAT64, DType.FLOAT32):
                value = float(value)
                if self._dtype == DType.FLOAT64:
                    data = value.to_bytes(8, 'little')
                else:
                    data = value.to_bytes(4, 'little')
            elif self._dtype == DType.INT64:
                data = value.to_bytes(8, 'little', signed=True)
            elif self._dtype == DType.INT32:
                data = value.to_bytes(4, 'little', signed=True)
            else:
                raise TypeError(f"Unsupported dtype: {self._dtype}")
        else:
            raise TypeError(f"Cannot convert {type(value)} to buffer")
        
        arena.write(offset, data)
        return buffer
    
    def _create_buffer_from_list(self, data: Union[List, Tuple], shape: Shape) -> TensorBuffer:
        """Create a buffer from nested list data."""
        from .tensor_shape import compute_size
        
        total_size = compute_size(shape)
        arena = get_arena()
        itemsize = self._dtype.itemsize()
        offset = arena.allocate(total_size * itemsize)
        buffer = TensorBuffer(offset, shape, dtype=self._dtype)
        
        # Flatten and assign values
        self._flatten_fill(data, buffer, 0)
        return buffer
    
    def _flatten_fill(self, data: Union[List, Tuple], buffer: TensorBuffer, idx: int) -> int:
        """Recursively fill buffer from nested list."""
        if not isinstance(data, (list, tuple)):
            # Leaf value
            buffer.setitem(idx, data)
            return idx + 1
        
        for item in data:
            idx = self._flatten_fill(item, buffer, idx)
        
        return idx
    
    @property
    def shape(self) -> Shape:
        """Get the tensor shape."""
        return self._shape
    
    @property
    def dtype(self) -> DType:
        """Get the tensor dtype."""
        return self._dtype
    
    @property
    def size(self) -> int:
        """Get the number of elements."""
        return compute_size(self._shape)
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self._shape)
    
    @property
    def requires_grad(self) -> bool:
        """Check if tensor requires gradients."""
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Set whether tensor requires gradients."""
        self._requires_grad = value
        if not value:
            self._grad = None
    
    @property
    def grad(self) -> Optional['Tensor']:
        """Get the gradient tensor."""
        return self._grad
    
    def _ensure_buffer(self) -> TensorBuffer:
        """Ensure we have a valid buffer."""
        if self._lazy_op is not None:
            # Evaluate lazy operation
            result = self._lazy_op.evaluate()
            self._buffer = result
            self._lazy_op = None
        return self._buffer
    
    def _get_item(self, idx: Union[int, Tuple[Union[int, slice, None], ...]]) -> Union[int, float, 'Tensor']:
        """Get item(s) from tensor."""
        buffer = self._ensure_buffer()
        result = buffer.getitem(idx)
        
        if isinstance(result, (int, float)):
            # Scalar value
            return result
        
        if isinstance(result, TensorBuffer):
            # Tensor slice
            return Tensor(result)
        
        return result
    
    def __getitem__(self, idx: Union[int, Tuple[Union[int, slice, None], ...]]) -> Union[int, float, 'Tensor']:
        """Get item(s) using index or slice."""
        return self._get_item(idx)
    
    def _set_item(self, idx: Union[int, Tuple[Union[int, slice, None], ...]], value: Union[int, float, 'Tensor']):
        """Set item(s) in tensor."""
        buffer = self._ensure_buffer()
        
        if isinstance(value, Tensor):
            # TODO: Implement tensor assignment
            raise NotImplementedError("Tensor assignment not yet implemented")
        else:
            # Scalar assignment
            if isinstance(idx, int):
                idx = (idx,)
            buffer.setitem(idx, value)
    
    def __setitem__(self, idx: Union[int, Tuple[Union[int, slice, None], ...]], value: Union[int, float, 'Tensor']):
        """Set item(s) using index or slice."""
        self._set_item(idx, value)
    
    def _binary_op(self, other: Union[int, float, 'Tensor'], op_name: str) -> 'Tensor':
        """Binary operation with lazy evaluation."""
        from ..compile.lazy_graph import LazyOp, LazyOpType
        
        if isinstance(other, (int, float)):
            other = Tensor(other)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type: {type(other)}")
        
        # Broadcast shapes
        new_shape = ShapeTracker.broadcast_shape(self._shape, other._shape)
        
        # Create lazy operation
        lazy_op = LazyOp(
            op_type=LazyOpType.from_string(op_name),
            left=self,
            right=other,
            shape=new_shape,
            dtype=promote_dtypes(self._dtype, other._dtype)
        )
        
        return Tensor(_lazy_op=lazy_op, requires_grad=self._requires_grad or other._requires_grad)
    
    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Addition operator."""
        return self._binary_op(other, 'add')
    
    def __radd__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse addition."""
        return Tensor(other) + self
    
    def __sub__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Subtraction operator."""
        return self._binary_op(other, 'sub')
    
    def __rsub__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse subtraction."""
        return Tensor(other) - self
    
    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Multiplication operator."""
        return self._binary_op(other, 'mul')
    
    def __rmul__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse multiplication."""
        return Tensor(other) * self
    
    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Division operator."""
        return self._binary_op(other, 'div')
    
    def __rtruediv__(self, other: Union[int, float]) -> 'Tensor':
        """Reverse division."""
        return Tensor(other) / self
    
    def __pow__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        """Power operator."""
        return self._binary_op(other, 'pow')
    
    def __neg__(self) -> 'Tensor':
        """Negation operator."""
        return self * Tensor(-1.0)
    
    def __pos__(self) -> 'Tensor':
        """Positive operator."""
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        if self._lazy_op is not None:
            return f"Tensor(lazy: {self._lazy_op}, shape={self._shape}, dtype={self._dtype})"
        
        buffer = self._ensure_buffer()
        
        if self.ndim == 0:
            # Scalar
            return f"Tensor({self._get_item(0)}, dtype={self._dtype.name})"
        
        # Show first few elements
        sample = []
        for i in range(min(self.size, 10)):
            idx = tuple(0 for _ in range(self.ndim))
            sample.append(self._get_item(idx))
        
        return (f"Tensor(shape={self._shape}, dtype={self._dtype.name}, "
                f"sample={sample}...)")
    
    def numpy(self):
        """Convert to NumPy array (if available)."""
        import numpy as np
        buffer = self._ensure_buffer()
        data = buffer.to_bytes()
        
        if self._dtype == DType.FLOAT64:
            return np.frombuffer(data, dtype=np.float64).reshape(self._shape)
        elif self._dtype == DType.FLOAT32:
            return np.frombuffer(data, dtype=np.float32).reshape(self._shape)
        elif self._dtype == DType.INT64:
            return np.frombuffer(data, dtype=np.int64).reshape(self._shape)
        elif self._dtype == DType.INT32:
            return np.frombuffer(data, dtype=np.int32).reshape(self._shape)
        else:
            raise ValueError(f"Unsupported dtype: {self._dtype}")
    
    @staticmethod
    def zeros(shape: Shape, dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create a tensor filled with zeros."""
        data = [0] * compute_size(shape)
        # Reshape into nested structure
        return Tensor(data, shape=shape, dtype=dtype)
    
    @staticmethod
    def ones(shape: Shape, dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create a tensor filled with ones."""
        data = [1] * compute_size(shape)
        return Tensor(data, shape=shape, dtype=dtype)
    
    @staticmethod
    def full(shape: Shape, fill_value: Union[int, float], dtype: DType = DType.FLOAT64) -> 'Tensor':
        """Create a tensor filled with a constant value."""
        data = [fill_value] * compute_size(shape)
        return Tensor(data, shape=shape, dtype=dtype)
    
    @staticmethod
    def arange(start: int, stop: int, step: int = 1, dtype: DType = DType.INT64) -> 'Tensor':
        """Create a tensor with a range of values."""
        data = list(range(start, stop, step))
        return Tensor(data, dtype=dtype)
    
    def copy(self) -> 'Tensor':
        """Create a deep copy of the tensor."""
        buffer = self._ensure_buffer()
        new_buffer = buffer.copy()
        return Tensor(new_buffer, requires_grad=self._requires_grad)
    
    def reshape(self, shape: Shape) -> 'Tensor':
        """Reshape the tensor."""
        buffer = self._ensure_buffer()
        new_buffer = buffer.view(shape)
        return Tensor(new_buffer, requires_grad=self._requires_grad)
    
    def squeeze(self, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        new_shape = ShapeTracker.squeeze(self._shape, dim)
        return self.reshape(new_shape)
    
    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add a dimension of size 1."""
        new_shape = ShapeTracker.unsqueeze(self._shape, dim)
        return self.reshape(new_shape)