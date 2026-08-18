"""
Tensor buffer for Metanion - handles raw memory storage with struct packing.
"""

import struct
from typing import Optional, Tuple, Any, Union, List
import math

from .dtype_system import DType
from .memory_arena import get_arena
from .tensor_shape import Shape, ShapeTracker
from ..exceptions import TensorError


class TensorBuffer:
    """Raw buffer for tensor data using memory arena."""
    
    def __init__(
        self,
        offset: int,
        shape: Shape,
        strides: Optional[Tuple[int, ...]] = None,
        dtype: DType = DType.FLOAT64,
        readonly: bool = False,
        owns_memory: bool = True
    ):
        self._offset = offset
        self._shape = shape
        self._dtype = dtype
        self._readonly = readonly
        self._owns_memory = owns_memory
        self._size = ShapeTracker.compute_size(shape)
        
        if strides is None:
            self._strides = self._compute_strides()
        else:
            self._strides = strides
        
        # For 1D tensors, store a simple flat index mapping
        self._is_flat = len(shape) == 1 or shape == ()
    
    def _compute_strides(self) -> Tuple[int, ...]:
        strides = []
        stride = self._dtype.itemsize()
        for dim in reversed(self._shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))
    
    @property
    def shape(self) -> Shape:
        return self._shape
    
    @property
    def strides(self) -> Tuple[int, ...]:
        return self._strides
    
    @property
    def dtype(self) -> DType:
        return self._dtype
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def nbytes(self) -> int:
        return self._size * self._dtype.itemsize()
    
    @property
    def offset(self) -> int:
        return self._offset
    
    def _get_index(self, idx: Tuple[int, ...]) -> int:
        """Convert multi-dimensional index to flat offset."""
        # Handle scalar case
        if self._shape == ():
            return self._offset
        
        # Handle 1D case with simple index
        if len(self._shape) == 1 and len(idx) == 1:
            flat_idx = idx[0]
            if flat_idx < 0 or flat_idx >= self._size:
                raise IndexError(f"Index {flat_idx} out of bounds for size {self._size}")
            return self._offset + flat_idx * self._strides[0]
        
        # Handle 2D case with tuple index
        if len(self._shape) == 2 and len(idx) == 2:
            flat_idx = idx[0] * self._shape[1] + idx[1]
            if flat_idx < 0 or flat_idx >= self._size:
                raise IndexError(f"Index {idx} out of bounds for shape {self._shape}")
            return self._offset + flat_idx * self._dtype.itemsize()
        
        # General case
        if len(idx) != len(self._shape):
            raise IndexError(
                f"Index {idx} has {len(idx)} dimensions, "
                f"buffer has {len(self._shape)} dimensions"
            )
        
        byte_offset = 0
        for i, (dim, stride, index) in enumerate(zip(self._shape, self._strides, idx)):
            if index < 0 or index >= dim:
                raise IndexError(
                    f"Index {index} out of bounds for dimension {i} (size {dim})"
                )
            byte_offset += index * stride
        
        return self._offset + byte_offset
    
    def _pack_value(self, value: Union[int, float]) -> bytes:
        if self._dtype == DType.FLOAT64:
            return struct.pack('d', float(value))
        elif self._dtype == DType.FLOAT32:
            return struct.pack('f', float(value))
        elif self._dtype == DType.INT64:
            return struct.pack('q', int(value))
        elif self._dtype == DType.INT32:
            return struct.pack('i', int(value))
        else:
            raise TypeError(f"Unsupported dtype for packing: {self._dtype}")
    
    def _unpack_value(self, data: bytes) -> Union[int, float]:
        if self._dtype == DType.FLOAT64:
            return struct.unpack('d', data)[0]
        elif self._dtype == DType.FLOAT32:
            return struct.unpack('f', data)[0]
        elif self._dtype == DType.INT64:
            return struct.unpack('q', data)[0]
        elif self._dtype == DType.INT32:
            return struct.unpack('i', data)[0]
        else:
            raise TypeError(f"Unsupported dtype for unpacking: {self._dtype}")
    
    def getitem(self, idx: Union[int, Tuple[int, ...]]) -> Union[int, float]:
        """Get a single element from the buffer."""
        if isinstance(idx, int):
            idx = (idx,)
        
        flat_idx = self._get_index(idx)
        arena = get_arena()
        data = arena.read(flat_idx, self._dtype.itemsize())
        
        return self._unpack_value(data)
    
    def setitem(self, idx: Union[int, Tuple[int, ...]], value: Union[int, float]) -> None:
        """Set a single element in the buffer."""
        if self._readonly:
            raise ValueError("Buffer is read-only")
        
        if isinstance(idx, int):
            idx = (idx,)
        
        flat_idx = self._get_index(idx)
        arena = get_arena()
        data = self._pack_value(value)
        arena.write(flat_idx, data)
    
    def fill(self, value: Union[int, float]) -> None:
        if self._readonly:
            raise ValueError("Buffer is read-only")
        
        arena = get_arena()
        data = self._pack_value(value) * self._size
        arena.write(self._offset, data)
    
    def copy(self) -> 'TensorBuffer':
        arena = get_arena()
        new_offset = arena.allocate(self.nbytes)
        
        data = arena.read(self._offset, self.nbytes)
        arena.write(new_offset, data)
        
        return TensorBuffer(
            offset=new_offset,
            shape=self._shape,
            strides=self._strides,
            dtype=self._dtype,
            readonly=False,
            owns_memory=True
        )
    
    def view(self, shape: Shape) -> 'TensorBuffer':
        new_shape = ShapeTracker.reshape(self._shape, shape)
        new_strides = self._compute_strides()
        
        return TensorBuffer(
            offset=self._offset,
            shape=new_shape,
            strides=new_strides,
            dtype=self._dtype,
            readonly=self._readonly,
            owns_memory=False
        )
    
    def to_bytes(self) -> bytes:
        arena = get_arena()
        return arena.read(self._offset, self.nbytes)
    
    def to_list(self) -> List[Union[int, float]]:
        """Convert buffer to a flat list of values."""
        result = []
        for i in range(self._size):
            if len(self._shape) == 1:
                result.append(self.getitem(i))
            elif self._shape == ():
                if i == 0:
                    result.append(self.getitem(()))
            elif len(self._shape) == 2:
                row = i // self._shape[1]
                col = i % self._shape[1]
                result.append(self.getitem((row, col)))
            else:
                # For higher dimensions, compute index tuple
                idx = []
                remaining = i
                for dim in reversed(self._shape):
                    if dim > 0:
                        idx.append(remaining % dim)
                        remaining //= dim
                    else:
                        idx.append(0)
                idx.reverse()
                try:
                    result.append(self.getitem(tuple(idx)))
                except:
                    result.append(0.0)
        return result
    
    def __len__(self) -> int:
        return self._size
    
    def __repr__(self) -> str:
        return (
            f"TensorBuffer(shape={self._shape}, "
            f"dtype={self._dtype.name}, "
            f"size={self._size}, "
            f"nbytes={self.nbytes})"
        )
    
    def __del__(self):
        if self._owns_memory and self._offset is not None:
            try:
                arena = get_arena()
                arena.free(self._offset)
            except:
                pass
