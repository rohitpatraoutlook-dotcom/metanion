"""
Tensor buffer abstraction over the memory arena.
Provides raw data storage with shape and stride information.
"""

from typing import Optional, Union, List, Tuple, Any
import math

from ..exceptions import TensorError, MemoryAllocationError
from .dtype_system import DType, is_numeric_dtype
from .memory_arena import get_arena
from .tensor_shape import Shape, Strides, ShapeTracker, compute_strides, compute_size


class TensorBuffer:
    """
    Raw buffer storage for tensor data.
    Manages memory in the arena and provides access methods.
    """
    
    __slots__ = (
        '_offset', '_shape', '_strides', '_dtype', '_size',
        '_readonly', '_owns_memory'
    )
    
    def __init__(
        self,
        offset: int,
        shape: Shape,
        strides: Optional[Strides] = None,
        dtype: DType = DType.FLOAT64,
        readonly: bool = False,
        owns_memory: bool = True
    ):
        """
        Initialize a tensor buffer.
        
        Args:
            offset: Byte offset in the memory arena.
            shape: Shape tuple.
            strides: Strides tuple. If None, computed from shape.
            dtype: Data type.
            readonly: Whether the buffer is read-only.
            owns_memory: Whether to free memory on deletion.
        """
        ShapeTracker.validate_shape(shape)
        
        self._offset = offset
        self._shape = shape
        self._dtype = dtype
        self._readonly = readonly
        self._owns_memory = owns_memory
        
        # Compute strides if not provided
        if strides is None:
            self._strides = compute_strides(shape, dtype.itemsize())
        else:
            self._strides = strides
        
        # Compute total size
        self._size = compute_size(shape)
    
    @property
    def offset(self) -> int:
        """Get the buffer offset."""
        return self._offset
    
    @property
    def shape(self) -> Shape:
        """Get the buffer shape."""
        return self._shape
    
    @property
    def strides(self) -> Strides:
        """Get the buffer strides."""
        return self._strides
    
    @property
    def dtype(self) -> DType:
        """Get the buffer dtype."""
        return self._dtype
    
    @property
    def size(self) -> int:
        """Get the number of elements."""
        return self._size
    
    @property
    def nbytes(self) -> int:
        """Get the total bytes used by the buffer."""
        return self._size * self._dtype.itemsize()
    
    @property
    def readonly(self) -> bool:
        """Check if buffer is read-only."""
        return self._readonly
    
    def _get_memoryview(self) -> memoryview:
        """Get a memoryview of the buffer."""
        arena = get_arena()
        return arena.get_buffer(self._offset, self.nbytes)
    
    def _get_index(self, idx: Union[int, Tuple[int, ...]]) -> int:
        """
        Convert multi-dimensional index to flat offset.
        
        Args:
            idx: Index tuple.
            
        Returns:
            Flat byte offset.
            
        Raises:
            IndexError: If index is out of bounds.
        """
        if isinstance(idx, int):
            idx = (idx,)
        
        if len(idx) != len(self._shape):
            raise IndexError(
                f"Index {idx} has {len(idx)} dimensions, "
                f"but buffer has {len(self._shape)}"
            )
        
        byte_offset = 0
        for i, (dim, stride, index) in enumerate(zip(self._shape, self._strides, idx)):
            if index < 0 or index >= dim:
                raise IndexError(f"Index {index} out of bounds for dimension {i} (size {dim})")
            byte_offset += index * stride
        
        return self._offset + byte_offset
    
    def _get_slice_indices(self, idx: Tuple[Union[int, slice, None], ...]) -> Tuple[Shape, Strides, int, int]:
        """
        Compute shape, strides, and offset for a slice.
        
        Args:
            idx: Slice tuple.
            
        Returns:
            Tuple of (new_shape, new_strides, new_offset, length).
        """
        new_shape = []
        new_strides = []
        new_offset = self._offset
        
        for i, (dim, stride, item) in enumerate(zip(self._shape, self._strides, idx)):
            if isinstance(item, int):
                # Integer index: dimension is removed
                new_offset += item * stride
                continue
            
            if isinstance(item, slice):
                start = item.start or 0
                stop = item.stop or dim
                step = item.step or 1
                
                if start < 0:
                    start = dim + start
                if stop < 0:
                    stop = dim + stop
                
                start = max(0, min(start, dim))
                stop = max(0, min(stop, dim))
                
                if start >= stop:
                    length = 0
                else:
                    length = (stop - start + step - 1) // step
                
                new_shape.append(length)
                new_strides.append(stride * step)
                new_offset += start * stride
            else:
                # None or Ellipsis - handle later
                raise NotImplementedError(f"Advanced indexing {item} not yet supported")
        
        if not new_shape:
            # Scalar
            new_shape = ()
            new_strides = ()
        
        return tuple(new_shape), tuple(new_strides), new_offset, len(new_shape)
    
    def getitem(self, idx: Union[int, Tuple[Union[int, slice, None], ...]]) -> Any:
        """
        Get item(s) from the buffer.
        
        Args:
            idx: Index or slice.
            
        Returns:
            Value (scalar) or TensorBuffer (sliced view).
        """
        if isinstance(idx, int):
            idx = (idx,)
        
        # Handle scalar indexing
        if all(isinstance(i, int) for i in idx):
            flat_idx = self._get_index(idx)
            arena = get_arena()
            data = arena.read(flat_idx, self._dtype.itemsize())
            
            # Convert bytes to Python value
            if self._dtype == DType.FLOAT64:
                return float.from_bytes(data, 'little')
            elif self._dtype == DType.FLOAT32:
                return float.from_bytes(data, 'little')
            elif self._dtype == DType.INT64:
                return int.from_bytes(data, 'little')
            elif self._dtype == DType.INT32:
                return int.from_bytes(data, 'little')
            else:
                raise ValueError(f"Unsupported dtype: {self._dtype}")
        
        # Handle slicing
        new_shape, new_strides, new_offset, ndim = self._get_slice_indices(idx)
        
        if new_shape == () and ndim == 0:
            # Scalar result
            arena = get_arena()
            data = arena.read(new_offset, self._dtype.itemsize())
            if self._dtype == DType.FLOAT64:
                return float.from_bytes(data, 'little')
            elif self._dtype == DType.FLOAT32:
                return float.from_bytes(data, 'little')
            elif self._dtype == DType.INT64:
                return int.from_bytes(data, 'little')
            elif self._dtype == DType.INT32:
                return int.from_bytes(data, 'little')
        
        # Return a view
        return TensorBuffer(
            offset=new_offset,
            shape=new_shape,
            strides=new_strides,
            dtype=self._dtype,
            readonly=self._readonly,
            owns_memory=False  # View doesn't own memory
        )
    
    def setitem(self, idx: Union[int, Tuple[int, ...]], value: Union[int, float]) -> None:
        """
        Set item in the buffer.
        
        Args:
            idx: Index.
            value: Value to set.
            
        Raises:
            TensorError: If buffer is read-only.
        """
        if self._readonly:
            raise TensorError("Cannot set value on read-only buffer")
        
        if isinstance(idx, int):
            idx = (idx,)
        
        flat_idx = self._get_index(idx)
        arena = get_arena()
        
        # Convert value to bytes
        if isinstance(value, float):
            if self._dtype == DType.FLOAT64:
                data = value.to_bytes(8, 'little')
            elif self._dtype == DType.FLOAT32:
                data = value.to_bytes(4, 'little')
            else:
                raise TypeError(f"Cannot assign float to dtype {self._dtype}")
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
            raise TypeError(f"Cannot assign type {type(value)} to buffer")
        
        arena.write(flat_idx, data)
    
    def fill(self, value: Union[int, float]) -> None:
        """
        Fill the entire buffer with a value.
        
        Args:
            value: Value to fill.
        """
        if self._readonly:
            raise TensorError("Cannot fill read-only buffer")
        
        arena = get_arena()
        
        if isinstance(value, float):
            if self._dtype == DType.FLOAT64:
                data = value.to_bytes(8, 'little') * self._size
            elif self._dtype == DType.FLOAT32:
                data = value.to_bytes(4, 'little') * self._size
            else:
                raise TypeError(f"Cannot fill dtype {self._dtype} with float")
        elif isinstance(value, int):
            if self._dtype in (DType.FLOAT64, DType.FLOAT32):
                value = float(value)
                if self._dtype == DType.FLOAT64:
                    data = value.to_bytes(8, 'little') * self._size
                else:
                    data = value.to_bytes(4, 'little') * self._size
            elif self._dtype == DType.INT64:
                data = value.to_bytes(8, 'little', signed=True) * self._size
            elif self._dtype == DType.INT32:
                data = value.to_bytes(4, 'little', signed=True) * self._size
            else:
                raise TypeError(f"Unsupported dtype: {self._dtype}")
        else:
            raise TypeError(f"Cannot fill with type {type(value)}")
        
        arena.write(self._offset, data)
    
    def copy(self) -> 'TensorBuffer':
        """
        Create a deep copy of the buffer.
        
        Returns:
            New TensorBuffer with copied data.
        """
        arena = get_arena()
        new_offset = arena.allocate(self.nbytes)
        
        # Copy data
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
        """
        Create a view of the buffer with a new shape.
        
        Args:
            shape: New shape (must have same number of elements).
            
        Returns:
            New TensorBuffer view.
        """
        new_shape = ShapeTracker.reshape(self._shape, shape)
        new_strides = compute_strides(new_shape, self._dtype.itemsize())
        
        return TensorBuffer(
            offset=self._offset,
            shape=new_shape,
            strides=new_strides,
            dtype=self._dtype,
            readonly=self._readonly,
            owns_memory=False
        )
    
    def to_bytes(self) -> bytes:
        """Return the raw bytes of the buffer."""
        arena = get_arena()
        return arena.read(self._offset, self.nbytes)
    
    def __len__(self) -> int:
        """Return the number of elements."""
        return self._size
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TensorBuffer(shape={self._shape}, dtype={self._dtype.name}, "
                f"size={self._size}, nbytes={self.nbytes})")
    
    def __del__(self):
        """Free memory if we own it."""
        if self._owns_memory and self._offset is not None:
            try:
                arena = get_arena()
                arena.free(self._offset)
            except:
                pass  # Ignore errors during cleanup