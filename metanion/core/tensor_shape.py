"""
Tensor shape and stride management for the Metanion engine.
Handles shape validation, broadcasting, and memory layout.
"""

from typing import Tuple, List, Optional, Union
from functools import lru_cache

from ..exceptions import ShapeMismatchError


Shape = Tuple[int, ...]
Strides = Tuple[int, ...]


class ShapeTracker:
    """
    Manages tensor shapes and stride calculations.
    Supports broadcasting and shape manipulation.
    """
    
    @staticmethod
    def validate_shape(shape: Shape) -> None:
        """
        Validate that a shape tuple is valid.
        
        Args:
            shape: Shape tuple to validate.
            
        Raises:
            ValueError: If shape is invalid.
        """
        if not isinstance(shape, tuple):
            raise ValueError(f"Shape must be tuple, got {type(shape)}")
        
        for dim in shape:
            if not isinstance(dim, int):
                raise ValueError(f"Shape dimension must be int, got {type(dim)}")
            if dim < 0:
                raise ValueError(f"Shape dimension cannot be negative: {dim}")
    
    @staticmethod
    def compute_strides(shape: Shape, dtype_size: int = 8) -> Strides:
        """
        Compute C-contiguous (row-major) strides for a given shape.
        
        Args:
            shape: Shape tuple.
            dtype_size: Size of each element in bytes.
            
        Returns:
            Strides tuple.
        """
        ShapeTracker.validate_shape(shape)
        
        strides = []
        stride = dtype_size
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        
        return tuple(reversed(strides))
    
    @staticmethod
    def compute_size(shape: Shape) -> int:
        """
        Compute total number of elements in a tensor.
        
        Args:
            shape: Shape tuple.
            
        Returns:
            Total number of elements.
        """
        if not shape:
            return 1  # Scalar tensor
        
        size = 1
        for dim in shape:
            size *= dim
        return size
    
    @staticmethod
    def broadcast_shape(shape1: Shape, shape2: Shape) -> Shape:
        """
        Compute the broadcasted shape of two tensors.
        
        Args:
            shape1: First shape.
            shape2: Second shape.
            
        Returns:
            Broadcasted shape.
            
        Raises:
            ShapeMismatchError: If shapes are incompatible.
        """
        # Reverse both shapes for easier alignment
        rev1 = list(reversed(shape1))
        rev2 = list(reversed(shape2))
        
        # Pad smaller shape with ones
        max_len = max(len(rev1), len(rev2))
        rev1 += [1] * (max_len - len(rev1))
        rev2 += [1] * (max_len - len(rev2))
        
        result = []
        for d1, d2 in zip(rev1, rev2):
            if d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            elif d1 == d2:
                result.append(d1)
            else:
                raise ShapeMismatchError(
                    f"Cannot broadcast shapes {shape1} and {shape2} "
                    f"at dimension {d1} vs {d2}"
                )
        
        return tuple(reversed(result))
    
    @staticmethod
    def broadcast_strides(shape1: Shape, strides1: Strides,
                          shape2: Shape, strides2: Strides) -> Tuple[Strides, Strides]:
        """
        Compute strides for broadcasted tensors.
        If a dimension is broadcasted, its stride becomes 0.
        
        Args:
            shape1: First shape.
            strides1: First strides.
            shape2: Second shape.
            strides2: Second strides.
            
        Returns:
            Tuple of (broadcasted_strides1, broadcasted_strides2).
        """
        broadcast_shape = ShapeTracker.broadcast_shape(shape1, shape2)
        
        # Align shapes to broadcast shape
        aligned1 = [1] * (len(broadcast_shape) - len(shape1)) + list(shape1)
        aligned2 = [1] * (len(broadcast_shape) - len(shape2)) + list(shape2)
        
        # Compute broadcasted strides
        bstrides1 = []
        bstrides2 = []
        
        for i, (d1, d2) in enumerate(zip(aligned1, aligned2)):
            # For broadcasted dimensions, stride is 0
            if d1 == 1 and d2 != 1:
                bstrides1.append(0)
            else:
                # Find original index for this dimension
                orig_idx = i - (len(broadcast_shape) - len(shape1))
                if orig_idx >= 0:
                    bstrides1.append(strides1[orig_idx] if orig_idx < len(strides1) else 0)
                else:
                    bstrides1.append(0)
            
            if d2 == 1 and d1 != 1:
                bstrides2.append(0)
            else:
                orig_idx = i - (len(broadcast_shape) - len(shape2))
                if orig_idx >= 0:
                    bstrides2.append(strides2[orig_idx] if orig_idx < len(strides2) else 0)
                else:
                    bstrides2.append(0)
        
        return tuple(bstrides1), tuple(bstrides2)
    
    @staticmethod
    def reshape(shape: Shape, new_shape: Shape) -> Shape:
        """
        Reshape a tensor to new dimensions if compatible.
        
        Args:
            shape: Original shape.
            new_shape: Target shape.
            
        Returns:
            Validated new shape.
            
        Raises:
            ShapeMismatchError: If reshape is impossible.
        """
        # Handle -1 in new_shape (infer dimension)
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ShapeMismatchError(
                    f"Cannot infer multiple dimensions: {new_shape}"
                )
            
            total_elements = ShapeTracker.compute_size(shape)
            known_elements = 1
            inferred_idx = -1
            
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    inferred_idx = i
                else:
                    known_elements *= dim
            
            if total_elements % known_elements != 0:
                raise ShapeMismatchError(
                    f"Cannot reshape {shape} to {new_shape}: "
                    f"{total_elements} not divisible by {known_elements}"
                )
            
            inferred_dim = total_elements // known_elements
            new_shape = tuple(
                inferred_dim if i == inferred_idx else dim
                for i, dim in enumerate(new_shape)
            )
        
        # Verify total elements match
        if ShapeTracker.compute_size(shape) != ShapeTracker.compute_size(new_shape):
            raise ShapeMismatchError(
                f"Cannot reshape {shape} to {new_shape}: "
                f"element count mismatch "
                f"({ShapeTracker.compute_size(shape)} vs {ShapeTracker.compute_size(new_shape)})"
            )
        
        ShapeTracker.validate_shape(new_shape)
        return new_shape
    
    @staticmethod
    def squeeze(shape: Shape, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Shape:
        """
        Remove dimensions of size 1.
        
        Args:
            shape: Original shape.
            dim: Specific dimension(s) to squeeze.
            
        Returns:
            Squeezed shape.
        """
        ShapeTracker.validate_shape(shape)
        
        if dim is None:
            return tuple(d for d in shape if d != 1)
        
        if isinstance(dim, int):
            dim = (dim,)
        
        result = []
        for i, d in enumerate(shape):
            if i in dim and d == 1:
                continue
            result.append(d)
        
        return tuple(result)
    
    @staticmethod
    def unsqueeze(shape: Shape, dim: int) -> Shape:
        """
        Add a dimension of size 1 at the specified position.
        
        Args:
            shape: Original shape.
            dim: Position to insert the new dimension.
            
        Returns:
            New shape with dimension inserted.
        """
        ShapeTracker.validate_shape(shape)
        
        if dim < 0:
            dim = len(shape) + dim + 1
        
        if dim < 0 or dim > len(shape):
            raise ValueError(f"Invalid dimension {dim} for shape {shape}")
        
        result = list(shape)
        result.insert(dim, 1)
        return tuple(result)
    
    @staticmethod
    def concat_shapes(shapes: List[Shape], axis: int) -> Shape:
        """
        Compute shape after concatenation along an axis.
        
        Args:
            shapes: List of input shapes.
            axis: Axis to concatenate along.
            
        Returns:
            Concatenated shape.
            
        Raises:
            ShapeMismatchError: If shapes are incompatible.
        """
        if not shapes:
            raise ValueError("At least one shape required")
        
        # Validate all shapes except concat axis match
        base_shape = list(shapes[0])
        total_axis_dim = 0
        
        for i, shape in enumerate(shapes):
            if len(shape) != len(base_shape):
                raise ShapeMismatchError(
                    f"Shapes must have same rank: {base_shape} vs {shape}"
                )
            
            for j, dim in enumerate(shape):
                if j == axis:
                    total_axis_dim += dim
                elif dim != base_shape[j]:
                    raise ShapeMismatchError(
                        f"Shapes mismatch at dimension {j}: "
                        f"{base_shape[j]} vs {dim}"
                    )
        
        result = list(base_shape)
        result[axis] = total_axis_dim
        return tuple(result)
    
    @staticmethod
    def is_scalar(shape: Shape) -> bool:
        """Check if shape represents a scalar (empty tuple)."""
        return shape == ()
    
    @staticmethod
    def is_vector(shape: Shape) -> bool:
        """Check if shape represents a vector (1D)."""
        return len(shape) == 1
    
    @staticmethod
    def is_matrix(shape: Shape) -> bool:
        """Check if shape represents a matrix (2D)."""
        return len(shape) == 2
    
    @staticmethod
    def is_ndim(shape: Shape, ndim: int) -> bool:
        """Check if shape has the given number of dimensions."""
        return len(shape) == ndim
    
    @staticmethod
    def normalize_axis(axis: int, ndim: int) -> int:
        """
        Normalize a negative axis index.
        
        Args:
            axis: Axis index (can be negative).
            ndim: Number of dimensions.
            
        Returns:
            Normalized axis index in [0, ndim-1].
        """
        if axis < 0:
            axis = ndim + axis
        
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Axis {axis} out of bounds for shape with {ndim} dimensions")
        
        return axis


# Convenience aliases
def broadcast_shape(shape1: Shape, shape2: Shape) -> Shape:
    """Alias for ShapeTracker.broadcast_shape."""
    return ShapeTracker.broadcast_shape(shape1, shape2)


def compute_strides(shape: Shape, dtype_size: int = 8) -> Strides:
    """Alias for ShapeTracker.compute_strides."""
    return ShapeTracker.compute_strides(shape, dtype_size)


def compute_size(shape: Shape) -> int:
    """Alias for ShapeTracker.compute_size."""
    return ShapeTracker.compute_size(shape)


def is_scalar(shape: Shape) -> bool:
    """Alias for ShapeTracker.is_scalar."""
    return ShapeTracker.is_scalar(shape)


def is_vector(shape: Shape) -> bool:
    """Alias for ShapeTracker.is_vector."""
    return ShapeTracker.is_vector(shape)


def is_matrix(shape: Shape) -> bool:
    """Alias for ShapeTracker.is_matrix."""
    return ShapeTracker.is_matrix(shape)