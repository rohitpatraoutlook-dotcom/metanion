"""
Metanion layer implementation.
A layer where weights are symbolic expressions instead of numbers.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np

from ..core import Tensor, DType, Shape
from ..symbolic import OpID, intern, lookup, get_pool, get_op_name
from ..symbolic import simplify, get_depth, count_nodes_in_subtree
from ..compile import compile_handle
from ..exceptions import ExpressionError, ShapeMismatchError


class MetanionLayer:
    """
    A layer in the Metanion engine.
    Weight matrix where each element is a symbolic expression.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: DType = DType.FLOAT64,
        max_depth: int = 5,
        use_bias: bool = True
    ):
        """
        Initialize a Metanion layer.
        
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            dtype: Data type for tensor operations.
            max_depth: Maximum depth of expression trees.
            use_bias: Whether to include a bias term.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.max_depth = max_depth
        self.use_bias = use_bias
        
        # Weight matrix: out_features x in_features
        self.weight_handles: List[List[int]] = []
        self.bias_handles: List[int] = []
        
        # Cache
        self._compiled_weights: Optional[List[List[callable]]] = None
        self._compiled_bias: Optional[List[callable]] = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with simple expressions."""
        pool = get_pool()
        
        # Initialize weights
        self.weight_handles = []
        for _ in range(self.out_features):
            row = []
            for _ in range(self.in_features):
                # Identity by default
                handle = intern(OpID.IDENTITY)
                row.append(handle)
            self.weight_handles.append(row)
        
        # Initialize bias
        if self.use_bias:
            for _ in range(self.out_features):
                handle = intern(OpID.CONST_ZERO)
                self.bias_handles.append(handle)
    
    def _get_weight_handle(self, row: int, col: int) -> int:
        """Get the handle at position (row, col)."""
        if row >= self.out_features or col >= self.in_features:
            raise IndexError(f"Index out of bounds: ({row}, {col})")
        return self.weight_handles[row][col]
    
    def _set_weight_handle(self, row: int, col: int, handle: int) -> None:
        """Set the handle at position (row, col)."""
        if row >= self.out_features or col >= self.in_features:
            raise IndexError(f"Index out of bounds: ({row}, {col})")
        self.weight_handles[row][col] = handle
        self._compiled_weights = None  # Invalidate cache
    
    def _get_bias_handle(self, idx: int) -> int:
        """Get the bias handle at index."""
        if not self.use_bias:
            raise ExpressionError("Layer does not use bias")
        if idx >= self.out_features:
            raise IndexError(f"Index out of bounds: {idx}")
        return self.bias_handles[idx]
    
    def _set_bias_handle(self, idx: int, handle: int) -> None:
        """Set the bias handle at index."""
        if not self.use_bias:
            raise ExpressionError("Layer does not use bias")
        if idx >= self.out_features:
            raise IndexError(f"Index out of bounds: {idx}")
        self.bias_handles[idx] = handle
        self._compiled_bias = None  # Invalidate cache
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features).
            
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        batch_size = x.shape[0] if len(x.shape) > 0 else 1
        
        # Ensure input shape matches
        if len(x.shape) >= 2 and x.shape[-1] != self.in_features:
            raise ShapeMismatchError(
                f"Expected input features {self.in_features}, got {x.shape[-1]}"
            )
        
        # If input is 1D, reshape to (1, in_features)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Get numpy for evaluation
        try:
            x_np = x.numpy()
        except:
            # If numpy conversion fails, try to get as list
            x_np = np.array(x) if hasattr(x, '__array__') else x
        
        # Initialize output
        y = np.zeros((batch_size, self.out_features))
        
        # For each output feature
        for i in range(self.out_features):
            # Compute weighted sum
            for j in range(self.in_features):
                handle = self._get_weight_handle(i, j)
                try:
                    func = self._compile_handle(handle)
                    # Evaluate on each sample
                    for b in range(batch_size):
                        val = x_np[b, j] if len(x_np.shape) > 1 else x_np[j]
                        result = func([float(val)])
                        y[b, i] += result
                except Exception:
                    # If evaluation fails, skip
                    pass
            
            # Add bias
            if self.use_bias:
                try:
                    bias_func = self._compile_handle(self._get_bias_handle(i))
                    for b in range(batch_size):
                        y[b, i] += bias_func([1.0])
                except Exception:
                    pass
        
        # Convert to Tensor
        return Tensor(y.tolist(), dtype=self.dtype)
    
    def _compile_handle(self, handle: int) -> callable:
        """Compile a handle to a callable function."""
        try:
            return compile_handle(handle)
        except Exception:
            # Fallback: identity function
            return lambda x: x[0] if x else 0.0
    
    def simplify(self) -> None:
        """Simplify all expressions in the layer."""
        pool = get_pool()
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                handle = self._get_weight_handle(i, j)
                if handle is not None:
                    self._set_weight_handle(i, j, simplify(handle))
            
            if self.use_bias:
                handle = self._get_bias_handle(i)
                if handle is not None:
                    self._set_bias_handle(i, simplify(handle))
        
        # Invalidate cache
        self._compiled_weights = None
        self._compiled_bias = None
    
    def get_depth(self) -> int:
        """Get the maximum depth of expressions in this layer."""
        pool = get_pool()
        max_depth = 0
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                handle = self._get_weight_handle(i, j)
                if handle is not None:
                    depth = get_depth(handle, pool.get_node)
                    max_depth = max(max_depth, depth)
            
            if self.use_bias:
                handle = self._get_bias_handle(i)
                if handle is not None:
                    depth = get_depth(handle, pool.get_node)
                    max_depth = max(max_depth, depth)
        
        return max_depth
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in this layer."""
        pool = get_pool()
        total = 0
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                handle = self._get_weight_handle(i, j)
                if handle is not None:
                    total += count_nodes_in_subtree(handle, pool.get_node)
            
            if self.use_bias:
                handle = self._get_bias_handle(i)
                if handle is not None:
                    total += count_nodes_in_subtree(handle, pool.get_node)
        
        return total
    
    def get_weight_expression(self, row: int, col: int) -> str:
        """Get string representation of a weight expression."""
        handle = self._get_weight_handle(row, col)
        node = lookup(handle)
        if node is None:
            return "None"
        return get_op_name(node.op)
    
    def get_bias_expression(self, idx: int) -> str:
        """Get string representation of a bias expression."""
        if not self.use_bias:
            return "No bias"
        handle = self._get_bias_handle(idx)
        node = lookup(handle)
        if node is None:
            return "None"
        return get_op_name(node.op)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MetanionLayer(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"depth={self.get_depth()}, "
                f"nodes={self.get_node_count()}, "
                f"use_bias={self.use_bias})")