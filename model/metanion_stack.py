"""
Stack of Metanion layers for building deep models.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field

from ..core import Tensor, DType, Shape
from .metanion_layer import MetanionLayer
from ..exceptions import ShapeMismatchError


@dataclass
class StackConfig:
    """Configuration for a layer stack."""
    layer_sizes: List[int] = field(default_factory=list)
    use_bias: bool = True
    max_depth: int = 5
    dtype: DType = DType.FLOAT64


class MetanionStack:
    """
    A stack of Metanion layers forming a deep model.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        use_bias: bool = True,
        max_depth: int = 5,
        dtype: DType = DType.FLOAT64
    ):
        """
        Initialize a stack of layers.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output].
            use_bias: Whether to use bias in all layers.
            max_depth: Maximum depth of expressions in all layers.
            dtype: Data type for tensor operations.
        """
        if len(layer_sizes) < 2:
            raise ValueError("Need at least 2 layers (input and output)")
        
        self.layer_sizes = layer_sizes
        self.use_bias = use_bias
        self.max_depth = max_depth
        self.dtype = dtype
        
        # Create layers
        self.layers: List[MetanionLayer] = []
        for i in range(len(layer_sizes) - 1):
            layer = MetanionLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                dtype=dtype,
                max_depth=max_depth,
                use_bias=use_bias
            )
            self.layers.append(layer)
        
        self.num_layers = len(self.layers)
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after all layers.
        """
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer.forward(current)
            
            # Check shape consistency
            if i < len(self.layers) - 1:
                # Intermediate layer: shape should be (batch_size, layer_sizes[i+1])
                pass
        
        return current
    
    def simplify(self) -> None:
        """Simplify all layers."""
        for layer in self.layers:
            layer.simplify()
    
    def get_depth(self) -> int:
        """Get the maximum depth across all layers."""
        return max(layer.get_depth() for layer in self.layers)
    
    def get_node_count(self) -> int:
        """Get the total number of nodes across all layers."""
        return sum(layer.get_node_count() for layer in self.layers)
    
    def get_layer(self, idx: int) -> MetanionLayer:
        """Get a specific layer."""
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")
        return self.layers[idx]
    
    def add_layer(self, layer: MetanionLayer) -> None:
        """Add a layer to the stack."""
        # Check compatibility
        if self.layers:
            prev_out = self.layers[-1].out_features
            if layer.in_features != prev_out:
                raise ShapeMismatchError(
                    f"Layer input features {layer.in_features} "
                    f"does not match previous output {prev_out}"
                )
        
        self.layers.append(layer)
        self.num_layers = len(self.layers)
        self.output_size = layer.out_features
    
    def get_layer_shapes(self) -> List[Tuple[int, int]]:
        """Get shapes of all layers."""
        return [(layer.in_features, layer.out_features) for layer in self.layers]
    
    def __repr__(self) -> str:
        """String representation."""
        shapes = self.get_layer_shapes()
        return (f"MetanionStack(layers={self.num_layers}, "
                f"shapes={shapes}, "
                f"total_depth={self.get_depth()}, "
                f"total_nodes={self.get_node_count()})")