"""
Binary decoder for the Metanion engine.
Deserializes models and expressions from binary format.
"""

from typing import Optional, List, Dict, Any, Tuple, BinaryIO
import struct
import json
import hashlib

from ..symbolic import OpID, intern, lookup, get_pool, get_op_name
from ..symbolic import ExpressionNode, ExpressionNodeFactory
from ..model import MetanionModel, MetanionLayer, ModelConfig
from ..core import DType
from ..exceptions import SerializationError
from .binary_encoder import BinaryHeader


class BinaryDecoder:
    """
    Decodes Metanion models and expressions from binary format.
    """
    
    def __init__(self):
        """Initialize the binary decoder."""
        self._op_codec: Dict[int, OpID] = {}
        self._build_op_codec()
        self._handle_cache: Dict[int, int] = {}  # position -> handle
        self._pool = get_pool()
    
    def _build_op_codec(self) -> None:
        """Build operation codec for deserialization."""
        op_ids = [op for op in OpID]
        for i, op in enumerate(op_ids):
            self._op_codec[i] = op
    
    def decode_handle(self, data: bytes, offset: int) -> Tuple[int, int]:
        """
        Decode a handle from bytes.
        
        Args:
            data: The data to decode.
            offset: Current offset in the data.
            
        Returns:
            Tuple of (handle, new_offset).
        """
        # Read op_code (2 bytes) + left (4 bytes) + right (4 bytes)
        op_code, left, right = struct.unpack('>HII', data[offset:offset + 2 + 4 + 4])
        offset += 2 + 4 + 4
        
        op = self._op_codec.get(op_code)
        if op is None:
            raise SerializationError(f"Unknown op code: {op_code}")
        
        # Adjust child handles (they are relative)
        left_handle = None if left == 0 else left
        right_handle = None if right == 0 else right
        
        # Check if this handle already exists in the pool
        existing = self._pool.get_handle(op, left_handle, right_handle)
        if existing is not None:
            return existing, offset
        
        # Create new handle
        handle = self._pool.intern(op, left_handle, right_handle)
        return handle, offset
    
    def decode_expression_pool(self, data: bytes, offset: int) -> int:
        """
        Decode the expression pool from bytes.
        
        Args:
            data: The data to decode.
            offset: Current offset in the data.
            
        Returns:
            New offset after decoding.
        """
        # Read number of handles
        num_handles = struct.unpack('>I', data[offset:offset + 4])[0]
        offset += 4
        
        # Decode each handle
        for _ in range(num_handles):
            handle, offset = self.decode_handle(data, offset)
        
        return offset
    
    def decode_layer(self, data: bytes, offset: int) -> Tuple[MetanionLayer, int]:
        """
        Decode a layer from bytes.
        
        Args:
            data: The data to decode.
            offset: Current offset in the data.
            
        Returns:
            Tuple of (layer, new_offset).
        """
        # Read layer header
        index, in_features, out_features, num_weights = struct.unpack(
            '>IIII', data[offset:offset + 16]
        )
        offset += 16
        
        # Read use_bias flag
        use_bias = struct.unpack('>B', data[offset:offset + 1])[0] == 1
        offset += 1
        
        # Create layer
        layer = MetanionLayer(
            in_features=in_features,
            out_features=out_features,
            dtype=DType.FLOAT64,
            use_bias=use_bias
        )
        
        # Decode weights
        weight_handles = []
        for _ in range(out_features):
            row = []
            for _ in range(in_features):
                handle, offset = self.decode_handle(data, offset)
                row.append(handle)
            weight_handles.append(row)
        layer.weight_handles = weight_handles
        
        # Decode bias
        if use_bias:
            bias_handles = []
            for _ in range(out_features):
                handle, offset = self.decode_handle(data, offset)
                bias_handles.append(handle)
            layer.bias_handles = bias_handles
        
        return layer, offset
    
    def decode_model(self, data: bytes) -> Tuple[MetanionModel, Dict[str, Any]]:
        """
        Decode a model from bytes.
        
        Args:
            data: The data to decode.
            
        Returns:
            Tuple of (model, metadata).
        """
        offset = 0
        
        # Read header
        header, offset = BinaryHeader.unpack(data)
        
        # Verify magic
        if header.magic != b'METN':
            raise SerializationError(f"Invalid magic: {header.magic}")
        
        # Create model config
        config = ModelConfig(
            layer_sizes=[],  # Will be determined from layers
            use_bias=True,
            max_depth=10
        )
        
        # Decode layers
        layers = []
        for _ in range(header.metadata.get('num_layers', 0)):
            layer, offset = self.decode_layer(data, offset)
            layers.append(layer)
        
        # Decode training history
        history_len = struct.unpack('>I', data[offset:offset + 4])[0]
        offset += 4
        history = json.loads(data[offset:offset + history_len].decode('utf-8'))
        offset += history_len
        
        # Decode expression pool
        pool_len = struct.unpack('>I', data[offset:offset + 4])[0]
        offset += 4
        offset = self.decode_expression_pool(data, offset)
        
        # Build model from layers
        if layers:
            layer_sizes = [layers[0].in_features]
            for layer in layers:
                layer_sizes.append(layer.out_features)
            config.layer_sizes = layer_sizes
        
        model = MetanionModel(config)
        model.stack.layers = layers
        model.stack.num_layers = len(layers)
        model.stack.input_size = layer_sizes[0] if layer_sizes else 1
        model.stack.output_size = layer_sizes[-1] if layer_sizes else 1
        
        # Restore training state
        model._training_history = history
        model._is_trained = header.metadata.get('trained', False)
        
        return model, header.metadata
    
    def decode_from_file(self, filepath: str) -> MetanionModel:
        """
        Decode a model from a file.
        
        Args:
            filepath: Path to the input file.
            
        Returns:
            The decoded model.
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        model, _ = self.decode_model(data)
        return model
    
    def decode_from_bytes(self, data: bytes) -> MetanionModel:
        """
        Decode a model from bytes.
        
        Args:
            data: The data to decode.
            
        Returns:
            The decoded model.
        """
        model, _ = self.decode_model(data)
        return model
    
    def verify_checksum(self, data: bytes, expected: str) -> bool:
        """Verify checksum of data."""
        actual = hashlib.sha256(data).hexdigest()
        return actual == expected