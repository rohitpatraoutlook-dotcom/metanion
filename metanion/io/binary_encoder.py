"""
Binary encoder for the Metanion engine.
Serializes models and expressions to binary format.
"""

from typing import Optional, List, Dict, Any, Tuple, BinaryIO
import struct
import json
import hashlib
from dataclasses import dataclass, field

from ..symbolic import OpID, get_op_name, lookup, intern, get_pool
from ..symbolic import get_op_metadata, get_op_arity
from ..model import MetanionModel, MetanionLayer
from ..exceptions import SerializationError


@dataclass
class BinaryHeader:
    """Header for binary serialization."""
    
    magic: bytes = b'METN'
    version: int = 1
    checksum: Optional[str] = None
    compression: str = 'none'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def pack(self) -> bytes:
        """Pack header to bytes."""
        # Magic (4 bytes) + Version (4 bytes) + Compression (4 bytes)
        # + Metadata length (4 bytes) + Checksum length (4 bytes)
        metadata_bytes = json.dumps(self.metadata).encode('utf-8')
        checksum_bytes = (self.checksum or '').encode('utf-8')
        
        header = struct.pack(
            '>4sIIIHH',
            self.magic,
            self.version,
            len(metadata_bytes),
            len(checksum_bytes),
            len(self.compression)
        )
        header += self.compression.encode('utf-8')
        header += metadata_bytes
        header += checksum_bytes
        
        return header
    
    @classmethod
    def unpack(cls, data: bytes) -> Tuple['BinaryHeader', int]:
        """Unpack header from bytes."""
        offset = 0
        magic, version, meta_len, checksum_len, comp_len = struct.unpack(
            '>4sIIIHH',
            data[offset:offset + 4 + 4 + 4 + 4 + 4 + 2]
        )
        offset += 4 + 4 + 4 + 4 + 4 + 2
        
        compression = data[offset:offset + comp_len].decode('utf-8')
        offset += comp_len
        
        metadata = json.loads(data[offset:offset + meta_len].decode('utf-8'))
        offset += meta_len
        
        checksum = data[offset:offset + checksum_len].decode('utf-8') if checksum_len > 0 else None
        offset += checksum_len
        
        return cls(
            magic=magic,
            version=version,
            checksum=checksum,
            compression=compression,
            metadata=metadata
        ), offset


class BinaryEncoder:
    """
    Encodes Metanion models and expressions to binary format.
    """
    
    def __init__(self):
        """Initialize the binary encoder."""
        self._op_codec: Dict[OpID, int] = {}
        self._reverse_op_codec: Dict[int, OpID] = {}
        self._build_op_codec()
    
    def _build_op_codec(self) -> None:
        """Build operation codec for compact serialization."""
        op_ids = [op for op in OpID]
        for i, op in enumerate(op_ids):
            self._op_codec[op] = i
            self._reverse_op_codec[i] = op
    
    def encode_handle(self, handle: int) -> bytes:
        """
        Encode a single handle to bytes.
        
        Args:
            handle: The handle to encode.
            
        Returns:
            Encoded bytes.
        """
        node = lookup(handle)
        if node is None:
            raise SerializationError(f"Handle {handle} not found")
        
        # Encode as: op_code (2 bytes) + left_handle (4 bytes) + right_handle (4 bytes)
        op_code = self._op_codec.get(node.op, 0)
        left = node.left or 0
        right = node.right or 0
        
        return struct.pack('>HII', op_code, left, right)
    
    def encode_expression_pool(self) -> bytes:
        """Encode the entire expression pool."""
        pool = get_pool()
        handles = pool.get_all_handles()
        
        # Header: number of handles (4 bytes)
        result = struct.pack('>I', len(handles))
        
        # Encode each handle
        for handle in sorted(handles):
            result += self.encode_handle(handle)
        
        return result
    
    def encode_model(self, model: MetanionModel) -> bytes:
        """
        Encode a complete model to bytes.
        
        Args:
            model: The model to encode.
            
        Returns:
            Encoded bytes.
        """
        # Header
        header = BinaryHeader(
            metadata={
                'num_layers': model.stack.num_layers,
                'input_size': model.stack.input_size,
                'output_size': model.stack.output_size,
                'trained': model._is_trained,
                'generations': len(model._training_history.get('generations', [])),
                'best_fitness': model._training_history.get('best_fitness', [None])[-1] if model._training_history.get('best_fitness') else None
            }
        )
        
        result = header.pack()
        
        # Encode each layer
        for i, layer in enumerate(model.stack.layers):
            result += self.encode_layer(layer, i)
        
        # Encode training history
        history = json.dumps(model._training_history).encode('utf-8')
        result += struct.pack('>I', len(history))
        result += history
        
        # Encode expression pool
        pool_data = self.encode_expression_pool()
        result += struct.pack('>I', len(pool_data))
        result += pool_data
        
        return result
    
    def encode_layer(self, layer: MetanionLayer, index: int) -> bytes:
        """
        Encode a single layer to bytes.
        
        Args:
            layer: The layer to encode.
            index: Layer index.
            
        Returns:
            Encoded bytes.
        """
        result = b''
        
        # Layer header
        result += struct.pack(
            '>IIII',
            index,
            layer.in_features,
            layer.out_features,
            len(layer.weight_handles)
        )
        result += struct.pack('>B', 1 if layer.use_bias else 0)
        
        # Encode weights
        for row in layer.weight_handles:
            for handle in row:
                result += self.encode_handle(handle)
        
        # Encode bias
        if layer.use_bias:
            for handle in layer.bias_handles:
                result += self.encode_handle(handle)
        
        return result
    
    def encode_to_file(self, model: MetanionModel, filepath: str) -> None:
        """
        Encode a model to a file.
        
        Args:
            model: The model to encode.
            filepath: Path to the output file.
        """
        data = self.encode_model(model)
        with open(filepath, 'wb') as f:
            f.write(data)
    
    def encode_to_bytes(self, model: MetanionModel) -> bytes:
        """
        Encode a model to bytes.
        
        Args:
            model: The model to encode.
            
        Returns:
            Encoded bytes.
        """
        return self.encode_model(model)
    
    def get_checksum(self, data: bytes) -> str:
        """Compute checksum for data."""
        return hashlib.sha256(data).hexdigest()