"""
IO module for the Metanion engine.
Handles serialization, deserialization, and checkpointing.
"""

from .binary_encoder import BinaryEncoder, BinaryHeader
from .binary_decoder import BinaryDecoder
from .checkpoint_manager import CheckpointManager, CheckpointMetadata

__all__ = [
    'BinaryEncoder',
    'BinaryHeader',
    'BinaryDecoder',
    'CheckpointManager',
    'CheckpointMetadata',
]