"""
Runtime module for the Metanion engine.
Manages JIT cache and garbage collection.
"""

from .jit_cache_manager import JITCacheManager, get_jit_cache
from .gc_controller import GCController, get_gc_controller, collect_garbage, suppress_gc

__all__ = [
    'JITCacheManager',
    'get_jit_cache',
    'GCController',
    'get_gc_controller',
    'collect_garbage',
    'suppress_gc',
]