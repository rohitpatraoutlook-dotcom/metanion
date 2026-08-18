"""Runtime module for Metanion."""

class JITCacheManager:
    pass

class GCController:
    pass

_JIT_CACHE = None
_GC_CONTROLLER = None

def get_jit_cache():
    global _JIT_CACHE
    if _JIT_CACHE is None:
        _JIT_CACHE = JITCacheManager()
    return _JIT_CACHE

def get_gc_controller():
    global _GC_CONTROLLER
    if _GC_CONTROLLER is None:
        _GC_CONTROLLER = GCController()
    return _GC_CONTROLLER

__all__ = ['get_jit_cache', 'get_gc_controller']
