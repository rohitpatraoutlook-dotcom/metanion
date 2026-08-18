"""Algebraic simplification for Metanion."""

class RewriteSystem:
    def __init__(self):
        self._rules = []
    def normalize(self, handle):
        return handle

_REWRITE_SYSTEM = None

def get_rewrite_system():
    global _REWRITE_SYSTEM
    if _REWRITE_SYSTEM is None:
        _REWRITE_SYSTEM = RewriteSystem()
    return _REWRITE_SYSTEM

__all__ = ['get_rewrite_system']
