"""Algebraic simplification and term rewriting for the Metanion engine."""

# Simple rewrite system placeholder
class RewriteSystem:
    def __init__(self):
        self._rules = []
    
    def add_rule(self, rule):
        self._rules.append(rule)
    
    def normalize(self, handle):
        return handle


# Global rewrite system
_REWRITE_SYSTEM = None


def get_rewrite_system():
    """Get or create the global rewrite system."""
    global _REWRITE_SYSTEM
    if _REWRITE_SYSTEM is None:
        _REWRITE_SYSTEM = RewriteSystem()
    return _REWRITE_SYSTEM


def simplify(handle):
    """Simplify an expression using the rewrite system."""
    return get_rewrite_system().normalize(handle)


__all__ = [
    'RewriteSystem',
    'get_rewrite_system',
    'simplify',
]
