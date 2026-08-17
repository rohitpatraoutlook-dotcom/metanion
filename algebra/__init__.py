"""
Algebraic simplification and term rewriting for the Metanion engine.
"""

from .rewrite_rules import (
    Pattern,
    RewriteRule,
    RewriteSystem,
    RewriteDirection,
    get_rewrite_system,
    simplify,
)

__all__ = [
    'Pattern',
    'RewriteRule',
    'RewriteSystem',
    'RewriteDirection',
    'get_rewrite_system',
    'simplify',
]