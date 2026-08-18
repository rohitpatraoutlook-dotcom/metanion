"""Model module for Metanion."""

try:
    from .metanion_model import MetanionModel
except ImportError:
    MetanionModel = None

__all__ = ['MetanionModel']
