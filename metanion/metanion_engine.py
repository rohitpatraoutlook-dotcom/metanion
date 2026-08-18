"""
Main Metanion engine entry point.
"""

from .core import Tensor
from .symbolic import OpID, intern, lookup, simplify
from .compile import compile_handle, get_compiler


class MetanionEngine:
    def __init__(self):
        self._version = "0.1.0"
    
    def get_version(self):
        return self._version
    
    def compile(self, handle):
        return compile_handle(handle)
    
    def simplify(self, handle):
        return simplify(handle)


_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MetanionEngine()
    return _ENGINE

def create_model(layer_sizes, **kwargs):
    """Create a model."""
    from .model import MetanionModel
    print(f"Creating model with layers: {layer_sizes}")
    return MetanionModel()


def train(X, y, **kwargs):
    """Train a model."""
    print("Training model...")
    return {}


def predict(X):
    """Predict using model."""
    return X

__all__ = ['MetanionEngine', 'get_engine', 'create_model', 'train', 'predict']
