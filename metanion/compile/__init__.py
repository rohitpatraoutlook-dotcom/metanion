"""Compilation module for Metanion."""

def compile_handle(handle):
    """Compile a handle to a callable function."""
    return lambda x: x

class StraightLineProgram:
    """Straight-line program representation."""
    def __init__(self, handle):
        self.handle = handle
    
    def evaluate(self, inputs):
        return inputs[0] if inputs else 0

__all__ = ['compile_handle', 'StraightLineProgram']
