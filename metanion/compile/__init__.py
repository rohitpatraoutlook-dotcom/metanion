"""Compilation module for Metanion."""

from .bytecode_compiler import compile_handle

def get_compiler():
    """Get the compiler instance."""
    from .bytecode_compiler import BytecodeCompiler
    return BytecodeCompiler()

class StraightLineProgram:
    """Straight-line program representation."""
    def __init__(self, handle):
        self.handle = handle
    def evaluate(self, inputs):
        if not inputs:
            return 0.0
        return inputs[0]

__all__ = ['compile_handle', 'get_compiler', 'StraightLineProgram']
