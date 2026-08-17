"""
Compilation and JIT engine for the Metanion engine.
Handles lazy evaluation, bytecode generation, and optimization.
"""

from .lazy_graph import (
    LazyOp,
    LazyOpType,
    LazyGraph,
)

from .straight_line_program import (
    SLPOpType,
    SLPInstruction,
    StraightLineProgram,
)

from .bytecode_compiler import (
    BytecodeCompiler,
    OptimizedBytecodeCompiler,
    get_compiler,
    compile_program,
    compile_handle,
)

__all__ = [
    'LazyOp',
    'LazyOpType',
    'LazyGraph',
    'SLPOpType',
    'SLPInstruction',
    'StraightLineProgram',
    'BytecodeCompiler',
    'OptimizedBytecodeCompiler',
    'get_compiler',
    'compile_program',
    'compile_handle',
]