"""Compilation module for Metanion."""

from .bytecode_compiler import compile_handle
from .straight_line_program import StraightLineProgram

__all__ = ['compile_handle', 'StraightLineProgram']
