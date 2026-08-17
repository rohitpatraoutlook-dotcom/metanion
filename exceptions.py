"""
Custom exceptions for the Metanion Engine.
These provide clear error messages for all failure modes.
"""


class MetanionError(Exception):
    """Base exception for all Metanion engine errors."""
    pass


class TensorError(MetanionError):
    """Raised for tensor-related errors (shape, dtype, memory)."""
    pass


class ShapeMismatchError(TensorError):
    """Raised when tensor shapes are incompatible for an operation."""
    pass


class MemoryAllocationError(TensorError):
    """Raised when the memory arena cannot allocate requested bytes."""
    pass


class DTypeError(TensorError):
    """Raised for dtype-related issues (unsupported, mismatch)."""
    pass


class PoolError(MetanionError):
    """Raised for expression pool errors."""
    pass


class PoolOverflowError(PoolError):
    """Raised when the pool exceeds maximum handle capacity."""
    pass


class HandleNotFoundError(PoolError):
    """Raised when a handle lookup fails."""
    pass


class ExpressionError(MetanionError):
    """Raised for expression tree errors."""
    pass


class TypeSignatureError(ExpressionError):
    """Raised when an operation's type signature is violated."""
    pass


class DepthLimitExceededError(ExpressionError):
    """Raised when expression depth exceeds MAX_DEPTH."""
    pass


class EvaluationError(MetanionError):
    """Raised during expression evaluation."""
    pass


class NumericError(EvaluationError):
    """Raised for numeric instability (inf, nan, division by zero)."""
    pass


class CompilationError(MetanionError):
    """Raised during JIT compilation."""
    pass


class GPRuntimeError(MetanionError):
    """Raised for genetic programming runtime issues."""
    pass


class SerializationError(MetanionError):
    """Raised for serialization/deserialization errors."""
    pass


class CheckpointError(SerializationError):
    """Raised for checkpoint saving/loading errors."""
    pass


class ConfigError(MetanionError):
    """Raised for configuration errors."""
    pass