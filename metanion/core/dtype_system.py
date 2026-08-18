"""
Data type system for the Metanion Tensor engine.
Provides dtype abstraction without relying on NumPy.
"""

from enum import Enum
from typing import Optional


class DType(Enum):
    """Supported data types."""
    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4
    
    def itemsize(self) -> int:
        """Return bytes per element."""
        if self in (DType.FLOAT32, DType.INT32):
            return 4
        elif self in (DType.FLOAT64, DType.INT64):
            return 8
        else:
            raise ValueError(f"Unknown dtype: {self}")
    
    def is_float(self) -> bool:
        """Return True if dtype is floating point."""
        return self in (DType.FLOAT32, DType.FLOAT64)
    
    def is_int(self) -> bool:
        """Return True if dtype is integer."""
        return self in (DType.INT32, DType.INT64)
    
    def to_numpy(self):
        """Return NumPy dtype equivalent (if NumPy is available)."""
        import numpy as np
        mapping = {
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
        }
        return mapping[self]
    
    @classmethod
    def from_numpy(cls, np_dtype):
        """Convert NumPy dtype to Metanion DType."""
        import numpy as np
        mapping = {
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.dtype('float32'): cls.FLOAT32,
            np.dtype('float64'): cls.FLOAT64,
            np.dtype('int32'): cls.INT32,
            np.dtype('int64'): cls.INT64,
        }
        if np_dtype not in mapping:
            # Default to float64 for safety
            return cls.FLOAT64
        return mapping[np_dtype]
    
    @classmethod
    def default(cls) -> 'DType':
        """Return default dtype (float64)."""
        return cls.FLOAT64


def promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    """
    Determine the result dtype for binary operations.
    Promotes to the higher precision type.
    """
    # Simple promotion: float64 > float32 > int64 > int32
    priority = {
        DType.FLOAT64: 4,
        DType.FLOAT32: 3,
        DType.INT64: 2,
        DType.INT32: 1,
    }
    if priority[dtype1] >= priority[dtype2]:
        return dtype1
    return dtype2


def is_numeric_dtype(dtype: DType) -> bool:
    """Return True if dtype is numeric (float or int)."""
    return dtype in (DType.FLOAT32, DType.FLOAT64, DType.INT32, DType.INT64)