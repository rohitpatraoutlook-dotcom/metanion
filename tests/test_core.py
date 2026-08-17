"""
Test the fundamental tensor operations.
Run from project root: python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metanion import Tensor, DType, Shape
from metanion.core import get_arena, reset_arena


def test_tensor_creation():
    """Test basic tensor creation and properties."""
    print("\n=== Test 1: Tensor Creation ===")
    
    # Create from list
    t1 = Tensor([1, 2, 3, 4, 5])
    print(f"t1: {t1}")
    print(f"t1.shape: {t1.shape}")
    print(f"t1.dtype: {t1.dtype}")
    assert t1.shape == (5,), "Shape mismatch"
    
    # Create from nested list
    t2 = Tensor([[1, 2], [3, 4]])
    print(f"t2: {t2}")
    assert t2.shape == (2, 2), "Shape mismatch"
    
    # Create zeros
    t3 = Tensor.zeros((3, 4))
    print(f"zeros(3,4): {t3}")
    assert t3.shape == (3, 4), "Shape mismatch"
    
    # Create ones
    t4 = Tensor.ones((2, 3))
    print(f"ones(2,3): {t4}")
    
    print(" Tensor creation tests passed!")


def test_tensor_operations():
    """Test tensor arithmetic operations."""
    print("\n=== Test 2: Tensor Operations ===")
    
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    
    # Addition
    t3 = t1 + t2
    print(f"{t1} + {t2} = {t3}")
    
    # Subtraction
    t4 = t2 - t1
    print(f"{t2} - {t1} = {t4}")
    
    # Multiplication
    t5 = t1 * t2
    print(f"{t1} * {t2} = {t5}")
    
    # Scalar operations
    t6 = t1 + 10
    print(f"{t1} + 10 = {t6}")
    
    print(" Tensor operations tests passed!")


if __name__ == "__main__":
    reset_arena()
    test_tensor_creation()
    test_tensor_operations()
    print("\n Core tests completed successfully!")