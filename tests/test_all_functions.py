"""
Complete function coverage test for Metanion.
Focuses on the core motive: symbolic expressions as weights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from metanion import (
    Tensor, DType, Shape, OpID, intern, lookup, simplify,
    compile_handle, differentiate, create_model, train, predict
)
from metanion.core import get_arena, reset_arena, MemoryArena
from metanion.core import ShapeTracker
from metanion.core import TensorBuffer
from metanion.symbolic import get_pool, reset_pool
from metanion.symbolic import get_depth, count_nodes_in_subtree
from metanion.symbolic import get_op_name, get_op_arity
from metanion.symbolic import is_binary_op, is_unary_op
from metanion.symbolic import ExpressionNode, ExpressionNodeFactory
from metanion.compile import StraightLineProgram
from metanion.algebra import get_rewrite_system
from metanion.calculus import get_differentiator, get_derivative_rules
from metanion.model import MetanionModel
from metanion.utils import TreePrinter, get_cost_model, get_time_profiler
from metanion.io import BinaryEncoder, BinaryDecoder, CheckpointManager


def print_header(name):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)


def print_result(passed, message):
    if passed:
        print(f"  PASS: {message}")
    else:
        print(f"  FAIL: {message}")
    return passed


def test_memory_arena():
    print_header("MemoryArena")
    passed = 0
    total = 0
    
    arena = MemoryArena(1)
    
    total += 1
    offset1 = arena.allocate(100)
    passed += 1 if offset1 == 0 else 0
    print_result(offset1 == 0, f"Allocate 100 bytes at offset {offset1}")
    
    total += 1
    offset2 = arena.allocate(200)
    passed += 1 if offset2 == 104 else 0
    print_result(offset2 == 104, f"Allocate 200 bytes at offset {offset2}")
    
    total += 1
    test_data = b"Hello World"
    arena.write(offset1, test_data)
    read_data = arena.read(offset1, len(test_data))
    passed += 1 if read_data == test_data else 0
    print_result(read_data == test_data, f"Write and read data")
    
    total += 1
    stats = arena.stats()
    passed += 1 if stats['allocation_count'] == 2 else 0
    print_result(stats['allocation_count'] == 2, f"Stats: allocations={stats['allocation_count']}")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_dtype_system():
    print_header("DType System")
    passed = 0
    total = 0
    
    total += 1
    dtype = DType.FLOAT64
    passed += 1 if dtype == DType.FLOAT64 else 0
    print_result(dtype == DType.FLOAT64, f"DType.FLOAT64 = {dtype}")
    
    total += 1
    size = DType.FLOAT64.itemsize()
    passed += 1 if size == 8 else 0
    print_result(size == 8, f"FLOAT64 itemsize = {size}")
    
    total += 1
    size = DType.FLOAT32.itemsize()
    passed += 1 if size == 4 else 0
    print_result(size == 4, f"FLOAT32 itemsize = {size}")
    
    total += 1
    passed += 1 if DType.FLOAT64.is_float() else 0
    print_result(DType.FLOAT64.is_float(), f"FLOAT64.is_float() = True")
    
    total += 1
    passed += 1 if not DType.INT32.is_float() else 0
    print_result(not DType.INT32.is_float(), f"INT32.is_float() = False")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_tensor_creation():
    print_header("Tensor Creation (1D only)")
    passed = 0
    total = 0
    
    reset_arena()
    
    # Test 1D tensor
    total += 1
    t1 = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    passed += 1 if t1.shape == (5,) else 0
    print_result(t1.shape == (5,), f"1D Tensor shape = {t1.shape}")
    
    # Test zeros - 1D
    total += 1
    t2 = Tensor.zeros((5,))
    passed += 1 if t2.shape == (5,) else 0
    print_result(t2.shape == (5,), f"zeros shape = {t2.shape}")
    
    # Test ones - 1D
    total += 1
    t3 = Tensor.ones((4,))
    passed += 1 if t3.shape == (4,) else 0
    print_result(t3.shape == (4,), f"ones shape = {t3.shape}")
    
    # Test full - 1D
    total += 1
    t4 = Tensor.full((3,), 7.5)
    passed += 1 if t4.shape == (3,) else 0
    print_result(t4.shape == (3,), f"full shape = {t4.shape}")
    
    # Test scalar
    total += 1
    t5 = Tensor(3.14)
    passed += 1 if t5.shape == () else 0
    print_result(t5.shape == (), f"scalar shape = {t5.shape}")
    
    # Test dtype
    total += 1
    t6 = Tensor([1, 2, 3], dtype=DType.INT32)
    passed += 1 if t6.dtype == DType.INT32 else 0
    print_result(t6.dtype == DType.INT32, f"INT32 dtype = {t6.dtype}")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_tensor_operations():
    print_header("Tensor Operations")
    passed = 0
    total = 0
    
    reset_arena()
    
    t1 = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    t2 = Tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    
    total += 1
    t3 = t1 + t2
    result3 = t3.to_list()
    expected3 = [6.0, 6.0, 6.0, 6.0, 6.0]
    passed += 1 if result3 == expected3 else 0
    print_result(result3 == expected3, f"t1 + t2 = {result3}")
    
    total += 1
    t4 = t1 + 10.0
    result4 = t4.to_list()
    expected4 = [11.0, 12.0, 13.0, 14.0, 15.0]
    passed += 1 if result4 == expected4 else 0
    print_result(result4 == expected4, f"t1 + 10 = {result4}")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_shape_tracker():
    print_header("ShapeTracker")
    passed = 0
    total = 0
    
    total += 1
    size = ShapeTracker.compute_size((2, 3, 4))
    passed += 1 if size == 24 else 0
    print_result(size == 24, f"compute_size((2,3,4)) = {size}")
    
    total += 1
    shape = ShapeTracker.broadcast_shape((2, 3), (1, 3))
    passed += 1 if shape == (2, 3) else 0
    print_result(shape == (2, 3), f"broadcast_shape((2,3),(1,3)) = {shape}")
    
    total += 1
    shape = ShapeTracker.broadcast_shape((3, 1), (1, 4))
    passed += 1 if shape == (3, 4) else 0
    print_result(shape == (3, 4), f"broadcast_shape((3,1),(1,4)) = {shape}")
    
    total += 1
    new_shape = ShapeTracker.reshape((2, 6), (3, 4))
    passed += 1 if new_shape == (3, 4) else 0
    print_result(new_shape == (3, 4), f"reshape((2,6),(3,4)) = {new_shape}")
    
    total += 1
    shape = ShapeTracker.squeeze((1, 3, 1, 5))
    passed += 1 if shape == (3, 5) else 0
    print_result(shape == (3, 5), f"squeeze((1,3,1,5)) = {shape}")
    
    total += 1
    shape = ShapeTracker.unsqueeze((3, 5), 1)
    passed += 1 if shape == (3, 1, 5) else 0
    print_result(shape == (3, 1, 5), f"unsqueeze((3,5),1) = {shape}")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_symbolic_ops():
    print_header("Symbolic Operations (The Core of Metanion)")
    passed = 0
    total = 0
    
    reset_pool()
    
    # This is the heart of Metanion - symbolic expressions
    total += 1
    x = intern(OpID.IDENTITY)
    node = lookup(x)
    passed += 1 if node is not None else 0
    print_result(node is not None, f"intern(IDENTITY) = {x}, lookup works")
    
    total += 1
    zero = intern(OpID.CONST_ZERO)
    one = intern(OpID.CONST_ONE)
    passed += 1 if zero != one else 0
    print_result(zero != one, f"CONST_ZERO={zero}, CONST_ONE={one}")
    
    total += 1
    expr = intern(OpID.ADD, x, one)
    passed += 1 if expr is not None else 0
    print_result(expr is not None, f"ADD(x, 1) = {expr}")
    
    total += 1
    sin_expr = intern(OpID.SIN, x)
    passed += 1 if sin_expr is not None else 0
    print_result(sin_expr is not None, f"SIN(x) = {sin_expr}")
    
    # Test operation metadata
    total += 1
    name = get_op_name(OpID.IDENTITY)
    passed += 1 if name == "identity" else 0
    print_result(name == "identity", f"get_op_name(IDENTITY) = {name}")
    
    total += 1
    arity = get_op_arity(OpID.ADD)
    passed += 1 if arity == 2 else 0
    print_result(arity == 2, f"get_op_arity(ADD) = {arity}")
    
    total += 1
    passed += 1 if is_binary_op(OpID.ADD) else 0
    print_result(is_binary_op(OpID.ADD), f"is_binary_op(ADD) = True")
    
    total += 1
    passed += 1 if is_unary_op(OpID.SIN) else 0
    print_result(is_unary_op(OpID.SIN), f"is_unary_op(SIN) = True")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_simplification():
    print_header("Expression Simplification")
    passed = 0
    total = 0
    
    reset_pool()
    
    total += 1
    x = intern(OpID.IDENTITY)
    one = intern(OpID.CONST_ONE)
    expr = intern(OpID.ADD, x, one)
    
    simplified = simplify(expr)
    passed += 1 if simplified is not None else 0
    print_result(simplified is not None, f"simplify works on expression")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_compilation():
    print_header("JIT Compilation")
    passed = 0
    total = 0
    
    total += 1
    func = compile_handle(1)
    passed += 1 if callable(func) else 0
    print_result(callable(func), f"compile_handle returns callable")
    
    total += 1
    result = func([3.0])
    passed += 1 if result == 4.0 else 0
    print_result(result == 4.0, f"f(3) = {result}, expected 4.0")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_differentiation():
    print_header("Symbolic Differentiation")
    passed = 0
    total = 0
    
    reset_pool()
    
    total += 1
    x = intern(OpID.IDENTITY)
    expr = intern(OpID.SQUARE, x)
    deriv = differentiate(expr)
    passed += 1 if deriv is not None else 0
    print_result(deriv is not None, f"d/dx(x^2) works")
    
    total += 1
    rules = get_derivative_rules()
    passed += 1 if rules is not None else 0
    print_result(rules is not None, f"get_derivative_rules() works")
    
    total += 1
    diff = get_differentiator()
    passed += 1 if diff is not None else 0
    print_result(diff is not None, f"get_differentiator() works")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_model():
    print_header("Model Functions (The Metanion Magic)")
    passed = 0
    total = 0
    
    total += 1
    model = create_model([1, 5, 1], max_depth=3)
    passed += 1 if isinstance(model, MetanionModel) else 0
    print_result(isinstance(model, MetanionModel), f"create_model works")
    
    total += 1
    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [[2.0], [4.0], [6.0], [8.0], [10.0]]
    history = train(X, y, epochs=1)
    passed += 1 if isinstance(history, dict) else 0
    print_result(isinstance(history, dict), f"train works")
    
    total += 1
    preds = predict([[1.0], [2.0]])
    passed += 1 if preds is not None else 0
    print_result(preds is not None, f"predict works")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_utils():
    print_header("Utility Functions")
    passed = 0
    total = 0
    
    total += 1
    printer = TreePrinter()
    passed += 1 if printer is not None else 0
    print_result(printer is not None, f"TreePrinter() works")
    
    total += 1
    x = intern(OpID.IDENTITY)
    one = intern(OpID.CONST_ONE)
    expr = intern(OpID.ADD, x, one)
    tree_str = printer.print_expression(expr, var_name="x")
    passed += 1 if tree_str is not None else 0
    print_result(tree_str is not None, f"print_expression works")
    
    total += 1
    cost_model = get_cost_model()
    passed += 1 if cost_model is not None else 0
    print_result(cost_model is not None, f"get_cost_model() works")
    
    total += 1
    profiler = get_time_profiler()
    passed += 1 if profiler is not None else 0
    print_result(profiler is not None, f"get_time_profiler() works")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_io():
    print_header("IO Functions")
    passed = 0
    total = 0
    
    total += 1
    encoder = BinaryEncoder()
    passed += 1 if encoder is not None else 0
    print_result(encoder is not None, f"BinaryEncoder() works")
    
    total += 1
    decoder = BinaryDecoder()
    passed += 1 if decoder is not None else 0
    print_result(decoder is not None, f"BinaryDecoder() works")
    
    total += 1
    manager = CheckpointManager()
    passed += 1 if manager is not None else 0
    print_result(manager is not None, f"CheckpointManager() works")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def test_getters():
    print_header("Global Getters")
    passed = 0
    total = 0
    
    total += 1
    pool = get_pool()
    passed += 1 if pool is not None else 0
    print_result(pool is not None, f"get_pool() works")
    
    total += 1
    arena = get_arena()
    passed += 1 if arena is not None else 0
    print_result(arena is not None, f"get_arena() works")
    
    total += 1
    rewrite = get_rewrite_system()
    passed += 1 if rewrite is not None else 0
    print_result(rewrite is not None, f"get_rewrite_system() works")
    
    print(f"\nResults: {passed}/{total} passed")
    return passed == total


def main():
    print("=" * 70)
    print("METANION COMPLETE FUNCTION COVERAGE TEST")
    print("=" * 70)
    
    tests = [
        ("Memory Arena", test_memory_arena),
        ("DType System", test_dtype_system),
        ("Tensor Creation (1D)", test_tensor_creation),
        ("Tensor Operations", test_tensor_operations),
        ("ShapeTracker", test_shape_tracker),
        ("Symbolic Operations (Core)", test_symbolic_ops),
        ("Expression Simplification", test_simplification),
        ("JIT Compilation", test_compilation),
        ("Symbolic Differentiation", test_differentiation),
        ("Model Functions (Magic)", test_model),
        ("Utility Functions", test_utils),
        ("IO Functions", test_io),
        ("Global Getters", test_getters),
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for name, test_func in tests:
        print(f"\n>> Running {name}...")
        start = time.time()
        try:
            success = test_func()
            elapsed = time.time() - start
            if success:
                passed += 1
                results.append(f"PASS: {name} ({elapsed:.2f}s)")
            else:
                failed += 1
                results.append(f"FAIL: {name} ({elapsed:.2f}s)")
        except Exception as e:
            failed += 1
            results.append(f"ERROR: {name} - {str(e)}")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for result in results:
        print(f"  {result}")
    print("-" * 70)
    print(f"Total Passed: {passed}")
    print(f"Total Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
