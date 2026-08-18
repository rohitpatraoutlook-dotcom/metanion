"""
Comprehensive Metanion Test Suite
Tests on different data types to identify problems.
"""

import sys
sys.path.insert(0, '/data/data/com.termux/files/home/metanion/research')
from metanion_research import run_gp, print_expr, test_expression
import numpy as np
import time

def test_linear():
    """Test 1: Linear data y = 2x + 1"""
    print("\n" + "="*60)
    print("TEST 1: Linear (y = 2x + 1)")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (100, 1))
    y = 2 * X + 1 + 0.05 * np.random.randn(100, 1)
    
    start = time.time()
    best = run_gp(X, y, pop_size=100, generations=50, max_depth=3, 
                  safe=True, add_bias=True, verbose=False, random_seed=42)
    elapsed = time.time() - start
    
    test_X = np.array([[-3.0], [0.0], [3.0]])
    true = 2 * test_X + 1
    mse, _ = test_expression(best.weight_handles[0], test_X, true)
    
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"MSE: {mse:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")
    
    return mse < 0.1


def test_polynomial():
    """Test 2: Polynomial y = x^3 - 2x^2 + x + 1"""
    print("\n" + "="*60)
    print("TEST 2: Polynomial (y = x^3 - 2x^2 + x + 1)")
    print("="*60)
    
    np.random.seed(123)
    X = np.random.uniform(-2, 2, (150, 1))
    y = X**3 - 2*X**2 + X + 1 + 0.1 * np.random.randn(150, 1)
    
    start = time.time()
    best = run_gp(X, y, pop_size=150, generations=80, max_depth=4,
                  safe=True, add_bias=True, verbose=False, random_seed=123)
    elapsed = time.time() - start
    
    test_X = np.array([[-1.5], [-0.5], [0.0], [0.5], [1.5]])
    true = test_X**3 - 2*test_X**2 + test_X + 1
    mse, _ = test_expression(best.weight_handles[0], test_X, true)
    
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"MSE: {mse:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")
    
    return mse < 0.1


def test_trigonometric():
    """Test 3: Trigonometric y = sin(x) + cos(x)"""
    print("\n" + "="*60)
    print("TEST 3: Trigonometric (y = sin(x) + cos(x))")
    print("="*60)
    
    np.random.seed(456)
    X = np.random.uniform(-np.pi, np.pi, (150, 1))
    y = np.sin(X) + np.cos(X) + 0.05 * np.random.randn(150, 1)
    
    start = time.time()
    best = run_gp(X, y, pop_size=150, generations=80, max_depth=4,
                  safe=True, add_bias=False, verbose=False, random_seed=456)
    elapsed = time.time() - start
    
    test_X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    true = np.sin(test_X) + np.cos(test_X)
    mse, _ = test_expression(best.weight_handles[0], test_X, true)
    
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"MSE: {mse:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")
    
    return mse < 0.1


def test_exponential():
    """Test 4: Exponential y = e^(x/2)"""
    print("\n" + "="*60)
    print("TEST 4: Exponential (y = e^(x/2))")
    print("="*60)
    
    np.random.seed(789)
    X = np.random.uniform(-2, 2, (150, 1))
    y = np.exp(X/2) + 0.05 * np.random.randn(150, 1)
    
    start = time.time()
    best = run_gp(X, y, pop_size=150, generations=80, max_depth=4,
                  safe=True, add_bias=False, verbose=False, random_seed=789)
    elapsed = time.time() - start
    
    test_X = np.array([[-1.5], [-0.5], [0.0], [0.5], [1.5]])
    true = np.exp(test_X/2)
    mse, _ = test_expression(best.weight_handles[0], test_X, true)
    
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"MSE: {mse:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")
    
    return mse < 0.1


def test_noisy_data():
    """Test 5: Noisy data with high noise"""
    print("\n" + "="*60)
    print("TEST 5: Noisy (y = 2x + 1 with high noise)")
    print("="*60)
    
    np.random.seed(101)
    X = np.random.uniform(-5, 5, (100, 1))
    y = 2 * X + 1 + 0.5 * np.random.randn(100, 1)  # High noise
    
    start = time.time()
    best = run_gp(X, y, pop_size=100, generations=50, max_depth=3,
                  safe=True, add_bias=True, verbose=False, random_seed=101)
    elapsed = time.time() - start
    
    test_X = np.array([[-3.0], [0.0], [3.0]])
    true = 2 * test_X + 1
    mse, _ = test_expression(best.weight_handles[0], test_X, true)
    
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"MSE: {mse:.6f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")
    
    return mse < 1.0  # Higher tolerance for noise


def main():
    print("="*80)
    print("🧪 METANION COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing on different data types...")
    
    results = {}
    tests = [
        ("Linear (2x+1)", test_linear),
        ("Polynomial (x^3...)", test_polynomial),
        ("Trigonometric (sin+cos)", test_trigonometric),
        ("Exponential (e^(x/2))", test_exponential),
        ("Noisy Data", test_noisy_data),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✅ PASS" if success else "⚠️ FAIL"
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results[name] = f"❌ ERROR: {str(e)[:50]}"
            failed += 1
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    for name, result in results.items():
        print(f"  {name:30} : {result}")
    print("-"*40)
    print(f"  Passed: {passed}, Failed: {failed}")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
