"""
METANION ULTIMATE TEST SUITE
Tests all data types in one go.
Run on Google Colab or any Python environment.
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_friedman1, make_friedman2

# Install metanion if not already installed
try:
    import metanion
except ImportError:
    !pip install git+https://github.com/rohitpatraoutlook-dotcom/metanion.git

# Now import
from metanion_research import run_gp, print_expr, test_expression


# ============================================================================
# 1. TEST DATA GENERATORS
# ============================================================================

def dataset_linear(n=200, noise=0.05):
    """y = 2x + 1"""
    X = np.random.uniform(-5, 5, (n, 1))
    y = 2 * X + 1 + noise * np.random.randn(n, 1)
    return X, y, "y = 2x + 1"


def dataset_polynomial(n=200, noise=0.1):
    """y = x^3 - 2x^2 + x + 1"""
    X = np.random.uniform(-2, 2, (n, 1))
    y = X**3 - 2*X**2 + X + 1 + noise * np.random.randn(n, 1)
    return X, y, "y = x^3 - 2x^2 + x + 1"


def dataset_trigonometric(n=200, noise=0.05):
    """y = sin(x) + cos(2x)"""
    X = np.random.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(X) + np.cos(2*X) + noise * np.random.randn(n, 1)
    return X, y, "y = sin(x) + cos(2x)"


def dataset_exponential(n=200, noise=0.05):
    """y = e^(x/2)"""
    X = np.random.uniform(-2, 2, (n, 1))
    y = np.exp(X/2) + noise * np.random.randn(n, 1)
    return X, y, "y = e^(x/2)"


def dataset_logarithmic(n=200, noise=0.05):
    """y = log(|x| + 1)"""
    X = np.random.uniform(-3, 3, (n, 1))
    y = np.log(np.abs(X) + 1) + noise * np.random.randn(n, 1)
    return X, y, "y = log(|x| + 1)"


def dataset_friedman(n=200, noise=0.1):
    """Friedman #1: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5"""
    X, y = make_friedman1(n_samples=n, n_features=5, noise=noise, random_state=42)
    y = y.reshape(-1, 1)
    return X, y, "Friedman #1 (5 features)"


def dataset_multivariate_linear(n=200, noise=0.05):
    """y = 2*x1 + 3*x2 - x3 + 5"""
    X = np.random.uniform(-5, 5, (n, 3))
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 5 + noise * np.random.randn(n)
    y = y.reshape(-1, 1)
    return X, y, "y = 2*x1 + 3*x2 - x3 + 5"


def dataset_sinc(n=200, noise=0.05):
    """y = sin(x)/x"""
    X = np.random.uniform(-10, 10, (n, 1))
    X = X[X != 0].reshape(-1, 1)[:n]
    y = np.sin(X) / X + noise * np.random.randn(n, 1)
    return X, y, "y = sin(x)/x"


def dataset_abs_sin(n=200, noise=0.05):
    """y = |sin(x)|"""
    X = np.random.uniform(-np.pi, np.pi, (n, 1))
    y = np.abs(np.sin(X)) + noise * np.random.randn(n, 1)
    return X, y, "y = |sin(x)|"


def dataset_rational(n=200, noise=0.05):
    """y = x / (1 + x^2)"""
    X = np.random.uniform(-5, 5, (n, 1))
    y = X / (1 + X**2) + noise * np.random.randn(n, 1)
    return X, y, "y = x / (1 + x^2)"


# ============================================================================
# 2. RUN ALL TESTS
# ============================================================================

def run_all_tests():
    datasets = [
        ("Linear", dataset_linear),
        ("Polynomial", dataset_polynomial),
        ("Trigonometric", dataset_trigonometric),
        ("Exponential", dataset_exponential),
        ("Logarithmic", dataset_logarithmic),
        ("Sinc", dataset_sinc),
        ("Abs Sin", dataset_abs_sin),
        ("Rational", dataset_rational),
        ("Multivariate", dataset_multivariate_linear),
        ("Friedman", dataset_friedman),
    ]

    print("=" * 80)
    print("METANION ULTIMATE TEST SUITE")
    print("=" * 80)
    print(f"Total Tests: {len(datasets)}\n")

    results = {}
    total_time = 0
    passed = 0
    failed = 0

    for name, dataset_func in datasets:
        try:
            print(f"▶ Testing {name}...")
            
            # Generate data
            X, y, desc = dataset_func()
            print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
            print(f"  Target: {desc}")
            
            # Run GP
            start = time.time()
            best = run_gp(X, y, pop_size=100, generations=50, max_depth=3,
                          safe=True, add_bias=True, verbose=False, random_seed=42)
            elapsed = time.time() - start
            total_time += elapsed
            
            # Test
            test_size = min(50, len(X) // 2)
            X_test = X[:test_size]
            y_test = y[:test_size]
            
            # Use compiled function directly
            from metanion import compile_handle
            f = compile_handle(best.weight_handles[0])
            preds = np.array([f(float(x[0])) if x.shape[0] == 1 else 0 for x in X_test])
            
            if X.shape[1] == 1:
                mse = np.mean((preds - y_test.flatten())**2)
            else:
                # For multivariate, we need a different approach
                # For now, just use the first feature
                mse = np.mean((preds - y_test.flatten())**2)
            
            # Check if expression is reasonable
            expr = print_expr(best.weight_handles[0])
            depth = best.depth
            nodes = best.node_count
            
            print(f"  Expression: {expr[:60]}..." if len(expr) > 60 else f"  Expression: {expr}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Depth: {depth}, Nodes: {nodes}")
            print(f"  Time: {elapsed:.2f}s")
            
            # Simple pass/fail: MSE < 1.0 and depth < 10
            success = (mse < 1.0) and (depth < 10)
            if success:
                passed += 1
                results[name] = {"status": "✅ PASS", "mse": mse, "time": elapsed}
            else:
                failed += 1
                results[name] = {"status": "⚠️ FAIL", "mse": mse, "time": elapsed}
                
        except Exception as e:
            failed += 1
            results[name] = {"status": f"❌ ERROR: {str(e)[:50]}", "mse": -1, "time": 0}
        
        print()

    # Summary
    print("=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Test Name':<20} | {'Status':<12} | {'MSE':<12} | {'Time (s)':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        mse_str = f"{result['mse']:.6f}" if result['mse'] >= 0 else "N/A"
        print(f"{name:<20} | {result['status']:<12} | {mse_str:<12} | {result['time']:<10.2f}")
    
    print("-" * 60)
    print(f"\nTotal Tests: {len(datasets)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
