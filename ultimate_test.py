"""
METANION ULTIMATE TEST SUITE - Island GP Default
"""

import sys
import os
sys.path.insert(0, '/data/data/com.termux/files/home/metanion/research')
from metanion_research import run_gp, print_expr, test_expression
from metanion import compile_handle
import numpy as np
import time


def dataset_linear(n=200, noise=0.05):
    X = np.random.uniform(-5, 5, (n, 1))
    y = 2 * X + 1 + noise * np.random.randn(n, 1)
    return X, y, "y = 2x + 1"


def dataset_polynomial(n=200, noise=0.1):
    X = np.random.uniform(-2, 2, (n, 1))
    y = X ** 3 - 2 * X ** 2 + X + 1 + noise * np.random.randn(n, 1)
    return X, y, "y = x^3 - 2x^2 + x + 1"


def dataset_trigonometric(n=200, noise=0.05):
    X = np.random.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(X) + np.cos(2 * X) + noise * np.random.randn(n, 1)
    return X, y, "y = sin(x) + cos(2x)"


def dataset_exponential(n=200, noise=0.05):
    X = np.random.uniform(-2, 2, (n, 1))
    y = np.exp(X / 2) + noise * np.random.randn(n, 1)
    return X, y, "y = e^(x/2)"


def dataset_logarithmic(n=200, noise=0.05):
    X = np.random.uniform(-3, 3, (n, 1))
    y = np.log(np.abs(X) + 1) + noise * np.random.randn(n, 1)
    return X, y, "y = log(|x| + 1)"


def dataset_sinc(n=200, noise=0.05):
    X = np.random.uniform(-10, 10, (n, 1))
    X = X[X != 0].reshape(-1, 1)
    X = X[:n]
    y = np.sin(X) / X + noise * np.random.randn(n, 1)
    return X, y, "y = sin(x)/x"


def dataset_abs_sin(n=200, noise=0.05):
    X = np.random.uniform(-np.pi, np.pi, (n, 1))
    y = np.abs(np.sin(X)) + noise * np.random.randn(n, 1)
    return X, y, "y = |sin(x)|"


def dataset_rational(n=200, noise=0.05):
    X = np.random.uniform(-5, 5, (n, 1))
    y = X / (1 + X ** 2) + noise * np.random.randn(n, 1)
    return X, y, "y = x / (1 + x^2)"


def dataset_multivariate(n=200, noise=0.05):
    X = np.random.uniform(-5, 5, (n, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5 + noise * np.random.randn(n)
    y = y.reshape(-1, 1)
    return X, y, "y = 2*x0 + 3*x1 - x2 + 5"


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
        ("Multivariate", dataset_multivariate),
    ]

    print("=" * 80)
    print("METANION ULTIMATE TEST SUITE (Island GP)")
    print("=" * 80)
    print(f"Total Tests: {len(datasets)}\n")

    results = {}
    total_time = 0
    passed = 0
    failed = 0

    for name, dataset_func in datasets:
        try:
            print(f"▶ Testing {name}...")

            X, y, desc = dataset_func()
            print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
            print(f"  Target: {desc}")

            start = time.time()
            best = run_gp(X, y, pop_size=100, generations=40, max_depth=4,
                          optimize_constants=True, verbose=False, random_seed=42)
            elapsed = time.time() - start
            total_time += elapsed

            X_test = X[:30]
            y_test = y[:30]

            f = compile_handle(best.weight_handles[0], n_features=X.shape[1])
            preds = np.array([f(list(x)) for x in X_test])

            mse = np.mean((preds - y_test.flatten()) ** 2)

            expr = print_expr(best.weight_handles[0], [f"x{i}" for i in range(X.shape[1])])
            if len(expr) > 80:
                expr = expr[:80] + "..."

            print(f"  Expression: {expr}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Depth: {best.depth}, Nodes: {best.node_count}")
            print(f"  Time: {elapsed:.2f}s")

            success = (mse < 0.5) and (best.depth < 12)
            if success:
                passed += 1
                results[name] = {"status": "PASS", "mse": mse, "time": elapsed}
            else:
                failed += 1
                results[name] = {"status": "FAIL", "mse": mse, "time": elapsed}

        except Exception as e:
            failed += 1
            results[name] = {"status": f"ERROR: {str(e)[:40]}", "mse": -1, "time": 0}
            print(f"  ERROR: {e}")

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Test':<15} | {'Status':<10} | {'MSE':<12} | {'Time':<10}")
    print("-" * 55)

    for name, result in results.items():
        mse_str = f"{result['mse']:.6f}" if result['mse'] >= 0 else "N/A"
        print(f"{name:<15} | {result['status']:<10} | {mse_str:<12} | {result['time']:<10.2f}")

    print("-" * 55)
    print(f"Total: {len(datasets)} | Passed: {passed} | Failed: {failed} | Time: {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
