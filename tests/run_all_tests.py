"""
Run all test suites from the tests/ directory.
Run from project root: python tests/run_all_tests.py
"""

import sys
import os
import time
import importlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all():
    """Run all test modules."""
    print("=" * 60)
    print("METANION ENGINE TEST SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    test_modules = [
        ('test_core', 'Core Tensor Engine'),
        ('test_symbolic', 'Symbolic Expression System'),
        ('test_compile', 'JIT Compilation'),
        ('test_calculus', 'Symbolic Differentiation'),
        ('test_training', 'End-to-End Training'),
        ('test_io', 'Serialization'),
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for module_name, description in test_modules:
        print(f"\n Testing {description}...")
        print("-" * 40)
        try:
            # Import and run the test module
            module = importlib.import_module(module_name)
            # Run the module directly (it will execute its main)
            if hasattr(module, 'main'):
                module.main()
            passed += 1
            results.append(f" {description}: PASSED")
        except Exception as e:
            print(f" Error: {e}")
            failed += 1
            results.append(f" {description}: FAILED - {e}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for result in results:
        print(result)
    print("-" * 60)
    print(f" Tests Passed: {passed}")
    print(f" Tests Failed: {failed}")
    print(f"  Total Time: {elapsed:.2f}s")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)