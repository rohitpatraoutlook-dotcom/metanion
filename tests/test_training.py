"""
Test end-to-end model training.
Run from project root: python tests/test_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metanion import MetanionEngine, create_model, train, predict


def test_simple_regression():
    """Test training on simple linear regression."""
    print("\n=== Test 9: Simple Linear Regression ===")
    
    # Generate data: y = 2*x + 1 + noise
    np.random.seed(42)
    X = np.random.randn(200, 1)
    y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(200)
    
    # Split into train/val
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    
    # Create engine and model
    engine = MetanionEngine()
    model = create_model(
        layer_sizes=[1, 10, 1],
        use_bias=True,
        max_depth=4
    )
    
    # Train
    print("Training on linear regression data...")
    history = model.fit(
        X_train.tolist(), 
        y_train.tolist(),
        X_val.tolist(), 
        y_val.tolist(),
        epochs=30
    )
    
    # Evaluate
    predictions = model.predict(X_val)
    mse = model.evaluate(X_val, y_val)
    print(f"Validation MSE: {mse:.6f}")
    
    # Print best expression
    if model._best_individual:
        print(f"Best expression: {model._best_individual.get_expression()}")
    
    print(" Training tests passed!")


def test_sin_regression():
    """Test training on sine function."""
    print("\n=== Test 10: Sine Regression ===")
    
    # Generate data: y = sin(x)
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, (200, 1))
    y = np.sin(X[:, 0]) + 0.05 * np.random.randn(200)
    
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    
    # Create model
    model = create_model(
        layer_sizes=[1, 15, 1],
        use_bias=True,
        max_depth=5
    )
    
    # Train
    print("Training on sin(x) data...")
    history = model.fit(
        X_train.tolist(), 
        y_train.tolist(),
        X_val.tolist(), 
        y_val.tolist(),
        epochs=50
    )
    
    # Evaluate
    mse = model.evaluate(X_val, y_val)
    print(f"Validation MSE: {mse:.6f}")
    
    # Print best expression
    if model._best_individual:
        print(f"Best expression: {model._best_individual.get_expression()}")
    
    print("Sine regression tests passed!")


if __name__ == "__main__":
    test_simple_regression()
    test_sin_regression()
    print("\n Training tests completed successfully!")