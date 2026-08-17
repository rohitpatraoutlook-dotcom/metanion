"""
Simple regression example using Metanion.
Run from project root: python examples/simple_regression.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metanion import MetanionEngine, create_model, train, predict


def main():
    print("=" * 60)
    print("🧪 Metanion - Simple Regression Example")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 1)
    y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(200)
    
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    
    # Create model
    print("\nCreating Metanion model...")
    model = create_model(
        layer_sizes=[1, 10, 1],
        use_bias=True,
        max_depth=4
    )
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train.tolist(),
        y_train.tolist(),
        X_val.tolist(),
        y_val.tolist(),
        epochs=30
    )
    
    # Evaluate
    print("\nEvaluating...")
    mse = model.evaluate(X_val, y_val)
    print(f"Validation MSE: {mse:.6f}")
    
    # Get best expression
    if model._best_individual:
        print(f"\nBest expression found: {model._best_individual.get_expression()}")
    
    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()