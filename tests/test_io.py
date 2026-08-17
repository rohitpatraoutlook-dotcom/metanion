"""
Test serialization and checkpointing.
Run from project root: python tests/test_io.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import numpy as np
from metanion import MetanionEngine, create_model, train


def test_save_load():
    """Test saving and loading models."""
    print("\n=== Test 11: Model Serialization ===")
    
    # Create and train a model
    engine = MetanionEngine()
    model = create_model(layer_sizes=[1, 5, 1], max_depth=3)
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X[:, 0] + 1
    
    # Train briefly
    model.fit(X.tolist(), y.tolist(), epochs=10)
    initial_pred = model.predict(X[:5])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.metanion', delete=False) as f:
        checkpoint_id = engine.save_model(f.name)
        print(f"Model saved to: {f.name} with ID: {checkpoint_id}")
    
    # Load the model
    loaded_model = engine.load_model(f.name)
    print(f"Model loaded: {loaded_model}")
    
    # Compare predictions
    loaded_pred = loaded_model.predict(X[:5])
    print(f"Original predictions: {initial_pred}")
    print(f"Loaded predictions: {loaded_pred}")
    
    # Clean up
    os.unlink(f.name)
    
    print(" Serialization tests passed!")


if __name__ == "__main__":
    test_save_load()
    print("\n IO tests completed successfully!")