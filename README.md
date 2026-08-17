# 🧠 Metanion - Zero-Weight Symbolic Tensor Engine

[![PyPI version](https://badge.fury.io/py/metanion.svg)](https://badge.fury.io/py/metanion)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Metanion** is a revolutionary tensor engine where **weights are symbolic expressions, not numbers**. It learns mathematical relationships using Genetic Programming and JIT compilation.

## 🎯 Why Metanion?

- ✅ **No Numerical Weights** - Only operation sequences stored
- ✅ **Explainable** - Outputs human-readable equations
- ✅ **Fast** - JIT compiled to Python bytecode
- ✅ **Differentiable** - Full symbolic differentiation
- ✅ **Lightweight** - Minimal memory footprint

## 🚀 Quick Start

```python
from metanion import create_model, train, predict
import numpy as np

# Create data: y = 2*x + 1 + noise
X = np.random.randn(200, 1)
y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(200)

# Create and train model
model = create_model([1, 10, 1])
train(X, y, epochs=30)

# Make predictions
predictions = predict(X)

# Get the learned equation
print(model._best_individual.get_expression())