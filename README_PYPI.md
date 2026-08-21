# Metanion - Zero-Weight Symbolic Regression Engine

[![PyPI version](https://badge.fury.io/py/metanion.svg)](https://badge.fury.io/py/metanion)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Metanion is a symbolic regression engine that discovers mathematical equations from data using Genetic Programming with a persistent Knowledge Base for continuous learning.

Unlike traditional neural networks, Metanion produces human-readable equations without using any weights or biases, making it fully interpretable and explainable.

---

## Quick Start

```python
from metanion import Metanion
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

# Initialize and train
model = Metanion(pop_size=50, generations=30)
model.fit(X, y, feature_names=["x"])

# Get results
print(model.explain())  # ((1 + 0) + (x + x))
print(model.score(X, y))  # 1.0
print(model.predict(np.array([[6], [7], [8]])))  # [13. 15. 17.]
```

---

Features

· Symbolic regression from data
· Genetic Programming for equation discovery
· Knowledge Base for continuous learning
· Zero-weight architecture
· Human-readable equations
· Save and load trained models

---

Installation

```bash
pip install metanion
```

Requirements:

· Python 3.8 or higher
· NumPy 1.19.0 or higher

---

Examples

Linear Regression

```python
X = np.linspace(0, 5, 20).reshape(-1, 1)
y = 2 * X.flatten() + 1

model = Metanion(pop_size=50, generations=30, verbose=False)
model.fit(X, y, feature_names=["x"])

print(model.explain())  # ((1 + 0) + (x + x))
print(model.score(X, y))  # 1.0
```

Quadratic Regression

```python
X = np.linspace(-5, 5, 20).reshape(-1, 1)
y = X.flatten()**2 + 2*X.flatten() + 1

model = Metanion(pop_size=100, generations=50, verbose=False)
model.fit(X, y, feature_names=["x"])

print(model.explain())  # ((1 + x) * (x + 1))
print(model.score(X, y))  # 1.0
```

Save and Load

```python
# Save model
model.save("my_model.metanion")

# Load model
model2 = Metanion()
model2.load("my_model.metanion")

print(model2.explain())
print(model2.score(X, y))
```

---

API Reference

Metanion Parameters

```python
model = Metanion(
    pop_size=100,        # Number of equations in population
    generations=50,      # Number of generations to evolve
    max_depth=5,         # Maximum depth of equation tree
    verbose=True         # Show training progress
)
```

Methods

Method Description
fit(X, y, feature_names) Train the model
predict(X) Make predictions
score(X, y) Calculate R² score
explain() Get the equation
save(path) Save model
load(path) Load model

---

Knowledge Base

Metanion automatically maintains a Knowledge Base that stores discovered equations. The Knowledge Base enables continuous learning - the model improves with each training run by reusing previously discovered equations.

---

License

MIT License

Source Code

GitHub: https://github.com/rohitpatraoutlook-dotcom/metanion

Citation

```bibtex
@software{metanion2025,
  title={Metanion: Zero-Weight Symbolic Regression Engine},
  author={Patra, Rohit},
  year={2025},
  url={https://github.com/rohitpatraoutlook-dotcom/metanion}
}
```

