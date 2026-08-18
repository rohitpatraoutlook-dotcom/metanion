import sys
sys.path.insert(0, '/data/data/com.termux/files/home/metanion/research')
from metanion_research import *

import numpy as np

# Polynomial data
np.random.seed(42)
X = np.random.uniform(-2, 2, (200, 1))
y = X**3 - 2*X**2 + X + 1 + 0.1 * np.random.randn(200, 1)

print("TEST: Polynomial with bias\n")
best = run_gp(X, y, pop_size=150, generations=80, max_depth=4,
              safe=True, add_bias=True, verbose=True, random_seed=42)

print("\n" + "="*80)
print("FINAL EXPRESSION")
print("="*80)
print(print_expr(best.weight_handles[0]))
