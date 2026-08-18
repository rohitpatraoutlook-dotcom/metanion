import sys
sys.path.insert(0, '/data/data/com.termux/files/home/metanion/research')
from metanion_research import run_gp, print_expr, test_expression
import numpy as np

print("Testing multivariate: y = 2*x0 + 3*x1 - x2 + 5")
np.random.seed(42)
X = np.random.uniform(-5, 5, (200, 3))
y = 2*X[:,0] + 3*X[:,1] - X[:,2] + 5 + 0.1*np.random.randn(200)

best = run_gp(X, y, pop_size=150, generations=60, max_depth=4, add_bias=True, verbose=True, random_seed=42)

print("\nLearned expression:", print_expr(best.weight_handles[0], ["x0","x1","x2"]))
print("Fitness:", best.fitness)
print("Depth:", best.depth, "Nodes:", best.node_count)

test_X = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
true = 2*test_X[:,0] + 3*test_X[:,1] - test_X[:,2] + 5
mse, preds = test_expression(best.weight_handles[0], test_X, true)
print("\nPredictions:", preds)
print("True:", true)
print("MSE:", mse)
