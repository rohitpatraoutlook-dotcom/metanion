"""
Metanion - Simple API for Symbolic Regression.
"""

import numpy as np
import sys
import os
import pickle

# Import from research module
try:
    from research.metanion_research import run_gp, print_expr, test_expression
except ImportError:
    try:
        from metanion_research import run_gp, print_expr, test_expression
    except ImportError:
        print("Warning: metanion_research not found. Using fallback implementation.")
        
        def run_gp(X, y, **kwargs):
            print("Running fallback GP...")
            from metanion.gp.individual import GPIndividual
            from metanion.symbolic import intern, OpID
            from metanion.compile import compile_handle
            import numpy as np
            
            expr = intern(OpID.IDENTITY)
            ind = GPIndividual(weight_handles=[expr], bias_handle=None, shape=(1, 1))
            ind.fitness = 0.0
            ind.depth = 1
            ind.node_count = 1
            return ind
        
        def print_expr(handle, var_names=None):
            return "x0"
        
        def test_expression(handle, X_test, y_true):
            return 0.0, np.array([0.0])

from metanion import compile_handle, intern, lookup, get_pool


class Metanion:
    def __init__(self,
                 pop_size=100,
                 generations=40,
                 max_depth=4,
                 add_bias=True,
                 optimize_constants=True,
                 verbose=False,
                 random_seed=None):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.add_bias = add_bias
        self.optimize_constants = optimize_constants
        self.verbose = verbose
        self.random_seed = random_seed

        self.best_ = None
        self.expression_ = None
        self.fitness_ = None
        self.depth_ = None
        self.nodes_ = None
        self.feature_names_ = None
        self._fitted = False
        self._handle = None

    def fit(self, X, y, feature_names=None):
        X = np.array(X)
        y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.feature_names_ = feature_names
        if self.feature_names_ is None:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]

        if self.verbose:
            print(f"Training Metanion on {X.shape[0]} samples, {X.shape[1]} features...")

        self.best_ = run_gp(
            X, y,
            pop_size=self.pop_size,
            generations=self.generations,
            max_depth=self.max_depth,
            add_bias=self.add_bias,
            optimize_constants=self.optimize_constants,
            verbose=self.verbose,
            random_seed=self.random_seed
        )

        self.fitness_ = self.best_.fitness
        self.depth_ = self.best_.depth
        self.nodes_ = self.best_.node_count
        self._handle = self.best_.weight_handles[0]
        self.expression_ = print_expr(self._handle, self.feature_names_)
        self._fitted = True

        if self.verbose:
            print(f"Training complete. Best fitness: {self.fitness_:.6f}")
            print(f"Expression: {self.expression_}")

        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.array(X)
        n_features = X.shape[1]
        f = compile_handle(self._handle, n_features=n_features)
        return np.array([f(list(x)) for x in X]).flatten()

    def explain(self):
        if not self._fitted:
            return "Model not fitted yet."
        return self.expression_

    def score(self, X, y):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        y_pred = self.predict(X)
        return np.mean((y_pred - y.flatten()) ** 2)

    def summary(self):
        if not self._fitted:
            print("Model not fitted yet.")
            return
        print("=" * 60)
        print("Metanion Model Summary")
        print("=" * 60)
        print(f"Expression:  {self.expression_}")
        print(f"Fitness:     {self.fitness_:.6f}")
        print(f"Depth:       {self.depth_}")
        print(f"Nodes:       {self.nodes_}")
        print(f"Features:    {self.feature_names_}")
        print("=" * 60)

    def save(self, filepath):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        node = lookup(self._handle)
        if node is None:
            raise ValueError("Invalid handle")
        model_data = {
            'handle': self._handle,
            'node': node,
            'expression': self.expression_,
            'fitness': self.fitness_,
            'depth': self.depth_,
            'nodes': self.nodes_,
            'feature_names': self.feature_names_,
            'pop_size': self.pop_size,
            'generations': self.generations,
            'max_depth': self.max_depth,
            'add_bias': self.add_bias,
            'optimize_constants': self.optimize_constants
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        if self.verbose:
            print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self._handle = model_data['handle']
        self.expression_ = model_data['expression']
        self.fitness_ = model_data['fitness']
        self.depth_ = model_data['depth']
        self.nodes_ = model_data['nodes']
        self.feature_names_ = model_data['feature_names']
        self.pop_size = model_data.get('pop_size', 100)
        self.generations = model_data.get('generations', 40)
        self.max_depth = model_data.get('max_depth', 4)
        self.add_bias = model_data.get('add_bias', True)
        self.optimize_constants = model_data.get('optimize_constants', True)
        self._fitted = True
        from metanion.gp.individual import GPIndividual
        self.best_ = GPIndividual(weight_handles=[self._handle], bias_handle=None, shape=(1, 1))
        self.best_.fitness = self.fitness_
        self.best_.depth = self.depth_
        self.best_.node_count = self.nodes_
        if self.verbose:
            print(f"Model loaded from {filepath}")
            print(f"Expression: {self.expression_}")
        return self
