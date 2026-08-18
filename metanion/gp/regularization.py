"""
Symbolic Regularization - Adaptive complexity control for GP.
"""

import numpy as np


class SymbolicRegularization:
    """
    Adaptive regularization based on data linearity.
    """
    
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.linearity_score = self._compute_linearity()
        self.complexity_penalty = self._compute_penalty()
        self.max_depth = self._compute_max_depth()
        self.allowed_ops = self._compute_allowed_ops()
    
    def _compute_linearity(self):
        """Compute linearity score (0-1)."""
        if len(self.X) < 2:
            return 0.5
        
        X_mean = np.mean(self.X, axis=0)
        y_mean = np.mean(self.y)
        
        try:
            cov = np.mean((self.X - X_mean) * (self.y - y_mean).reshape(-1, 1), axis=0)
            var_x = np.var(self.X, axis=0)
            var_y = np.var(self.y)
            
            if var_x.sum() == 0 or var_y == 0:
                return 0.5
            
            corr = np.abs(cov / (np.sqrt(var_x) * np.sqrt(var_y)))
            return float(np.mean(corr))
        except:
            return 0.5
    
    def _compute_penalty(self):
        if self.linearity_score > 0.8:
            return 0.1
        elif self.linearity_score > 0.5:
            return 0.05
        else:
            return 0.01
    
    def _compute_max_depth(self):
        if self.linearity_score > 0.8:
            return 3
        elif self.linearity_score > 0.5:
            return 4
        else:
            return 5
    
    def _compute_allowed_ops(self):
        if self.linearity_score > 0.8:
            return ['add', 'sub', 'mul', 'const', 'var']
        elif self.linearity_score > 0.5:
            return ['add', 'sub', 'mul', 'div', 'const', 'var', 'square', 'sqrt']
        else:
            return ['add', 'sub', 'mul', 'div', 'const', 'var', 'square', 'sqrt', 'sin', 'cos', 'exp', 'log']
    
    def get_params(self):
        return {
            'linearity_score': self.linearity_score,
            'complexity_penalty': self.complexity_penalty,
            'max_depth': self.max_depth,
            'allowed_ops': self.allowed_ops
        }
    
    def explain(self):
        print("\n" + "=" * 60)
        print("SYMBOLIC REGULARIZATION")
        print("=" * 60)
        print(f"Data linearity score: {self.linearity_score:.4f}")
        print(f"Complexity penalty: {self.complexity_penalty:.4f}")
        print(f"Adaptive max depth: {self.max_depth}")
        print(f"Allowed operations: {self.allowed_ops}")
        print("=" * 60)
