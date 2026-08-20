import numpy as np
import time
import pickle
import random
from ..symbolic import OpID, intern, lookup
from ..compile import compile_handle
from ..gp import GPIndividual, PopulationManager, RobustFitness

class Metanion:
    def __init__(self, pop_size=200, generations=80, max_depth=5, verbose=True, random_seed=42):
        self.pop_size, self.generations, self.max_depth = pop_size, generations, max_depth
        self.verbose, self.random_seed = verbose, random_seed
        random.seed(random_seed); np.random.seed(random_seed)
        self._fitted = False
        self._handle = None
        self.feature_names = None
        self.training_time = 0.0

    def fit(self, X, y, feature_names=None):
        X, y = np.array(X), np.array(y).flatten()
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        evaluator = RobustFitness(X, y, lambda_depth=0.001)
        from ..gp.initialization import PopulationInitializer, InitializationMethod
        initializer = PopulationInitializer(pop_size=self.pop_size, shape=(X.shape[1], 1), max_depth=self.max_depth)
        population = initializer.initialize()

        manager = PopulationManager(population, self.pop_size, evaluator, elitism_count=5, crossover_rate=0.8, mutation_rate=0.3, tournament_size=5, max_depth=self.max_depth)

        print(f"🧠 Training on {X.shape[0]} samples...")
        start = time.time()
        for gen in range(self.generations):
            manager._evolve_one_generation()
            if self.verbose and (gen+1) % 10 == 0:
                stats = manager.get_stats()
                print(f"Gen {gen+1}/{self.generations} | best: {stats['best_fitness']:.6f}")
        self.training_time = time.time() - start

        best = manager.get_best()
        self._handle = best.weight_handles[0]
        self.fitness_, self.depth_, self.nodes_ = best.fitness, best.depth, best.node_count
        self._fitted = True
        if self.verbose:
            print(f"✅ Done in {self.training_time:.2f}s")
            print(f"📐 {self.explain()}")
        return self

    def predict(self, X):
        if not self._fitted: raise RuntimeError("Not fitted")
        X = np.array(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        func = compile_handle(self._handle, n_features=X.shape[1])
        return np.array([func(list(row)) for row in X]).flatten()

    def explain(self):
        if not self._fitted: return "Not fitted"
        def _print(h):
            n = lookup(h)
            if n is None: return "None"
            op = n[0]
            if op == OpID.IDENTITY: return self.feature_names[0]
            if op in (OpID.CONST_ZERO, OpID.CONST_ONE): return str(0 if op == OpID.CONST_ZERO else 1)
            if op == OpID.CONST: return f"{n[1]:.4f}"
            left = _print(n[1]) if n[1] is not None else ""
            right = _print(n[2]) if n[2] is not None else ""
            from ..symbolic import get_op_name
            return f"({left} {get_op_name(op)} {right})"
        return _print(self._handle)

    def score(self, X, y):
        yp = self.predict(X)
        yt = np.array(y).flatten()
        return 1 - np.mean((yp - yt)**2) / (np.var(yt) + 1e-10)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'handle': self._handle, 'feature_names': self.feature_names, 'fitness': self.fitness_, 'depth': self.depth_, 'nodes': self.nodes_}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self._handle, self.feature_names, self.fitness_, self.depth_, self.nodes_ = d['handle'], d.get('feature_names', ['x']), d.get('fitness', float('inf')), d.get('depth', 0), d.get('nodes', 0)
        self._fitted = True
        return self
