import os
import pickle
import gzip
import hashlib
import numpy as np

class SmartKnowledgeManager:
    def __init__(self, storage_path="knowledge_base/"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.knowledge = {'patterns': {}, 'metadata': {'total_equations': 0, 'best_r2': 0.0}}
        self._load()

    def _load(self):
        f = os.path.join(self.storage_path, "knowledge_base.pkl.gz")
        if os.path.exists(f):
            with gzip.open(f, 'rb') as fp:
                self.knowledge = pickle.load(fp)

    def _save(self):
        with gzip.open(os.path.join(self.storage_path, "knowledge_base.pkl.gz"), 'wb') as f:
            pickle.dump(self.knowledge, f)

    def add_equation(self, equation, r2, depth, nodes, dataset_name):
        if r2 < 0.1 or depth > 8:
            return False
        h = hashlib.md5(equation.encode()).hexdigest()[:16]
        self.knowledge['patterns'][h] = {'equation': equation, 'r2': float(r2), 'depth': int(depth), 'nodes': int(nodes), 'dataset': dataset_name}
        self.knowledge['metadata']['total_equations'] = len(self.knowledge['patterns'])
        if r2 > self.knowledge['metadata']['best_r2']:
            self.knowledge['metadata']['best_r2'] = r2
        self._save()
        return True

    def summary(self):
        return f"Total: {self.knowledge['metadata']['total_equations']} equations, Best R²: {self.knowledge['metadata']['best_r2']:.6f}"

class SmartTrainer:
    def __init__(self, km):
        self.knowledge = km

    def train_on_dataset(self, X, y, name, feature_names=None, n_runs=3, **kwargs):
        from . import Metanion
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        best = None
        best_r2 = -float('inf')
        for run in range(n_runs):
            seed = kwargs.get('random_seed', 42) + run
            model = Metanion(random_seed=seed, verbose=False, **kwargs)
            model.fit(X, y, feature_names=feature_names)
            r2 = model.score(X, y)
            if r2 > best_r2:
                best_r2 = r2
                best = {'model': model, 'r2': r2, 'equation': model.explain(), 'depth': model.depth_, 'nodes': model.nodes_}
        if best:
            self.knowledge.add_equation(best['equation'], best['r2'], best['depth'], best['nodes'], name)
            os.makedirs("models", exist_ok=True)
            best['model'].save(f"models/{name}.metanion")
        return best
