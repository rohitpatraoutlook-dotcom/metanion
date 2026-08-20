import numpy as np
import urllib.request
import tempfile
import os
from . import Metanion
from .knowledge_manager import SmartKnowledgeManager


class AutoMetanion:
    def __init__(self, verbose=True, kb_path="knowledge_base/"):
        self.verbose = verbose
        self.knowledge = SmartKnowledgeManager(kb_path)
        self.model = None
        self.feature_names = []
        self.target_name = None

    def fit(self, data, target_col=None, **kwargs):
        """
        Fit on data.

        Usage:
            auto.fit(X, y)                    # X features, y target
            auto.fit("data.csv")              # auto-detect target (last column)
            auto.fit("data.csv", "target")    # specify column name
            auto.fit(csv_array, target_col=0) # specify column index
        """
        # If target_col is array-like, treat as (X, y)
        if target_col is not None and isinstance(target_col, (np.ndarray, list, tuple)):
            X = np.array(data)
            y = np.array(target_col).flatten()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            self.feature_names = feature_names
            self.target_name = "target"
            X, y = self._preprocess(X, y)
            self.model = Metanion(
                pop_size=kwargs.get('pop_size', 250),
                generations=kwargs.get('generations', 100),
                max_depth=kwargs.get('max_depth', 5),
                verbose=self.verbose,
                random_seed=kwargs.get('random_seed', 42)
            )
            self.model.fit(X, y, feature_names=self.feature_names)
            return self

        # Otherwise, load data from file or combined array
        X, y, feature_names, target_name = self._load_data(data, target_col)
        self.feature_names = feature_names
        self.target_name = target_name

        if self.verbose:
            print(f"📊 Dataset: {len(X)} samples, {len(self.feature_names)} features")
            print(f"   Target: {self.target_name}")

        X, y = self._preprocess(X, y)

        self.model = Metanion(
            pop_size=kwargs.get('pop_size', 250),
            generations=kwargs.get('generations', 100),
            max_depth=kwargs.get('max_depth', 5),
            verbose=self.verbose,
            random_seed=kwargs.get('random_seed', 42)
        )
        self.model.fit(X, y, feature_names=self.feature_names)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def explain(self):
        return self.model.explain()

    def score(self, X=None, y=None):
        return self.model.score(X, y)

    def _load_data(self, data, target_col=None):
        # If data is numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            # If no target specified, use last column
            if target_col is None:
                if data.shape[1] < 2:
                    raise ValueError("Need at least 2 columns: one feature and one target.")
                X = data[:, :-1]
                y = data[:, -1]
                feature_names = [f"x{i}" for i in range(X.shape[1])]
                return X, y, feature_names, "target"
            # If target_col is string
            if isinstance(target_col, str):
                # If it's like 'target', use last column
                if target_col.lower() in ['target', 'y', 'label']:
                    X = data[:, :-1]
                    y = data[:, -1]
                    feature_names = [f"x{i}" for i in range(X.shape[1])]
                    return X, y, feature_names, target_col
                # Try to convert to int
                try:
                    target_col = int(target_col)
                except:
                    raise ValueError(f"Invalid target_col: {target_col}")
            # If target_col is numpy array, convert to int
            if isinstance(target_col, np.ndarray):
                target_col = int(target_col.flat[0]) if target_col.size == 1 else int(target_col[0])
            if isinstance(target_col, list):
                target_col = int(target_col[0])
            # Now target_col should be int
            idx = int(target_col)
            if idx < 0 or idx >= data.shape[1]:
                raise IndexError(f"Column index {idx} out of bounds.")
            cols = [i for i in range(data.shape[1]) if i != idx]
            X = data[:, cols]
            y = data[:, idx]
            feature_names = [f"x{i}" for i in range(X.shape[1])]
            return X, y, feature_names, f"col_{idx}"

        # If data is list
        elif isinstance(data, list):
            return self._load_data(np.array(data), target_col)

        # If data is file path or URL
        elif isinstance(data, str):
            if data.startswith('http'):
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    urllib.request.urlretrieve(data, tmp.name)
                    arr = np.loadtxt(tmp.name, delimiter=',', skiprows=1)
                    os.unlink(tmp.name)
            else:
                arr = np.loadtxt(data, delimiter=',', skiprows=1)
            # Try to get column names from header (for string target_col)
            if isinstance(target_col, str):
                try:
                    with open(data, 'r') as f:
                        first_line = f.readline().strip()
                        headers = first_line.split(',')
                    if len(headers) == arr.shape[1] and target_col in headers:
                        idx = headers.index(target_col)
                        return self._load_data(arr, idx)
                except:
                    pass
            return self._load_data(arr, target_col)

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _preprocess(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y, nan=np.nanmean(y))
        if len(X) > 20:
            z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
            mask = (z < 3).all(axis=1)
            if not mask.all():
                X, y = X[mask], y[mask]
        if X.min() > 0 and X.max() - X.min() > 1000:
            X = np.log10(X + 1e-8)
        if y.min() > 0 and y.max() - y.min() > 1000:
            y = np.log10(y + 1e-8)
        return X, y
