import numpy as np
class RobustFitness:
    def __init__(self, X, y, lambda_depth=0.001):
        self.X, self.y, self.lambda_depth = np.array(X), np.array(y).flatten(), lambda_depth
        self._cache = {}
    def evaluate(self, ind):
        if ind.id in self._cache: return self._cache[ind.id]
        try:
            func = ind.compile()
        except:
            ind.fitness = 1e10; self._cache[ind.id] = 1e10; return 1e10
        pred = []
        for row in self.X:
            try:
                val = func(list(row))
                if not np.isfinite(val): val = 1e-6
                pred.append(val)
            except:
                pred.append(1e-6)
        pred = np.array(pred)
        fitness = np.mean((self.y - pred)**2) + self.lambda_depth * ind.depth
        if not np.isfinite(fitness): fitness = 1e10
        ind.fitness = fitness
        self._cache[ind.id] = fitness
        return fitness
