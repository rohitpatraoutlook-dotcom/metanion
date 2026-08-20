import random
from enum import Enum
from ..symbolic import OpID, intern, BINARY_OPS
from .individual import GPIndividual
class InitializationMethod(Enum): RANDOM = "random"
class PopulationInitializer:
    def __init__(self, pop_size, shape, max_depth=5, op_set=None, method=InitializationMethod.RANDOM):
        self.pop_size, self.max_depth = pop_size, max_depth
        self.op_set = op_set or BINARY_OPS.copy()
    def initialize(self):
        pop = []
        for _ in range(self.pop_size):
            pop.append(GPIndividual(weight_handles=[self._random_grow(self.max_depth, min_depth=2)]))
        return pop
    def _random_grow(self, max_depth, min_depth=0, depth=0):
        if depth >= max_depth:
            return intern(random.choice([OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE]))
        if depth < min_depth:
            op = random.choice([OpID.ADD, OpID.MUL])
            return intern(op, self._random_grow(max_depth, min_depth, depth+1), self._random_grow(max_depth, min_depth, depth+1))
        return intern(random.choice([OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE]))
