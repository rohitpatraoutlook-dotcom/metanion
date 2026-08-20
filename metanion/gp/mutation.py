import random
from ..symbolic import OpID, intern
def safe_mutate(ind, max_depth=5):
    if random.random() < 0.4 and ind.weight_handles:
        idx = random.randint(0, len(ind.weight_handles)-1)
        ind.weight_handles[idx] = _random_subtree(max_depth)
    return ind
def _random_subtree(max_depth, depth=0):
    if depth >= max_depth:
        return intern(random.choice([OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE]))
    op = random.choice([OpID.ADD, OpID.MUL])
    return intern(op, _random_subtree(max_depth, depth+1), _random_subtree(max_depth, depth+1))
