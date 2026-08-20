import random
from dataclasses import dataclass, field
from ..symbolic import lookup, get_depth, count_nodes_in_subtree, get_pool
from ..compile import compile_handle
@dataclass
class GPIndividual:
    weight_handles: list = field(default_factory=list)
    fitness: float = float('inf')
    depth: int = 0
    node_count: int = 0
    id: int = field(default_factory=lambda: random.randint(0, 10**9))
    def __post_init__(self):
        pool = get_pool()
        max_d = 0; total = 0
        for h in self.weight_handles:
            if h is not None:
                max_d = max(max_d, get_depth(h, pool.get_node))
                total += count_nodes_in_subtree(h, pool.get_node)
        self.depth, self.node_count = max_d, total
    def compile(self):
        return compile_handle(self.weight_handles[0]) if self.weight_handles else lambda x: 0.0
    def copy(self):
        return GPIndividual(weight_handles=self.weight_handles.copy(), id=random.randint(0, 10**9))
