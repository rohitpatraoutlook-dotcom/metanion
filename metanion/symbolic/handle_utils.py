from .op_enum import get_op_arity
def get_depth(h, lookup_func, max_depth=50):
    if max_depth <= 0: return max_depth
    node = lookup_func(h)
    if node is None: return 0
    if isinstance(node, tuple):
        op, arity = node[0], get_op_arity(node[0])
        if arity == 0: return 1
        ld = get_depth(node[1], lookup_func, max_depth-1) if len(node)>1 and node[1] is not None else 0
        rd = get_depth(node[2], lookup_func, max_depth-1) if len(node)>2 and node[2] is not None else 0
        return 1 + max(ld, rd)
    return 1
def count_nodes_in_subtree(h, lookup_func, max_count=100):
    if max_count <= 0: return max_count
    node = lookup_func(h)
    if node is None: return 0
    if isinstance(node, tuple):
        arity = get_op_arity(node[0])
        total = 1
        if arity >= 1 and len(node)>1 and node[1] is not None and total < max_count:
            total += count_nodes_in_subtree(node[1], lookup_func, max_count-total)
        if arity >= 2 and len(node)>2 and node[2] is not None and total < max_count:
            total += count_nodes_in_subtree(node[2], lookup_func, max_count-total)
        return min(total, max_count)
    return 1
