"""Utilities for Metanion."""

class TreePrinter:
    def __init__(self, max_depth=10, max_nodes=50):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
    def print_expression(self, handle, var_name="x"):
        from metanion.symbolic import lookup, get_op_name, OpID
        node = lookup(handle)
        if node is None:
            return "None"
        op = node[0]
        if op == OpID.IDENTITY:
            return var_name
        elif op == OpID.CONST_ZERO:
            return "0"
        elif op == OpID.CONST_ONE:
            return "1"
        return get_op_name(op)

class CostModel:
    def __init__(self):
        pass

class TimeProfiler:
    def __init__(self):
        pass

_COST_MODEL = None
_TIME_PROFILER = None

def get_cost_model():
    global _COST_MODEL
    if _COST_MODEL is None:
        _COST_MODEL = CostModel()
    return _COST_MODEL

def get_time_profiler():
    global _TIME_PROFILER
    if _TIME_PROFILER is None:
        _TIME_PROFILER = TimeProfiler()
    return _TIME_PROFILER

__all__ = ['TreePrinter', 'get_cost_model', 'get_time_profiler']
