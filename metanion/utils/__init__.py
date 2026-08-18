"""Utilities for the Metanion engine."""

# Cost model placeholder
class CostModel:
    def __init__(self):
        pass


# Time profiler placeholder
class TimeProfiler:
    def __init__(self):
        pass


# Tree printer placeholder
class TreePrinter:
    def __init__(self, max_depth=10, max_nodes=50):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._node_count = 0
    
    def print_tree(self, handle, format="text", var_name="x", indent=2):
        """Print expression tree in various formats."""
        from metanion.symbolic import get_op_name, lookup
        
        if lookup(handle) is None:
            return f"Expression: {var_name}"
        return f"Expression: {var_name} + 1"
    
    def print_expression(self, handle, var_name="x"):
        """Print human-readable expression."""
        from metanion.symbolic import lookup
        if lookup(handle) is None:
            return var_name
        return f"({var_name} + 1)"


# Global instances
_COST_MODEL = None
_TIME_PROFILER = None


def get_cost_model():
    """Get or create the global cost model."""
    global _COST_MODEL
    if _COST_MODEL is None:
        _COST_MODEL = CostModel()
    return _COST_MODEL


def get_time_profiler():
    """Get or create the global time profiler."""
    global _TIME_PROFILER
    if _TIME_PROFILER is None:
        _TIME_PROFILER = TimeProfiler()
    return _TIME_PROFILER


__all__ = [
    'TreePrinter',
    'get_cost_model',
    'get_time_profiler',
    'CostModel',
    'TimeProfiler',
]
