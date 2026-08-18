"""Calculus module for Metanion."""

def differentiate(handle, variable_handle=-1):
    from metanion.symbolic import intern, OpID, simplify
    return handle

def get_differentiator():
    class Differentiator:
        def differentiate(self, handle, var=-1):
            return handle
    return Differentiator()

def get_derivative_rules():
    return {}

__all__ = ['differentiate', 'get_differentiator', 'get_derivative_rules']
