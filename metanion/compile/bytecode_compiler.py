"""
Bytecode compiler for Metanion - generates Python functions from expression trees.
"""

import types
import math
from ..symbolic import lookup, get_op_name, OpID


def compile_handle(handle, n_features=1):
    """
    Compile an expression to a Python function that accepts a list `x` of length n_features.
    """
    def compile_node(handle):
        node = lookup(handle)
        if node is None:
            return "0.0"
        op = node[0]
        if op == OpID.IDENTITY:
            # For backward compatibility: use first variable if only one feature
            return "x[0]"
        elif op == OpID.CONST_ZERO:
            return "0.0"
        elif op == OpID.CONST_ONE:
            return "1.0"
        elif op == OpID.CONST:
            val = node[1]
            return str(float(val))
        elif op == OpID.VAR:
            idx = node[1]
            if idx < n_features:
                return f"x[{idx}]"
            else:
                # Fallback to 0 if index out of range
                return "0.0"
        elif op == OpID.ADD:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"({left} + {right})"
        elif op == OpID.SUB:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"({left} - {right})"
        elif op == OpID.MUL:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"({left} * {right})"
        elif op == OpID.DIV:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"(safe_div({left}, {right}))"
        elif op == OpID.POWER:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"(safe_pow({left}, {right}))"
        elif op == OpID.SIN:
            arg = compile_node(node[1])
            return f"(safe_sin({arg}))"
        elif op == OpID.COS:
            arg = compile_node(node[1])
            return f"(safe_cos({arg}))"
        elif op == OpID.EXP:
            arg = compile_node(node[1])
            return f"(safe_exp({arg}))"
        elif op == OpID.LOG:
            arg = compile_node(node[1])
            return f"(safe_log({arg}))"
        elif op == OpID.SQUARE:
            arg = compile_node(node[1])
            return f"({arg} * {arg})"
        elif op == OpID.SQRT:
            arg = compile_node(node[1])
            return f"(safe_sqrt({arg}))"
        elif op == OpID.NEG:
            arg = compile_node(node[1])
            return f"(-{arg})"
        else:
            return "0.0"

    expr_str = compile_node(handle)

    # Inject safe functions
    namespace = {
        'safe_div': lambda a,b: a/b if abs(b) > 1e-12 else 0.0,
        'safe_pow': lambda a,b: a**b if not (a<0 and abs(b-round(b))>1e-12) else 0.0,
        'safe_sin': math.sin,
        'safe_cos': math.cos,
        'safe_exp': math.exp,
        'safe_log': lambda x: math.log(x) if x > 0 else 0.0,
        'safe_sqrt': lambda x: math.sqrt(x) if x >= 0 else 0.0,
    }

    func_code = f"def _compiled(x): return {expr_str}"
    exec(func_code, namespace)
    return namespace['_compiled']
