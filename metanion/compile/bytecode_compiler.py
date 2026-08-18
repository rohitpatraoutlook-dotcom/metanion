"""
Bytecode compiler with COMPOSE support and safe operations.
"""

import math
from ..symbolic import lookup, get_op_name, OpID


def compile_handle(handle, n_features=1):
    def compile_node(handle):
        node = lookup(handle)
        if node is None:
            return "0.0"
        op = node[0]
        
        if op == OpID.IDENTITY:
            return "x[0]"
        elif op == OpID.CONST_ZERO:
            return "0.0"
        elif op == OpID.CONST_ONE:
            return "1.0"
        elif op == OpID.CONST:
            return str(float(node[1]))
        elif op == OpID.VAR:
            idx = node[1]
            return f"x[{idx}]" if idx < n_features else "0.0"
        elif op == OpID.ADD:
            return f"({compile_node(node[1])} + {compile_node(node[2])})"
        elif op == OpID.SUB:
            return f"({compile_node(node[1])} - {compile_node(node[2])})"
        elif op == OpID.MUL:
            return f"({compile_node(node[1])} * {compile_node(node[2])})"
        elif op == OpID.DIV:
            return f"(safe_div({compile_node(node[1])}, {compile_node(node[2])}))"
        elif op == OpID.POWER:
            return f"(safe_pow({compile_node(node[1])}, {compile_node(node[2])}))"
        elif op == OpID.SQRT:
            return f"(safe_sqrt({compile_node(node[1])}))"
        elif op == OpID.SQUARE:
            return f"({compile_node(node[1])} * {compile_node(node[1])})"
        elif op == OpID.CUBE:
            return f"({compile_node(node[1])} * {compile_node(node[1])} * {compile_node(node[1])})"
        elif op == OpID.EXP:
            return f"(safe_exp({compile_node(node[1])}))"
        elif op == OpID.LOG:
            return f"(safe_log({compile_node(node[1])}))"
        elif op == OpID.LOG10:
            return f"(safe_log10({compile_node(node[1])}))"
        elif op == OpID.SIN:
            return f"(safe_sin({compile_node(node[1])}))"
        elif op == OpID.COS:
            return f"(safe_cos({compile_node(node[1])}))"
        elif op == OpID.TAN:
            return f"(safe_tan({compile_node(node[1])}))"
        elif op == OpID.ABS:
            return f"(safe_abs({compile_node(node[1])}))"
        elif op == OpID.INVERSE:
            return f"(safe_inv({compile_node(node[1])}))"
        elif op == OpID.NEG:
            return f"(-{compile_node(node[1])})"
        elif op == OpID.COMPOSE:
            # f(g(x)) = f(g(x))
            return f"({compile_node(node[1])}({compile_node(node[2])}))"
        else:
            return "0.0"

    expr_str = compile_node(handle)

    namespace = {
        'safe_div': lambda a,b: a/b if abs(b) > 1e-12 else 0.0,
        'safe_pow': lambda a,b: a**b if not (a<0 and abs(b-round(b))>1e-12) else 0.0,
        'safe_sin': math.sin,
        'safe_cos': math.cos,
        'safe_tan': math.tan,
        'safe_exp': math.exp,
        'safe_log': lambda x: math.log(x) if x > 0 else 0.0,
        'safe_log10': lambda x: math.log10(x) if x > 0 else 0.0,
        'safe_sqrt': lambda x: math.sqrt(x) if x >= 0 else 0.0,
        'safe_abs': abs,
        'safe_inv': lambda x: 1.0/x if abs(x) > 1e-12 else 0.0,
    }

    func_code = f"def _compiled(x): return {expr_str}"
    exec(func_code, namespace)
    return namespace['_compiled']
