import math
from ..symbolic import lookup, OpID

def compile_handle(handle, n_features=1):
    def _compile(h):
        node = lookup(h)
        if node is None: return "0.0"
        op = node[0]
        if op == OpID.IDENTITY: return "x[0]"
        if op == OpID.CONST_ZERO: return "0.0"
        if op == OpID.CONST_ONE: return "1.0"
        if op == OpID.CONST: return str(float(node[1]))
        if op == OpID.VAR:
            idx = node[1]
            return f"x[{idx}]" if idx < n_features else "0.0"
        if op == OpID.ADD: return f"({_compile(node[1])} + {_compile(node[2])})"
        if op == OpID.SUB: return f"({_compile(node[1])} - {_compile(node[2])})"
        if op == OpID.MUL: return f"({_compile(node[1])} * {_compile(node[2])})"
        if op == OpID.DIV: return f"(safe_div({_compile(node[1])}, {_compile(node[2])}))"
        if op == OpID.SQRT: return f"(safe_sqrt({_compile(node[1])}))"
        if op == OpID.EXP: return f"(safe_exp({_compile(node[1])}))"
        if op == OpID.LOG: return f"(safe_log({_compile(node[1])}))"
        if op == OpID.LOG10: return f"(safe_log10({_compile(node[1])}))"
        if op == OpID.SIN: return f"(safe_sin({_compile(node[1])}))"
        if op == OpID.COS: return f"(safe_cos({_compile(node[1])}))"
        return "0.0"
    expr = _compile(handle)
    ns = {
        'safe_div': lambda a,b: a/b if abs(b)>1e-12 else 0.0,
        'safe_sqrt': lambda x: math.sqrt(x) if x>=0 else 0.0,
        'safe_exp': lambda x: math.exp(min(x, 50)),
        'safe_log': lambda x: math.log(x) if x>0 else 0.0,
        'safe_log10': lambda x: math.log10(x) if x>0 else 0.0,
        'safe_sin': math.sin,
        'safe_cos': math.cos,
    }
    exec(f"def _compiled(x): return {expr}", ns)
    return ns['_compiled']
