"""
Bytecode compiler for Metanion - generates Python functions from expression trees.
"""

import types
from ..symbolic import lookup, get_op_name, OpID


def compile_handle(handle):
    """
    Compile an expression handle to a Python callable function.
    """
    def compile_node(handle):
        node = lookup(handle)
        if node is None:
            return "0.0"
        op = node[0]
        if op == OpID.IDENTITY:
            return "x"
        elif op == OpID.CONST_ZERO:
            return "0.0"
        elif op == OpID.CONST_ONE:
            return "1.0"
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
            return f"({left} / {right})"
        elif op == OpID.POWER:
            left = compile_node(node[1])
            right = compile_node(node[2])
            return f"({left} ** {right})"
        elif op == OpID.SIN:
            arg = compile_node(node[1])
            return f"sin({arg})"
        elif op == OpID.COS:
            arg = compile_node(node[1])
            return f"cos({arg})"
        elif op == OpID.EXP:
            arg = compile_node(node[1])
            return f"exp({arg})"
        elif op == OpID.LOG:
            arg = compile_node(node[1])
            return f"log({arg})"
        elif op == OpID.SQUARE:
            arg = compile_node(node[1])
            return f"({arg} * {arg})"
        elif op == OpID.SQRT:
            arg = compile_node(node[1])
            return f"sqrt({arg})"
        elif op == OpID.NEG:
            arg = compile_node(node[1])
            return f"(-{arg})"
        else:
            # Fallback: treat as constant 0
            return "0.0"
    
    # Build expression string
    expr_str = compile_node(handle)
    
    # Create function
    namespace = {
        'sin': __import__('math').sin,
        'cos': __import__('math').cos,
        'exp': __import__('math').exp,
        'log': __import__('math').log,
        'sqrt': __import__('math').sqrt,
    }
    
    func_code = f"def _compiled(x): return {expr_str}"
    
    # Compile and execute
    exec(func_code, namespace)
    return namespace['_compiled']

