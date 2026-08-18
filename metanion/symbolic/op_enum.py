"""Operation enumeration for Metanion."""

from enum import IntEnum


class OpID(IntEnum):
    IDENTITY = 1
    CONST_ZERO = 2
    CONST_ONE = 3
    CONST = 4
    VAR = 5              # NEW: variable with index
    ADD = 6
    SUB = 7
    MUL = 8
    DIV = 9
    POWER = 10
    SIN = 11
    COS = 12
    TANH = 13
    EXP = 14
    LOG = 15
    SQUARE = 16
    CUBE = 17
    SQRT = 18
    RELU = 19
    SIGMOID = 20
    NEG = 21
    INVERSE = 22
    LOG2 = 23
    LOG10 = 24
    ASIN = 25
    ACOS = 26
    ATAN = 27
    SINH = 28
    COSH = 29
    COTH = 30
    ASINH = 31
    ACOSH = 32
    ATANH = 33
    ERF = 34
    GAMMA = 35
    SUM = 36
    MEAN = 37
    MAX = 38
    MIN = 39
    WHERE = 40
    GREATER = 41
    LESS = 42
    EQUAL = 43
    DERIVATIVE = 44
    SECOND_DERIVATIVE = 45
    INTEGRAL = 46
    CLIP = 47
    SIGN = 48
    ROUND = 49
    FLOOR = 50
    CEIL = 51
    STOP_GRADIENT = 52
    IDENTITY_GRAD = 53


class OpCategory:
    IDENTITY = 1
    CONSTANT = 2
    ARITHMETIC = 3
    EXPONENTIAL = 4
    LOGARITHMIC = 5
    TRIGONOMETRIC = 6
    HYPERBOLIC = 7
    ACTIVATION = 8
    SPECIAL = 9
    REDUCTION = 10
    LOGICAL = 11
    CALCULUS = 12


def get_op_name(op: OpID) -> str:
    names = {
        OpID.IDENTITY: "identity",
        OpID.CONST_ZERO: "0",
        OpID.CONST_ONE: "1",
        OpID.CONST: "const",
        OpID.VAR: "var",
        OpID.ADD: "+",
        OpID.SUB: "-",
        OpID.MUL: "*",
        OpID.DIV: "/",
        OpID.POWER: "^",
        OpID.SIN: "sin",
        OpID.COS: "cos",
        OpID.TANH: "tanh",
        OpID.EXP: "exp",
        OpID.LOG: "log",
        OpID.SQUARE: "square",
        OpID.CUBE: "cube",
        OpID.SQRT: "sqrt",
        OpID.RELU: "relu",
        OpID.SIGMOID: "sigmoid",
        OpID.NEG: "neg",
        OpID.INVERSE: "inverse",
        OpID.LOG2: "log2",
        OpID.LOG10: "log10",
        OpID.ASIN: "asin",
        OpID.ACOS: "acos",
        OpID.ATAN: "atan",
        OpID.SINH: "sinh",
        OpID.COSH: "cosh",
        OpID.COTH: "coth",
        OpID.ASINH: "asinh",
        OpID.ACOSH: "acosh",
        OpID.ATANH: "atanh",
        OpID.ERF: "erf",
        OpID.GAMMA: "gamma",
        OpID.SUM: "sum",
        OpID.MEAN: "mean",
        OpID.MAX: "max",
        OpID.MIN: "min",
        OpID.WHERE: "where",
        OpID.GREATER: ">",
        OpID.LESS: "<",
        OpID.EQUAL: "==",
        OpID.DERIVATIVE: "d/dx",
        OpID.SECOND_DERIVATIVE: "d²/dx²",
        OpID.INTEGRAL: "∫",
        OpID.CLIP: "clip",
        OpID.SIGN: "sign",
        OpID.ROUND: "round",
        OpID.FLOOR: "floor",
        OpID.CEIL: "ceil",
        OpID.STOP_GRADIENT: "stop_grad",
        OpID.IDENTITY_GRAD: "identity_grad",
    }
    return names.get(op, f"op_{op.value}")


def get_op_arity(op: OpID) -> int:
    if op in [OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST, OpID.VAR]:
        return 0
    unary_ops = {
        OpID.IDENTITY, OpID.NEG, OpID.INVERSE, OpID.SIN, OpID.COS,
        OpID.TANH, OpID.EXP, OpID.LOG, OpID.SQUARE, OpID.CUBE,
        OpID.SQRT, OpID.RELU, OpID.SIGMOID, OpID.LOG2, OpID.LOG10,
        OpID.ASIN, OpID.ACOS, OpID.ATAN, OpID.SINH, OpID.COSH,
        OpID.COTH, OpID.ASINH, OpID.ACOSH, OpID.ATANH, OpID.ERF,
        OpID.GAMMA, OpID.SUM, OpID.MEAN, OpID.MAX, OpID.MIN,
        OpID.DERIVATIVE, OpID.SECOND_DERIVATIVE, OpID.INTEGRAL,
        OpID.SIGN, OpID.ROUND, OpID.FLOOR, OpID.CEIL,
        OpID.STOP_GRADIENT, OpID.IDENTITY_GRAD
    }
    if op in unary_ops:
        return 1
    if op in {OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV, OpID.POWER,
              OpID.GREATER, OpID.LESS, OpID.EQUAL}:
        return 2
    if op == OpID.WHERE:
        return 3
    return 0


def is_binary_op(op: OpID) -> bool:
    return get_op_arity(op) == 2


def is_unary_op(op: OpID) -> bool:
    return get_op_arity(op) == 1


def is_constant_op(op: OpID) -> bool:
    return op in {OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST}


def is_differentiable_op(op: OpID) -> bool:
    non_diff = {OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST, OpID.VAR,
                OpID.WHERE, OpID.GREATER, OpID.LESS, OpID.EQUAL,
                OpID.SUM, OpID.MEAN, OpID.MAX, OpID.MIN}
    return op not in non_diff
