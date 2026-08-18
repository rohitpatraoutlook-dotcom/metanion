"""Operation enumeration for Metanion."""

from enum import IntEnum


class OpID(IntEnum):
    IDENTITY = 1
    CONST_ZERO = 2
    CONST_ONE = 3
    ADD = 4
    SUB = 5
    MUL = 6
    DIV = 7
    POWER = 8
    SIN = 9
    COS = 10
    TANH = 11
    EXP = 12
    LOG = 13
    SQUARE = 14
    CUBE = 15
    SQRT = 16
    RELU = 17
    SIGMOID = 18
    NEG = 19
    INVERSE = 20
    LOG2 = 21
    LOG10 = 22
    ASIN = 23
    ACOS = 24
    ATAN = 25
    SINH = 26
    COSH = 27
    COTH = 28
    ASINH = 29
    ACOSH = 30
    ATANH = 31
    ERF = 32
    GAMMA = 33
    SUM = 34
    MEAN = 35
    MAX = 36
    MIN = 37
    WHERE = 38
    GREATER = 39
    LESS = 40
    EQUAL = 41
    DERIVATIVE = 42
    SECOND_DERIVATIVE = 43
    INTEGRAL = 44
    CLIP = 45
    SIGN = 46
    ROUND = 47
    FLOOR = 48
    CEIL = 49
    STOP_GRADIENT = 50
    IDENTITY_GRAD = 51


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
    """Get the name of an operation."""
    names = {
        OpID.IDENTITY: "identity",
        OpID.CONST_ZERO: "0",
        OpID.CONST_ONE: "1",
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
    """Get the arity of an operation."""
    unary_ops = {
        OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE, OpID.NEG, OpID.INVERSE,
        OpID.SIN, OpID.COS, OpID.TANH, OpID.EXP, OpID.LOG, OpID.SQUARE, OpID.CUBE,
        OpID.SQRT, OpID.RELU, OpID.SIGMOID, OpID.LOG2, OpID.LOG10, OpID.ASIN,
        OpID.ACOS, OpID.ATAN, OpID.SINH, OpID.COSH, OpID.COTH, OpID.ASINH,
        OpID.ACOSH, OpID.ATANH, OpID.ERF, OpID.GAMMA, OpID.SUM, OpID.MEAN,
        OpID.MAX, OpID.MIN, OpID.DERIVATIVE, OpID.SECOND_DERIVATIVE, OpID.INTEGRAL,
        OpID.SIGN, OpID.ROUND, OpID.FLOOR, OpID.CEIL, OpID.STOP_GRADIENT,
        OpID.IDENTITY_GRAD
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
    """Check if an operation is binary."""
    return get_op_arity(op) == 2


def is_unary_op(op: OpID) -> bool:
    """Check if an operation is unary."""
    return get_op_arity(op) == 1


def is_constant_op(op: OpID) -> bool:
    """Check if an operation is a constant."""
    return op in {OpID.CONST_ZERO, OpID.CONST_ONE}


def is_differentiable_op(op: OpID) -> bool:
    """Check if an operation is differentiable."""
    non_diff = {OpID.CONST_ZERO, OpID.CONST_ONE, OpID.WHERE, OpID.GREATER,
                OpID.LESS, OpID.EQUAL, OpID.SUM, OpID.MEAN, OpID.MAX, OpID.MIN}
    return op not in non_diff
