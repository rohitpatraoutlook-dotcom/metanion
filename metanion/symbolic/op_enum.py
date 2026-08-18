"""Operation enumeration for Metanion - Full operations set."""

from enum import IntEnum


class OpID(IntEnum):
    IDENTITY = 1
    CONST_ZERO = 2
    CONST_ONE = 3
    CONST = 4
    VAR = 5
    ADD = 6
    SUB = 7
    MUL = 8
    DIV = 9
    POWER = 10
    SIN = 11
    COS = 12
    TAN = 13
    EXP = 14
    LOG = 15
    LOG10 = 16
    SQRT = 17
    SQUARE = 18
    CUBE = 19
    RELU = 20
    SIGMOID = 21
    TANH = 22
    NEG = 23
    ABS = 24
    INVERSE = 25
    COMPOSE = 26
    
    # Additional trig
    ASIN = 27
    ACOS = 28
    ATAN = 29
    SINH = 30
    COSH = 31
    
    # Reductions
    SUM = 32
    MEAN = 33
    MAX = 34
    MIN = 35
    
    # Logical
    WHERE = 36
    GREATER = 37
    LESS = 38
    EQUAL = 39


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
        OpID.TAN: "tan",
        OpID.EXP: "exp",
        OpID.LOG: "log",
        OpID.LOG10: "log10",
        OpID.SQRT: "sqrt",
        OpID.SQUARE: "square",
        OpID.CUBE: "cube",
        OpID.RELU: "relu",
        OpID.SIGMOID: "sigmoid",
        OpID.TANH: "tanh",
        OpID.NEG: "neg",
        OpID.ABS: "abs",
        OpID.INVERSE: "inv",
        OpID.COMPOSE: "compose",
        OpID.ASIN: "asin",
        OpID.ACOS: "acos",
        OpID.ATAN: "atan",
        OpID.SINH: "sinh",
        OpID.COSH: "cosh",
        OpID.SUM: "sum",
        OpID.MEAN: "mean",
        OpID.MAX: "max",
        OpID.MIN: "min",
        OpID.WHERE: "where",
        OpID.GREATER: ">",
        OpID.LESS: "<",
        OpID.EQUAL: "==",
    }
    return names.get(op, f"op_{op.value}")


def get_op_arity(op: OpID) -> int:
    if op in [OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST, OpID.VAR]:
        return 0
    
    unary_ops = {
        OpID.IDENTITY, OpID.NEG, OpID.ABS, OpID.INVERSE,
        OpID.SIN, OpID.COS, OpID.TAN, OpID.EXP, OpID.LOG, OpID.LOG10,
        OpID.SQRT, OpID.SQUARE, OpID.CUBE, OpID.RELU, OpID.SIGMOID,
        OpID.TANH, OpID.ASIN, OpID.ACOS, OpID.ATAN, OpID.SINH, OpID.COSH,
        OpID.SUM, OpID.MEAN, OpID.MAX, OpID.MIN
    }
    
    if op in unary_ops:
        return 1
    
    if op in {OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV, OpID.POWER,
              OpID.GREATER, OpID.LESS, OpID.EQUAL}:
        return 2
    
    if op in {OpID.WHERE, OpID.COMPOSE}:
        return 2  # COMPOSE takes two expressions
    
    return 0


def is_binary_op(op: OpID) -> bool:
    return get_op_arity(op) == 2


def is_unary_op(op: OpID) -> bool:
    return get_op_arity(op) == 1


def is_constant_op(op: OpID) -> bool:
    return op in {OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST}
