"""
Operation enumeration for the Metanion engine.
Defines all primitive operations as symbolic IDs.
"""

from enum import IntEnum, auto
from typing import Dict, Any, Tuple, Optional, List


class OpID(IntEnum):
    """
    Unique identifier for each primitive operation.
    Values are used as handles in the expression pool.
    """
    
    # --- Identity & Constants ---
    IDENTITY = auto()       # f(x) = x
    CONST_ZERO = auto()     # f(x) = 0
    CONST_ONE = auto()      # f(x) = 1
    
    # --- Arithmetic (Binary) ---
    ADD = auto()            # a + b
    SUB = auto()            # a - b
    MUL = auto()            # a * b
    DIV = auto()            # a / b
    POWER = auto()          # a ** b
    
    # --- Arithmetic (Unary) ---
    NEG = auto()            # -x
    ABS = auto()            # |x|
    SQRT = auto()           # sqrt(x)
    CBRT = auto()           # cbrt(x)
    INVERSE = auto()        # 1/x
    
    # --- Exponential ---
    EXP = auto()            # e^x
    EXP2 = auto()           # 2^x
    EXP10 = auto()          # 10^x
    EXPM1 = auto()          # e^x - 1 (stable)
    
    # --- Logarithmic ---
    LOG = auto()            # ln(x)
    LOG2 = auto()           # log2(x)
    LOG10 = auto()          # log10(x)
    LOG1P = auto()          # ln(1+x) (stable)
    
    # --- Trigonometric ---
    SIN = auto()            # sin(x)
    COS = auto()            # cos(x)
    TAN = auto()            # tan(x)
    COT = auto()            # cot(x)
    SEC = auto()            # sec(x)
    CSC = auto()            # csc(x)
    
    # --- Inverse Trigonometric ---
    ASIN = auto()           # arcsin(x)
    ACOS = auto()           # arccos(x)
    ATAN = auto()           # arctan(x)
    ACOT = auto()           # arccot(x)
    
    # --- Hyperbolic ---
    SINH = auto()           # sinh(x)
    COSH = auto()           # cosh(x)
    TANH = auto()           # tanh(x)
    COTH = auto()           # coth(x)
    ASINH = auto()          # arsinh(x)
    ACOSH = auto()          # arcosh(x)
    ATANH = auto()          # artanh(x)
    
    # --- Activation Functions ---
    SIGMOID = auto()        # 1/(1+e^-x)
    RELU = auto()           # max(0, x)
    LEAKY_RELU = auto()     # max(0.01x, x)
    GELU = auto()           # x * Φ(x)
    SWISH = auto()          # x * sigmoid(x)
    ELU = auto()            # x if x>0 else α(e^x-1)
    SOFTPLUS = auto()       # ln(1+e^x)
    
    # --- Special Functions ---
    ERF = auto()            # error function
    GAMMA = auto()          # gamma function
    SIGMA = auto()          # sigmoid-like
    BETA = auto()           # beta function
    
    # --- Reduction (Aggregation) ---
    SUM = auto()            # sum of elements
    MEAN = auto()           # mean of elements
    VAR = auto()            # variance of elements
    STD = auto()            # standard deviation
    MAX = auto()            # maximum of elements
    MIN = auto()            # minimum of elements
    
    # --- Logical ---
    WHERE = auto()          # if cond then a else b
    GREATER = auto()        # a > b
    LESS = auto()           # a < b
    EQUAL = auto()          # a == b
    AND = auto()            # logical AND
    OR = auto()             # logical OR
    NOT = auto()            # logical NOT
    
    # --- Calculus ---
    DERIVATIVE = auto()     # d/dx (symbolic)
    SECOND_DERIVATIVE = auto()  # d²/dx²
    INTEGRAL = auto()       # ∫ (symbolic)
    
    # --- Statistical ---
    GAUSSIAN = auto()       # e^(-x²/2)
    LOGISTIC = auto()       # logistic function
    
    # --- Custom ---
    CLIP = auto()           # clip to [min, max]
    SIGN = auto()           # sign of x
    ROUND = auto()          # round to nearest integer
    FLOOR = auto()          # floor of x
    CEIL = auto()           # ceiling of x
    
    # --- Composite (for convenience) ---
    SQUARE = auto()         # x²
    CUBE = auto()           # x³
    LOGIT = auto()          # ln(x/(1-x))
    SOFTMAX = auto()        # e^x / sum(e^x)
    
    # --- Gradient / Flow Control ---
    STOP_GRADIENT = auto()  # Stop gradient propagation
    IDENTITY_GRAD = auto()  # Identity with gradient


class OpCategory(IntEnum):
    """Category of operation for organization and constraints."""
    IDENTITY = 1
    CONSTANT = 2
    ARITHMETIC = 3
    EXPONENTIAL = 4
    LOGARITHMIC = 5
    TRIGONOMETRIC = 6
    INVERSE_TRIG = 7
    HYPERBOLIC = 8
    ACTIVATION = 9
    SPECIAL = 10
    REDUCTION = 11
    LOGICAL = 12
    CALCULUS = 13
    STATISTICAL = 14
    CUSTOM = 15
    COMPOSITE = 16


# Op metadata: category and arity
OP_METADATA: Dict[OpID, Tuple[OpCategory, int]] = {
    # Identity & Constants
    OpID.IDENTITY: (OpCategory.IDENTITY, 1),
    OpID.CONST_ZERO: (OpCategory.CONSTANT, 1),
    OpID.CONST_ONE: (OpCategory.CONSTANT, 1),
    
    # Arithmetic (Binary)
    OpID.ADD: (OpCategory.ARITHMETIC, 2),
    OpID.SUB: (OpCategory.ARITHMETIC, 2),
    OpID.MUL: (OpCategory.ARITHMETIC, 2),
    OpID.DIV: (OpCategory.ARITHMETIC, 2),
    OpID.POWER: (OpCategory.ARITHMETIC, 2),
    
    # Arithmetic (Unary)
    OpID.NEG: (OpCategory.ARITHMETIC, 1),
    OpID.ABS: (OpCategory.ARITHMETIC, 1),
    OpID.SQRT: (OpCategory.ARITHMETIC, 1),
    OpID.CBRT: (OpCategory.ARITHMETIC, 1),
    OpID.INVERSE: (OpCategory.ARITHMETIC, 1),
    
    # Exponential
    OpID.EXP: (OpCategory.EXPONENTIAL, 1),
    OpID.EXP2: (OpCategory.EXPONENTIAL, 1),
    OpID.EXP10: (OpCategory.EXPONENTIAL, 1),
    OpID.EXPM1: (OpCategory.EXPONENTIAL, 1),
    
    # Logarithmic
    OpID.LOG: (OpCategory.LOGARITHMIC, 1),
    OpID.LOG2: (OpCategory.LOGARITHMIC, 1),
    OpID.LOG10: (OpCategory.LOGARITHMIC, 1),
    OpID.LOG1P: (OpCategory.LOGARITHMIC, 1),
    
    # Trigonometric
    OpID.SIN: (OpCategory.TRIGONOMETRIC, 1),
    OpID.COS: (OpCategory.TRIGONOMETRIC, 1),
    OpID.TAN: (OpCategory.TRIGONOMETRIC, 1),
    OpID.COT: (OpCategory.TRIGONOMETRIC, 1),
    OpID.SEC: (OpCategory.TRIGONOMETRIC, 1),
    OpID.CSC: (OpCategory.TRIGONOMETRIC, 1),
    
    # Inverse Trigonometric
    OpID.ASIN: (OpCategory.INVERSE_TRIG, 1),
    OpID.ACOS: (OpCategory.INVERSE_TRIG, 1),
    OpID.ATAN: (OpCategory.INVERSE_TRIG, 1),
    OpID.ACOT: (OpCategory.INVERSE_TRIG, 1),
    
    # Hyperbolic
    OpID.SINH: (OpCategory.HYPERBOLIC, 1),
    OpID.COSH: (OpCategory.HYPERBOLIC, 1),
    OpID.TANH: (OpCategory.HYPERBOLIC, 1),
    OpID.COTH: (OpCategory.HYPERBOLIC, 1),
    OpID.ASINH: (OpCategory.HYPERBOLIC, 1),
    OpID.ACOSH: (OpCategory.HYPERBOLIC, 1),
    OpID.ATANH: (OpCategory.HYPERBOLIC, 1),
    
    # Activation
    OpID.SIGMOID: (OpCategory.ACTIVATION, 1),
    OpID.RELU: (OpCategory.ACTIVATION, 1),
    OpID.LEAKY_RELU: (OpCategory.ACTIVATION, 1),
    OpID.GELU: (OpCategory.ACTIVATION, 1),
    OpID.SWISH: (OpCategory.ACTIVATION, 1),
    OpID.ELU: (OpCategory.ACTIVATION, 1),
    OpID.SOFTPLUS: (OpCategory.ACTIVATION, 1),
    
    # Special
    OpID.ERF: (OpCategory.SPECIAL, 1),
    OpID.GAMMA: (OpCategory.SPECIAL, 1),
    OpID.SIGMA: (OpCategory.SPECIAL, 1),
    OpID.BETA: (OpCategory.SPECIAL, 2),
    
    # Reduction
    OpID.SUM: (OpCategory.REDUCTION, 1),
    OpID.MEAN: (OpCategory.REDUCTION, 1),
    OpID.VAR: (OpCategory.REDUCTION, 1),
    OpID.STD: (OpCategory.REDUCTION, 1),
    OpID.MAX: (OpCategory.REDUCTION, 1),
    OpID.MIN: (OpCategory.REDUCTION, 1),
    
    # Logical
    OpID.WHERE: (OpCategory.LOGICAL, 3),
    OpID.GREATER: (OpCategory.LOGICAL, 2),
    OpID.LESS: (OpCategory.LOGICAL, 2),
    OpID.EQUAL: (OpCategory.LOGICAL, 2),
    OpID.AND: (OpCategory.LOGICAL, 2),
    OpID.OR: (OpCategory.LOGICAL, 2),
    OpID.NOT: (OpCategory.LOGICAL, 1),
    
    # Calculus
    OpID.DERIVATIVE: (OpCategory.CALCULUS, 1),
    OpID.SECOND_DERIVATIVE: (OpCategory.CALCULUS, 1),
    OpID.INTEGRAL: (OpCategory.CALCULUS, 1),
    
    # Statistical
    OpID.GAUSSIAN: (OpCategory.STATISTICAL, 1),
    OpID.LOGISTIC: (OpCategory.STATISTICAL, 1),
    
    # Custom
    OpID.CLIP: (OpCategory.CUSTOM, 3),
    OpID.SIGN: (OpCategory.CUSTOM, 1),
    OpID.ROUND: (OpCategory.CUSTOM, 1),
    OpID.FLOOR: (OpCategory.CUSTOM, 1),
    OpID.CEIL: (OpCategory.CUSTOM, 1),
    
    # Composite
    OpID.SQUARE: (OpCategory.COMPOSITE, 1),
    OpID.CUBE: (OpCategory.COMPOSITE, 1),
    OpID.LOGIT: (OpCategory.COMPOSITE, 1),
    OpID.SOFTMAX: (OpCategory.COMPOSITE, 2),
    
    # Gradient
    OpID.STOP_GRADIENT: (OpCategory.CUSTOM, 1),
    OpID.IDENTITY_GRAD: (OpCategory.IDENTITY, 1),
}


# Operation names for display
OP_NAMES: Dict[OpID, str] = {
    OpID.IDENTITY: "identity",
    OpID.CONST_ZERO: "0",
    OpID.CONST_ONE: "1",
    OpID.ADD: "+",
    OpID.SUB: "-",
    OpID.MUL: "*",
    OpID.DIV: "/",
    OpID.POWER: "^",
    OpID.NEG: "neg",
    OpID.ABS: "abs",
    OpID.SQRT: "sqrt",
    OpID.CBRT: "cbrt",
    OpID.INVERSE: "inv",
    OpID.EXP: "exp",
    OpID.EXP2: "exp2",
    OpID.EXP10: "exp10",
    OpID.EXPM1: "expm1",
    OpID.LOG: "log",
    OpID.LOG2: "log2",
    OpID.LOG10: "log10",
    OpID.LOG1P: "log1p",
    OpID.SIN: "sin",
    OpID.COS: "cos",
    OpID.TAN: "tan",
    OpID.COT: "cot",
    OpID.SEC: "sec",
    OpID.CSC: "csc",
    OpID.ASIN: "asin",
    OpID.ACOS: "acos",
    OpID.ATAN: "atan",
    OpID.ACOT: "acot",
    OpID.SINH: "sinh",
    OpID.COSH: "cosh",
    OpID.TANH: "tanh",
    OpID.COTH: "coth",
    OpID.ASINH: "asinh",
    OpID.ACOSH: "acosh",
    OpID.ATANH: "atanh",
    OpID.SIGMOID: "sigmoid",
    OpID.RELU: "relu",
    OpID.LEAKY_RELU: "leaky_relu",
    OpID.GELU: "gelu",
    OpID.SWISH: "swish",
    OpID.ELU: "elu",
    OpID.SOFTPLUS: "softplus",
    OpID.ERF: "erf",
    OpID.GAMMA: "gamma",
    OpID.SIGMA: "sigma",
    OpID.BETA: "beta",
    OpID.SUM: "sum",
    OpID.MEAN: "mean",
    OpID.VAR: "var",
    OpID.STD: "std",
    OpID.MAX: "max",
    OpID.MIN: "min",
    OpID.WHERE: "where",
    OpID.GREATER: ">",
    OpID.LESS: "<",
    OpID.EQUAL: "==",
    OpID.AND: "and",
    OpID.OR: "or",
    OpID.NOT: "not",
    OpID.DERIVATIVE: "d/dx",
    OpID.SECOND_DERIVATIVE: "d²/dx²",
    OpID.INTEGRAL: "∫",
    OpID.GAUSSIAN: "gaussian",
    OpID.LOGISTIC: "logistic",
    OpID.CLIP: "clip",
    OpID.SIGN: "sign",
    OpID.ROUND: "round",
    OpID.FLOOR: "floor",
    OpID.CEIL: "ceil",
    OpID.SQUARE: "square",
    OpID.CUBE: "cube",
    OpID.LOGIT: "logit",
    OpID.SOFTMAX: "softmax",
    OpID.STOP_GRADIENT: "stop_grad",
    OpID.IDENTITY_GRAD: "identity_grad",
}


def get_op_arity(op_id: OpID) -> int:
    """Get the arity (number of arguments) for an operation."""
    return OP_METADATA.get(op_id, (OpCategory.CUSTOM, 0))[1]


def get_op_category(op_id: OpID) -> OpCategory:
    """Get the category for an operation."""
    return OP_METADATA.get(op_id, (OpCategory.CUSTOM, 0))[0]


def get_op_name(op_id: OpID) -> str:
    """Get the display name for an operation."""
    return OP_NAMES.get(op_id, f"UNKNOWN_{op_id}")


def is_binary_op(op_id: OpID) -> bool:
    """Check if operation is binary (arity = 2)."""
    return get_op_arity(op_id) == 2


def is_unary_op(op_id: OpID) -> bool:
    """Check if operation is unary (arity = 1)."""
    return get_op_arity(op_id) == 1


def is_ternary_op(op_id: OpID) -> bool:
    """Check if operation is ternary (arity = 3)."""
    return get_op_arity(op_id) == 3


def is_reduction_op(op_id: OpID) -> bool:
    """Check if operation is a reduction (arity = 1 but aggregates)."""
    return get_op_category(op_id) == OpCategory.REDUCTION


def is_constant_op(op_id: OpID) -> bool:
    """Check if operation generates a constant."""
    return get_op_category(op_id) == OpCategory.CONSTANT


def is_arithmetic_op(op_id: OpID) -> bool:
    """Check if operation is arithmetic."""
    return get_op_category(op_id) == OpCategory.ARITHMETIC


def is_trigonometric_op(op_id: OpID) -> bool:
    """Check if operation is trigonometric."""
    return get_op_category(op_id) in (
        OpCategory.TRIGONOMETRIC,
        OpCategory.INVERSE_TRIG,
        OpCategory.HYPERBOLIC,
    )


def is_logical_op(op_id: OpID) -> bool:
    """Check if operation is logical."""
    return get_op_category(op_id) == OpCategory.LOGICAL


def is_calculus_op(op_id: OpID) -> bool:
    """Check if operation is calculus-related."""
    return get_op_category(op_id) == OpCategory.CALCULUS


def is_differentiable_op(op_id: OpID) -> bool:
    """Check if operation is differentiable (for symbolic derivative)."""
    # All operations except logical and constants are differentiable
    if is_logical_op(op_id):
        return False
    if is_constant_op(op_id):
        return False
    if op_id in (OpID.GREATER, OpID.LESS, OpID.EQUAL):
        return False
    if op_id in (OpID.WHERE, OpID.NOT, OpID.AND, OpID.OR):
        return False
    return True