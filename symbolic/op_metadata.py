"""
Detailed metadata for all operations in the Metanion engine.
Includes cost models, type signatures, and derivative rules.
"""

from typing import Dict, Tuple, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

from .op_enum import OpID, OpCategory, get_op_arity, get_op_category


@dataclass
class OpMetadata:
    """
    Complete metadata for a single operation.
    """
    op_id: OpID
    name: str
    category: OpCategory
    arity: int
    
    # Cost model (nanoseconds per operation)
    cost_ns: float = 1.0
    
    # Type signature: list of input types and output type
    # 'N' = numeric (int or float), 'F' = float, 'I' = int, 'B' = boolean
    type_signature: Tuple[str, ...] = ()
    
    # Is the operation differentiable?
    differentiable: bool = True
    
    # Is the operation safe (no domain errors)?
    safe: bool = False
    
    # Does the operation require special handling?
    special: bool = False
    
    # Description of the operation
    description: str = ""
    
    # Whether this operation is a constant generator
    is_constant_generator: bool = False
    
    def __post_init__(self):
        """Auto-fill type signature if not provided."""
        if not self.type_signature:
            if self.arity == 0:
                self.type_signature = ()
            elif self.arity == 1:
                self.type_signature = ('N', 'N')
            elif self.arity == 2:
                self.type_signature = ('N', 'N', 'N')
            elif self.arity == 3:
                self.type_signature = ('N', 'N', 'N', 'N')
            else:
                self.type_signature = ('N',) * (self.arity + 1)


# Complete metadata for all operations
OP_METADATA_DETAILED: Dict[OpID, OpMetadata] = {}

# Identity & Constants
OP_METADATA_DETAILED[OpID.IDENTITY] = OpMetadata(
    OpID.IDENTITY, "identity", OpCategory.IDENTITY, 1,
    cost_ns=0.1, differentiable=True, safe=True,
    description="f(x) = x"
)
OP_METADATA_DETAILED[OpID.CONST_ZERO] = OpMetadata(
    OpID.CONST_ZERO, "zero", OpCategory.CONSTANT, 1,
    cost_ns=0.1, differentiable=False, safe=True,
    is_constant_generator=True, description="f(x) = 0"
)
OP_METADATA_DETAILED[OpID.CONST_ONE] = OpMetadata(
    OpID.CONST_ONE, "one", OpCategory.CONSTANT, 1,
    cost_ns=0.1, differentiable=False, safe=True,
    is_constant_generator=True, description="f(x) = 1"
)

# Arithmetic (Binary)
OP_METADATA_DETAILED[OpID.ADD] = OpMetadata(
    OpID.ADD, "add", OpCategory.ARITHMETIC, 2,
    cost_ns=0.5, differentiable=True, safe=True,
    description="f(a,b) = a + b"
)
OP_METADATA_DETAILED[OpID.SUB] = OpMetadata(
    OpID.SUB, "sub", OpCategory.ARITHMETIC, 2,
    cost_ns=0.5, differentiable=True, safe=True,
    description="f(a,b) = a - b"
)
OP_METADATA_DETAILED[OpID.MUL] = OpMetadata(
    OpID.MUL, "mul", OpCategory.ARITHMETIC, 2,
    cost_ns=0.5, differentiable=True, safe=True,
    description="f(a,b) = a * b"
)
OP_METADATA_DETAILED[OpID.DIV] = OpMetadata(
    OpID.DIV, "div", OpCategory.ARITHMETIC, 2,
    cost_ns=0.7, differentiable=True, safe=False,
    description="f(a,b) = a / b"
)
OP_METADATA_DETAILED[OpID.POWER] = OpMetadata(
    OpID.POWER, "power", OpCategory.ARITHMETIC, 2,
    cost_ns=1.0, differentiable=True, safe=False,
    description="f(a,b) = a ** b"
)

# Arithmetic (Unary)
OP_METADATA_DETAILED[OpID.NEG] = OpMetadata(
    OpID.NEG, "neg", OpCategory.ARITHMETIC, 1,
    cost_ns=0.3, differentiable=True, safe=True,
    description="f(x) = -x"
)
OP_METADATA_DETAILED[OpID.ABS] = OpMetadata(
    OpID.ABS, "abs", OpCategory.ARITHMETIC, 1,
    cost_ns=0.4, differentiable=False, safe=True,
    description="f(x) = |x|"
)
OP_METADATA_DETAILED[OpID.SQRT] = OpMetadata(
    OpID.SQRT, "sqrt", OpCategory.ARITHMETIC, 1,
    cost_ns=2.0, differentiable=True, safe=False,
    description="f(x) = sqrt(x)"
)
OP_METADATA_DETAILED[OpID.CBRT] = OpMetadata(
    OpID.CBRT, "cbrt", OpCategory.ARITHMETIC, 1,
    cost_ns=2.0, differentiable=True, safe=False,
    description="f(x) = cbrt(x)"
)
OP_METADATA_DETAILED[OpID.INVERSE] = OpMetadata(
    OpID.INVERSE, "inverse", OpCategory.ARITHMETIC, 1,
    cost_ns=0.8, differentiable=True, safe=False,
    description="f(x) = 1/x"
)

# Exponential
OP_METADATA_DETAILED[OpID.EXP] = OpMetadata(
    OpID.EXP, "exp", OpCategory.EXPONENTIAL, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = e^x"
)
OP_METADATA_DETAILED[OpID.EXP2] = OpMetadata(
    OpID.EXP2, "exp2", OpCategory.EXPONENTIAL, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = 2^x"
)
OP_METADATA_DETAILED[OpID.EXP10] = OpMetadata(
    OpID.EXP10, "exp10", OpCategory.EXPONENTIAL, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = 10^x"
)
OP_METADATA_DETAILED[OpID.EXPM1] = OpMetadata(
    OpID.EXPM1, "expm1", OpCategory.EXPONENTIAL, 1,
    cost_ns=3.0, differentiable=True, safe=True,
    description="f(x) = e^x - 1"
)

# Logarithmic
OP_METADATA_DETAILED[OpID.LOG] = OpMetadata(
    OpID.LOG, "log", OpCategory.LOGARITHMIC, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = ln(x)"
)
OP_METADATA_DETAILED[OpID.LOG2] = OpMetadata(
    OpID.LOG2, "log2", OpCategory.LOGARITHMIC, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = log2(x)"
)
OP_METADATA_DETAILED[OpID.LOG10] = OpMetadata(
    OpID.LOG10, "log10", OpCategory.LOGARITHMIC, 1,
    cost_ns=3.0, differentiable=True, safe=False,
    description="f(x) = log10(x)"
)
OP_METADATA_DETAILED[OpID.LOG1P] = OpMetadata(
    OpID.LOG1P, "log1p", OpCategory.LOGARITHMIC, 1,
    cost_ns=3.0, differentiable=True, safe=True,
    description="f(x) = ln(1+x)"
)

# Trigonometric
OP_METADATA_DETAILED[OpID.SIN] = OpMetadata(
    OpID.SIN, "sin", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=4.0, differentiable=True, safe=True,
    description="f(x) = sin(x)"
)
OP_METADATA_DETAILED[OpID.COS] = OpMetadata(
    OpID.COS, "cos", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=4.0, differentiable=True, safe=True,
    description="f(x) = cos(x)"
)
OP_METADATA_DETAILED[OpID.TAN] = OpMetadata(
    OpID.TAN, "tan", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = tan(x)"
)
OP_METADATA_DETAILED[OpID.COT] = OpMetadata(
    OpID.COT, "cot", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = cot(x)"
)
OP_METADATA_DETAILED[OpID.SEC] = OpMetadata(
    OpID.SEC, "sec", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = sec(x)"
)
OP_METADATA_DETAILED[OpID.CSC] = OpMetadata(
    OpID.CSC, "csc", OpCategory.TRIGONOMETRIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = csc(x)"
)

# Inverse Trigonometric
OP_METADATA_DETAILED[OpID.ASIN] = OpMetadata(
    OpID.ASIN, "asin", OpCategory.INVERSE_TRIG, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = arcsin(x)"
)
OP_METADATA_DETAILED[OpID.ACOS] = OpMetadata(
    OpID.ACOS, "acos", OpCategory.INVERSE_TRIG, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = arccos(x)"
)
OP_METADATA_DETAILED[OpID.ATAN] = OpMetadata(
    OpID.ATAN, "atan", OpCategory.INVERSE_TRIG, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = arctan(x)"
)
OP_METADATA_DETAILED[OpID.ACOT] = OpMetadata(
    OpID.ACOT, "acot", OpCategory.INVERSE_TRIG, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = arccot(x)"
)

# Hyperbolic
OP_METADATA_DETAILED[OpID.SINH] = OpMetadata(
    OpID.SINH, "sinh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = sinh(x)"
)
OP_METADATA_DETAILED[OpID.COSH] = OpMetadata(
    OpID.COSH, "cosh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = cosh(x)"
)
OP_METADATA_DETAILED[OpID.TANH] = OpMetadata(
    OpID.TANH, "tanh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = tanh(x)"
)
OP_METADATA_DETAILED[OpID.COTH] = OpMetadata(
    OpID.COTH, "coth", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = coth(x)"
)
OP_METADATA_DETAILED[OpID.ASINH] = OpMetadata(
    OpID.ASINH, "asinh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = arsinh(x)"
)
OP_METADATA_DETAILED[OpID.ACOSH] = OpMetadata(
    OpID.ACOSH, "acosh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = arcosh(x)"
)
OP_METADATA_DETAILED[OpID.ATANH] = OpMetadata(
    OpID.ATANH, "atanh", OpCategory.HYPERBOLIC, 1,
    cost_ns=5.0, differentiable=True, safe=False,
    description="f(x) = artanh(x)"
)

# Activation Functions
OP_METADATA_DETAILED[OpID.SIGMOID] = OpMetadata(
    OpID.SIGMOID, "sigmoid", OpCategory.ACTIVATION, 1,
    cost_ns=6.0, differentiable=True, safe=True,
    description="f(x) = 1/(1+e^-x)"
)
OP_METADATA_DETAILED[OpID.RELU] = OpMetadata(
    OpID.RELU, "relu", OpCategory.ACTIVATION, 1,
    cost_ns=1.0, differentiable=False, safe=True,
    description="f(x) = max(0, x)"
)
OP_METADATA_DETAILED[OpID.LEAKY_RELU] = OpMetadata(
    OpID.LEAKY_RELU, "leaky_relu", OpCategory.ACTIVATION, 1,
    cost_ns=1.0, differentiable=False, safe=True,
    description="f(x) = max(0.01x, x)"
)
OP_METADATA_DETAILED[OpID.GELU] = OpMetadata(
    OpID.GELU, "gelu", OpCategory.ACTIVATION, 1,
    cost_ns=10.0, differentiable=True, safe=True,
    description="f(x) = x * Φ(x)"
)
OP_METADATA_DETAILED[OpID.SWISH] = OpMetadata(
    OpID.SWISH, "swish", OpCategory.ACTIVATION, 1,
    cost_ns=8.0, differentiable=True, safe=True,
    description="f(x) = x * sigmoid(x)"
)
OP_METADATA_DETAILED[OpID.ELU] = OpMetadata(
    OpID.ELU, "elu", OpCategory.ACTIVATION, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = x if x>0 else α(e^x-1)"
)
OP_METADATA_DETAILED[OpID.SOFTPLUS] = OpMetadata(
    OpID.SOFTPLUS, "softplus", OpCategory.ACTIVATION, 1,
    cost_ns=5.0, differentiable=True, safe=True,
    description="f(x) = ln(1+e^x)"
)

# Special Functions
OP_METADATA_DETAILED[OpID.ERF] = OpMetadata(
    OpID.ERF, "erf", OpCategory.SPECIAL, 1,
    cost_ns=15.0, differentiable=True, safe=True,
    description="f(x) = erf(x)"
)
OP_METADATA_DETAILED[OpID.GAMMA] = OpMetadata(
    OpID.GAMMA, "gamma", OpCategory.SPECIAL, 1,
    cost_ns=20.0, differentiable=True, safe=False,
    description="f(x) = Γ(x)"
)
OP_METADATA_DETAILED[OpID.SIGMA] = OpMetadata(
    OpID.SIGMA, "sigma", OpCategory.SPECIAL, 1,
    cost_ns=10.0, differentiable=True, safe=True,
    description="f(x) = sigmoid-like function"
)
OP_METADATA_DETAILED[OpID.BETA] = OpMetadata(
    OpID.BETA, "beta", OpCategory.SPECIAL, 2,
    cost_ns=25.0, differentiable=True, safe=False,
    description="f(a,b) = B(a,b)"
)

# Reduction
OP_METADATA_DETAILED[OpID.SUM] = OpMetadata(
    OpID.SUM, "sum", OpCategory.REDUCTION, 1,
    cost_ns=5.0, differentiable=False, safe=True,
    description="f(x) = sum(x)"
)
OP_METADATA_DETAILED[OpID.MEAN] = OpMetadata(
    OpID.MEAN, "mean", OpCategory.REDUCTION, 1,
    cost_ns=5.0, differentiable=False, safe=True,
    description="f(x) = mean(x)"
)
OP_METADATA_DETAILED[OpID.VAR] = OpMetadata(
    OpID.VAR, "var", OpCategory.REDUCTION, 1,
    cost_ns=10.0, differentiable=False, safe=True,
    description="f(x) = variance(x)"
)
OP_METADATA_DETAILED[OpID.STD] = OpMetadata(
    OpID.STD, "std", OpCategory.REDUCTION, 1,
    cost_ns=10.0, differentiable=False, safe=True,
    description="f(x) = std(x)"
)
OP_METADATA_DETAILED[OpID.MAX] = OpMetadata(
    OpID.MAX, "max", OpCategory.REDUCTION, 1,
    cost_ns=5.0, differentiable=False, safe=True,
    description="f(x) = max(x)"
)
OP_METADATA_DETAILED[OpID.MIN] = OpMetadata(
    OpID.MIN, "min", OpCategory.REDUCTION, 1,
    cost_ns=5.0, differentiable=False, safe=True,
    description="f(x) = min(x)"
)

# Logical
OP_METADATA_DETAILED[OpID.WHERE] = OpMetadata(
    OpID.WHERE, "where", OpCategory.LOGICAL, 3,
    cost_ns=2.0, differentiable=False, safe=True,
    type_signature=('B', 'N', 'N', 'N'),
    description="f(cond, a, b) = a if cond else b"
)
OP_METADATA_DETAILED[OpID.GREATER] = OpMetadata(
    OpID.GREATER, "gt", OpCategory.LOGICAL, 2,
    cost_ns=0.5, differentiable=False, safe=True,
    type_signature=('N', 'N', 'B'),
    description="f(a,b) = a > b"
)
OP_METADATA_DETAILED[OpID.LESS] = OpMetadata(
    OpID.LESS, "lt", OpCategory.LOGICAL, 2,
    cost_ns=0.5, differentiable=False, safe=True,
    type_signature=('N', 'N', 'B'),
    description="f(a,b) = a < b"
)
OP_METADATA_DETAILED[OpID.EQUAL] = OpMetadata(
    OpID.EQUAL, "eq", OpCategory.LOGICAL, 2,
    cost_ns=0.5, differentiable=False, safe=True,
    type_signature=('N', 'N', 'B'),
    description="f(a,b) = a == b"
)
OP_METADATA_DETAILED[OpID.AND] = OpMetadata(
    OpID.AND, "and", OpCategory.LOGICAL, 2,
    cost_ns=0.3, differentiable=False, safe=True,
    type_signature=('B', 'B', 'B'),
    description="f(a,b) = a and b"
)
OP_METADATA_DETAILED[OpID.OR] = OpMetadata(
    OpID.OR, "or", OpCategory.LOGICAL, 2,
    cost_ns=0.3, differentiable=False, safe=True,
    type_signature=('B', 'B', 'B'),
    description="f(a,b) = a or b"
)
OP_METADATA_DETAILED[OpID.NOT] = OpMetadata(
    OpID.NOT, "not", OpCategory.LOGICAL, 1,
    cost_ns=0.2, differentiable=False, safe=True,
    type_signature=('B', 'B'),
    description="f(x) = not x"
)

# Calculus
OP_METADATA_DETAILED[OpID.DERIVATIVE] = OpMetadata(
    OpID.DERIVATIVE, "derivative", OpCategory.CALCULUS, 1,
    cost_ns=10.0, differentiable=False, safe=False,
    special=True,
    description="f'(x) = d/dx f(x)"
)
OP_METADATA_DETAILED[OpID.SECOND_DERIVATIVE] = OpMetadata(
    OpID.SECOND_DERIVATIVE, "second_derivative", OpCategory.CALCULUS, 1,
    cost_ns=20.0, differentiable=False, safe=False,
    special=True,
    description="f''(x) = d²/dx² f(x)"
)
OP_METADATA_DETAILED[OpID.INTEGRAL] = OpMetadata(
    OpID.INTEGRAL, "integral", OpCategory.CALCULUS, 1,
    cost_ns=30.0, differentiable=False, safe=False,
    special=True,
    description="∫ f(x) dx"
)

# Statistical
OP_METADATA_DETAILED[OpID.GAUSSIAN] = OpMetadata(
    OpID.GAUSSIAN, "gaussian", OpCategory.STATISTICAL, 1,
    cost_ns=8.0, differentiable=True, safe=True,
    description="f(x) = e^(-x²/2)"
)
OP_METADATA_DETAILED[OpID.LOGISTIC] = OpMetadata(
    OpID.LOGISTIC, "logistic", OpCategory.STATISTICAL, 1,
    cost_ns=6.0, differentiable=True, safe=True,
    description="f(x) = 1/(1+e^-x)"
)

# Custom
OP_METADATA_DETAILED[OpID.CLIP] = OpMetadata(
    OpID.CLIP, "clip", OpCategory.CUSTOM, 3,
    cost_ns=1.5, differentiable=False, safe=True,
    type_signature=('N', 'N', 'N', 'N'),
    description="f(x, min, max) = clip(x, min, max)"
)
OP_METADATA_DETAILED[OpID.SIGN] = OpMetadata(
    OpID.SIGN, "sign", OpCategory.CUSTOM, 1,
    cost_ns=0.5, differentiable=False, safe=True,
    description="f(x) = sign(x)"
)
OP_METADATA_DETAILED[OpID.ROUND] = OpMetadata(
    OpID.ROUND, "round", OpCategory.CUSTOM, 1,
    cost_ns=1.0, differentiable=False, safe=True,
    description="f(x) = round(x)"
)
OP_METADATA_DETAILED[OpID.FLOOR] = OpMetadata(
    OpID.FLOOR, "floor", OpCategory.CUSTOM, 1,
    cost_ns=1.0, differentiable=False, safe=True,
    description="f(x) = floor(x)"
)
OP_METADATA_DETAILED[OpID.CEIL] = OpMetadata(
    OpID.CEIL, "ceil", OpCategory.CUSTOM, 1,
    cost_ns=1.0, differentiable=False, safe=True,
    description="f(x) = ceil(x)"
)

# Composite
OP_METADATA_DETAILED[OpID.SQUARE] = OpMetadata(
    OpID.SQUARE, "square", OpCategory.COMPOSITE, 1,
    cost_ns=0.6, differentiable=True, safe=True,
    description="f(x) = x²"
)
OP_METADATA_DETAILED[OpID.CUBE] = OpMetadata(
    OpID.CUBE, "cube", OpCategory.COMPOSITE, 1,
    cost_ns=0.7, differentiable=True, safe=True,
    description="f(x) = x³"
)
OP_METADATA_DETAILED[OpID.LOGIT] = OpMetadata(
    OpID.LOGIT, "logit", OpCategory.COMPOSITE, 1,
    cost_ns=6.0, differentiable=True, safe=False,
    description="f(x) = ln(x/(1-x))"
)
OP_METADATA_DETAILED[OpID.SOFTMAX] = OpMetadata(
    OpID.SOFTMAX, "softmax", OpCategory.COMPOSITE, 2,
    cost_ns=10.0, differentiable=True, safe=False,
    description="f(x, y) = e^x / (e^x + e^y)"
)

# Gradient
OP_METADATA_DETAILED[OpID.STOP_GRADIENT] = OpMetadata(
    OpID.STOP_GRADIENT, "stop_gradient", OpCategory.CUSTOM, 1,
    cost_ns=0.1, differentiable=False, safe=True,
    special=True,
    description="Stop gradient propagation"
)
OP_METADATA_DETAILED[OpID.IDENTITY_GRAD] = OpMetadata(
    OpID.IDENTITY_GRAD, "identity_grad", OpCategory.IDENTITY, 1,
    cost_ns=0.1, differentiable=True, safe=True,
    description="Identity with gradient (f(x)=x)"
)


def get_op_metadata(op_id: OpID) -> OpMetadata:
    """Get detailed metadata for an operation."""
    if op_id not in OP_METADATA_DETAILED:
        raise ValueError(f"Unknown operation ID: {op_id}")
    return OP_METADATA_DETAILED[op_id]


def get_op_cost_ns(op_id: OpID) -> float:
    """Get the cost (in nanoseconds) for an operation."""
    return get_op_metadata(op_id).cost_ns


def is_op_differentiable(op_id: OpID) -> bool:
    """Check if an operation is differentiable."""
    return get_op_metadata(op_id).differentiable


def is_op_safe(op_id: OpID) -> bool:
    """Check if an operation is safe (no domain errors)."""
    return get_op_metadata(op_id).safe


def get_operations_by_category(category: OpCategory) -> List[OpID]:
    """Get all operations in a category."""
    return [op_id for op_id, meta in OP_METADATA_DETAILED.items()
            if meta.category == category]


def get_all_operation_ids() -> List[OpID]:
    """Get all operation IDs."""
    return list(OP_METADATA_DETAILED.keys())


def get_operation_ids_for_type_signature(sig: Tuple[str, ...]) -> List[OpID]:
    """Get all operations matching a type signature."""
    result = []
    for op_id, meta in OP_METADATA_DETAILED.items():
        if meta.type_signature == sig:
            result.append(op_id)
    return result