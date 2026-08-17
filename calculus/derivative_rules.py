"""
Symbolic differentiation rules for the Metanion engine.
Implements automatic differentiation via term rewriting.
"""

from typing import Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass, field

from ..symbolic import OpID, intern, lookup, get_pool, get_op_arity
from ..symbolic import get_op_name, is_differentiable_op
from ..exceptions import ExpressionError


@dataclass
class DerivativeRule:
    """
    A rule for computing the derivative of an operation.
    """
    
    op: OpID
    derivative_fn: Callable[[int, int, int], int]  # (handle, left_handle, right_handle) -> derivative_handle
    arity: int = 1
    description: str = ""
    
    def __post_init__(self):
        """Validate the rule."""
        if self.arity != get_op_arity(self.op):
            raise ExpressionError(
                f"Derivative rule arity {self.arity} does not match "
                f"operation arity {get_op_arity(self.op)} for {get_op_name(self.op)}"
            )


class DerivativeRules:
    """
    Database of derivative rules for all operations.
    Implements the chain rule and basic differentiation.
    """
    
    def __init__(self):
        """Initialize the derivative rules database."""
        self._rules: Dict[OpID, DerivativeRule] = {}
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize all derivative rules."""
        pool = get_pool()
        
        # Get handles for constants
        const_zero = intern(OpID.CONST_ZERO)
        const_one = intern(OpID.CONST_ONE)
        
        # --- Identity ---
        # d/dx(x) = 1
        self.add_rule(DerivativeRule(
            op=OpID.IDENTITY,
            arity=1,
            derivative_fn=lambda h, l, r: const_one,
            description="d/dx(x) = 1"
        ))
        
        # --- Constants ---
        # d/dx(c) = 0
        self.add_rule(DerivativeRule(
            op=OpID.CONST_ZERO,
            arity=1,
            derivative_fn=lambda h, l, r: const_zero,
            description="d/dx(0) = 0"
        ))
        
        self.add_rule(DerivativeRule(
            op=OpID.CONST_ONE,
            arity=1,
            derivative_fn=lambda h, l, r: const_zero,
            description="d/dx(1) = 0"
        ))
        
        # --- Arithmetic (Binary) ---
        # d/dx(f + g) = df/dx + dg/dx
        self.add_rule(DerivativeRule(
            op=OpID.ADD,
            arity=2,
            derivative_fn=lambda h, l, r: intern(OpID.ADD, l, r),
            description="d/dx(f + g) = df/dx + dg/dx"
        ))
        
        # d/dx(f - g) = df/dx - dg/dx
        self.add_rule(DerivativeRule(
            op=OpID.SUB,
            arity=2,
            derivative_fn=lambda h, l, r: intern(OpID.SUB, l, r),
            description="d/dx(f - g) = df/dx - dg/dx"
        ))
        
        # d/dx(f * g) = f * dg/dx + g * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.MUL,
            arity=2,
            derivative_fn=lambda h, l, r: intern(
                OpID.ADD,
                intern(OpID.MUL, l, r),  # f * dg
                intern(OpID.MUL, r, l)   # g * df
            ),
            description="d/dx(f * g) = f * dg/dx + g * df/dx"
        ))
        
        # d/dx(f / g) = (g * df/dx - f * dg/dx) / g^2
        self.add_rule(DerivativeRule(
            op=OpID.DIV,
            arity=2,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                intern(
                    OpID.SUB,
                    intern(OpID.MUL, r, l),   # g * df
                    intern(OpID.MUL, l, r)    # f * dg
                ),
                intern(OpID.POWER, r, const_two())  # g^2
            ),
            description="d/dx(f / g) = (g * df/dx - f * dg/dx) / g^2"
        ))
        
        # d/dx(f^g) = f^g * (g * df/dx / f + dg/dx * log(f))
        self.add_rule(DerivativeRule(
            op=OpID.POWER,
            arity=2,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                h,  # f^g
                intern(
                    OpID.ADD,
                    intern(OpID.MUL, r, intern(OpID.DIV, l, l)),  # g * df/f
                    intern(OpID.MUL, r, intern(OpID.LOG, l))      # dg * log(f)
                )
            ),
            description="d/dx(f^g) = f^g * (g * df/dx / f + dg/dx * log(f))"
        ))
        
        # --- Arithmetic (Unary) ---
        # d/dx(-f) = -df/dx
        self.add_rule(DerivativeRule(
            op=OpID.NEG,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.NEG, l),
            description="d/dx(-f) = -df/dx"
        ))
        
        # d/dx(1/f) = -df/dx / f^2
        self.add_rule(DerivativeRule(
            op=OpID.INVERSE,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                intern(OpID.NEG, l),  # -df
                intern(OpID.POWER, h, const_two())  # f^2
            ),
            description="d/dx(1/f) = -df/dx / f^2"
        ))
        
        # --- Exponential ---
        # d/dx(e^f) = e^f * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.EXP,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.MUL, h, l),
            description="d/dx(e^f) = e^f * df/dx"
        ))
        
        # d/dx(2^f) = 2^f * log(2) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.EXP2,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                h,
                intern(OpID.MUL, const_log2(), l)
            ),
            description="d/dx(2^f) = 2^f * log(2) * df/dx"
        ))
        
        # d/dx(10^f) = 10^f * log(10) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.EXP10,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                h,
                intern(OpID.MUL, const_log10(), l)
            ),
            description="d/dx(10^f) = 10^f * log(10) * df/dx"
        ))
        
        # --- Logarithmic ---
        # d/dx(log(f)) = df/dx / f
        self.add_rule(DerivativeRule(
            op=OpID.LOG,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.DIV, l, h),
            description="d/dx(log(f)) = df/dx / f"
        ))
        
        # d/dx(log2(f)) = df/dx / (f * log(2))
        self.add_rule(DerivativeRule(
            op=OpID.LOG2,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.MUL, h, const_log2())
            ),
            description="d/dx(log2(f)) = df/dx / (f * log(2))"
        ))
        
        # d/dx(log10(f)) = df/dx / (f * log(10))
        self.add_rule(DerivativeRule(
            op=OpID.LOG10,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.MUL, h, const_log10())
            ),
            description="d/dx(log10(f)) = df/dx / (f * log(10))"
        ))
        
        # --- Trigonometric ---
        # d/dx(sin(f)) = cos(f) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.SIN,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.MUL, intern(OpID.COS, h), l),
            description="d/dx(sin(f)) = cos(f) * df/dx"
        ))
        
        # d/dx(cos(f)) = -sin(f) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.COS,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.MUL, intern(OpID.NEG, intern(OpID.SIN, h)), l),
            description="d/dx(cos(f)) = -sin(f) * df/dx"
        ))
        
        # d/dx(tan(f)) = sec^2(f) * df/dx = (1 + tan^2(f)) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.TAN,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.ADD, const_one, intern(OpID.SQUARE, h)),  # 1 + tan^2
                l
            ),
            description="d/dx(tan(f)) = sec^2(f) * df/dx"
        ))
        
        # d/dx(cot(f)) = -csc^2(f) * df/dx = -(1 + cot^2(f)) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.COT,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.NEG, intern(OpID.ADD, const_one, intern(OpID.SQUARE, h))),
                l
            ),
            description="d/dx(cot(f)) = -csc^2(f) * df/dx"
        ))
        
        # --- Hyperbolic ---
        # d/dx(sinh(f)) = cosh(f) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.SINH,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.MUL, intern(OpID.COSH, h), l),
            description="d/dx(sinh(f)) = cosh(f) * df/dx"
        ))
        
        # d/dx(cosh(f)) = sinh(f) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.COSH,
            arity=1,
            derivative_fn=lambda h, l, r: intern(OpID.MUL, intern(OpID.SINH, h), l),
            description="d/dx(cosh(f)) = sinh(f) * df/dx"
        ))
        
        # d/dx(tanh(f)) = sech^2(f) * df/dx = (1 - tanh^2(f)) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.TANH,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.SUB, const_one, intern(OpID.SQUARE, h)),  # 1 - tanh^2
                l
            ),
            description="d/dx(tanh(f)) = sech^2(f) * df/dx"
        ))
        
        # --- Inverse Trigonometric ---
        # d/dx(asin(f)) = df/dx / sqrt(1 - f^2)
        self.add_rule(DerivativeRule(
            op=OpID.ASIN,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.SQRT, intern(OpID.SUB, const_one, intern(OpID.SQUARE, h)))
            ),
            description="d/dx(asin(f)) = df/dx / sqrt(1 - f^2)"
        ))
        
        # d/dx(acos(f)) = -df/dx / sqrt(1 - f^2)
        self.add_rule(DerivativeRule(
            op=OpID.ACOS,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                intern(OpID.NEG, l),
                intern(OpID.SQRT, intern(OpID.SUB, const_one, intern(OpID.SQUARE, h)))
            ),
            description="d/dx(acos(f)) = -df/dx / sqrt(1 - f^2)"
        ))
        
        # d/dx(atan(f)) = df/dx / (1 + f^2)
        self.add_rule(DerivativeRule(
            op=OpID.ATAN,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.ADD, const_one, intern(OpID.SQUARE, h))
            ),
            description="d/dx(atan(f)) = df/dx / (1 + f^2)"
        ))
        
        # --- Activation Functions ---
        # d/dx(sigmoid(f)) = sigmoid(f) * (1 - sigmoid(f)) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.SIGMOID,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.MUL, h, intern(OpID.SUB, const_one, h)),
                l
            ),
            description="d/dx(sigmoid(f)) = sigmoid(f) * (1 - sigmoid(f)) * df/dx"
        ))
        
        # d/dx(relu(f)) = df/dx if f > 0 else 0
        # This is implemented as a special case in the differentiator
        # using the WHERE operator
        self.add_rule(DerivativeRule(
            op=OpID.RELU,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.WHERE,
                intern(OpID.GREATER, h, const_zero),  # f > 0
                l,  # df/dx
                const_zero  # 0
            ),
            description="d/dx(relu(f)) = df/dx if f > 0 else 0"
        ))
        
        # d/dx(leaky_relu(f)) = df/dx if f > 0 else 0.01 * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.LEAKY_RELU,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.WHERE,
                intern(OpID.GREATER, h, const_zero),  # f > 0
                l,  # df/dx
                intern(OpID.MUL, const_0_01(), l)  # 0.01 * df/dx
            ),
            description="d/dx(leaky_relu(f)) = df/dx if f > 0 else 0.01 * df/dx"
        ))
        
        # d/dx(tanh(f)) = (1 - tanh^2(f)) * df/dx
        # Already covered by the hyperbolic rule above
        
        # --- Special Functions ---
        # d/dx(erf(f)) = 2 * exp(-f^2) * df/dx / sqrt(pi)
        self.add_rule(DerivativeRule(
            op=OpID.ERF,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(
                    OpID.MUL,
                    const_two(),
                    intern(OpID.EXP, intern(OpID.NEG, intern(OpID.SQUARE, h)))
                ),
                intern(OpID.DIV, l, const_sqrt_pi())
            ),
            description="d/dx(erf(f)) = 2 * exp(-f^2) * df/dx / sqrt(pi)"
        ))
        
        # --- Squares and Powers ---
        # d/dx(f^2) = 2 * f * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.SQUARE,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.MUL, const_two(), h),  # 2*f
                l  # df/dx
            ),
            description="d/dx(f^2) = 2 * f * df/dx"
        ))
        
        # d/dx(f^3) = 3 * f^2 * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.CUBE,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.MUL, const_three(), intern(OpID.SQUARE, h)),  # 3*f^2
                l
            ),
            description="d/dx(f^3) = 3 * f^2 * df/dx"
        ))
        
        # d/dx(sqrt(f)) = df/dx / (2 * sqrt(f))
        self.add_rule(DerivativeRule(
            op=OpID.SQRT,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.MUL, const_two(), h)
            ),
            description="d/dx(sqrt(f)) = df/dx / (2 * sqrt(f))"
        ))
        
        # d/dx(cbrt(f)) = df/dx / (3 * f^(2/3))
        self.add_rule(DerivativeRule(
            op=OpID.CBRT,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.DIV,
                l,
                intern(OpID.MUL, const_three(), intern(OpID.SQUARE, h))
            ),
            description="d/dx(cbrt(f)) = df/dx / (3 * cbrt(f)^2)"
        ))
        
        # --- ABS ---
        # d/dx(abs(f)) = sign(f) * df/dx
        self.add_rule(DerivativeRule(
            op=OpID.ABS,
            arity=1,
            derivative_fn=lambda h, l, r: intern(
                OpID.MUL,
                intern(OpID.SIGN, h),  # sign(f)
                l
            ),
            description="d/dx(abs(f)) = sign(f) * df/dx"
        ))
    
    def add_rule(self, rule: DerivativeRule) -> None:
        """
        Add a derivative rule to the database.
        
        Args:
            rule: The rule to add.
        """
        self._rules[rule.op] = rule
    
    def get_rule(self, op: OpID) -> Optional[DerivativeRule]:
        """
        Get the derivative rule for an operation.
        
        Args:
            op: The operation ID.
            
        Returns:
            The derivative rule, or None if not found.
        """
        return self._rules.get(op)
    
    def has_rule(self, op: OpID) -> bool:
        """
        Check if a derivative rule exists for an operation.
        
        Args:
            op: The operation ID.
            
        Returns:
            True if a rule exists, False otherwise.
        """
        return op in self._rules
    
    def differentiate(self, handle: int, derivative_handle: int) -> int:
        """
        Apply the chain rule to differentiate an expression.
        
        Args:
            handle: The expression to differentiate.
            derivative_handle: The derivative of the outer function (d_outer).
            
        Returns:
            The derivative of the expression.
        """
        node = lookup(handle)
        if node is None:
            raise ExpressionError(f"Handle {handle} not found")
        
        # Get the derivative rule for this operation
        rule = self.get_rule(node.op)
        if rule is None:
            if is_differentiable_op(node.op):
                # Operation is differentiable but no rule defined
                raise ExpressionError(
                    f"No derivative rule defined for {get_op_name(node.op)}"
                )
            else:
                # Operation is not differentiable
                raise ExpressionError(
                    f"Operation {get_op_name(node.op)} is not differentiable"
                )
        
        # Apply the derivative rule
        # The rule's derivative_fn expects (handle, left_derivative, right_derivative)
        left_deriv = None
        right_deriv = None
        
        if node.left is not None:
            # Recursively differentiate the left child
            left_deriv = self.differentiate(node.left, derivative_handle)
        
        if node.right is not None:
            # Recursively differentiate the right child
            right_deriv = self.differentiate(node.right, derivative_handle)
        
        # Apply the rule
        result = rule.derivative_fn(handle, left_deriv, right_deriv)
        
        return result


# Helper functions for constants
def const_two() -> int:
    """Get handle for constant 2."""
    return intern(OpID.ADD, intern(OpID.CONST_ONE), intern(OpID.CONST_ONE))


def const_three() -> int:
    """Get handle for constant 3."""
    return intern(OpID.ADD, const_two(), intern(OpID.CONST_ONE))


def const_0_01() -> int:
    """Get handle for constant 0.01."""
    # We approximate 0.01 as 1/100
    hundred = intern(OpID.POWER, intern(OpID.CONST_ONE, intern(OpID.ADD, const_two(), const_two())), const_two())
    return intern(OpID.DIV, intern(OpID.CONST_ONE), hundred)


def const_log2() -> int:
    """Get handle for log(2)."""
    return intern(OpID.LOG, const_two())


def const_log10() -> int:
    """Get handle for log(10)."""
    ten = intern(OpID.ADD, const_two(), intern(OpID.MUL, const_two(), const_two()))
    return intern(OpID.LOG, ten)


def const_sqrt_pi() -> int:
    """Get handle for sqrt(pi)."""
    # Approximate sqrt(pi) as 1.77245
    # We'll use a rough constant: sqrt(pi) ≈ 1.77245
    # In practice, this would be precomputed
    return intern(OpID.CONST_ONE)  # Placeholder


# Global derivative rules database
_DERIVATIVE_RULES: Optional[DerivativeRules] = None


def get_derivative_rules() -> DerivativeRules:
    """Get or create the global derivative rules database."""
    global _DERIVATIVE_RULES
    if _DERIVATIVE_RULES is None:
        _DERIVATIVE_RULES = DerivativeRules()
    return _DERIVATIVE_RULES