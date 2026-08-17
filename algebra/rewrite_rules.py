"""
Algebraic rewrite rules for the Metanion engine.
Implements term rewriting system (TRS) for expression simplification.
"""

from typing import Optional, Tuple, List, Dict, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from ..symbolic import OpID, get_op_arity, get_op_name, intern, lookup
from ..symbolic import ExpressionNode, get_pool
from ..exceptions import ExpressionError


class RewriteDirection(Enum):
    """Direction of rewrite rule application."""
    LEFT_TO_RIGHT = "->"
    RIGHT_TO_LEFT = "<-"
    BOTH = "<->"


@dataclass
class Pattern:
    """
    A pattern for matching expression trees.
    Supports variables and wildcards.
    """
    
    op: Optional[OpID] = None
    left: Optional['Pattern'] = None
    right: Optional['Pattern'] = None
    is_variable: bool = False
    var_name: Optional[str] = None
    is_wildcard: bool = False
    
    def __post_init__(self):
        """Validate the pattern."""
        if self.is_variable and self.var_name is None:
            self.var_name = f"_var_{id(self)}"
    
    def match(self, handle: int, pool) -> Optional[Dict[str, int]]:
        """
        Match this pattern against an expression.
        
        Args:
            handle: The handle to match against.
            pool: The expression pool.
            
        Returns:
            A mapping of variable names to handles, or None if match fails.
        """
        if self.is_wildcard:
            return {}
        
        if self.is_variable:
            return {self.var_name: handle}
        
        node = pool.get_node(handle)
        if node is None:
            return None
        
        if node.op != self.op:
            return None
        
        # Match children
        bindings = {}
        children = node.get_children()
        
        if self.left is not None:
            if len(children) < 1:
                return None
            left_match = self.left.match(children[0], pool)
            if left_match is None:
                return None
            bindings.update(left_match)
        
        if self.right is not None:
            if len(children) < 2:
                return None
            right_match = self.right.match(children[1], pool)
            if right_match is None:
                return None
            bindings.update(right_match)
        
        return bindings
    
    def substitute(self, bindings: Dict[str, int], pool) -> int:
        """
        Substitute variables in the pattern with values from bindings.
        
        Args:
            bindings: Mapping of variable names to handles.
            pool: The expression pool.
            
        Returns:
            The handle of the substituted expression.
        """
        if self.is_variable:
            if self.var_name not in bindings:
                raise ExpressionError(f"Variable {self.var_name} not bound")
            return bindings[self.var_name]
        
        if self.is_wildcard:
            # Wildcard - return some default
            return intern(OpID.CONST_ZERO)
        
        left_handle = None
        right_handle = None
        
        if self.left is not None:
            left_handle = self.left.substitute(bindings, pool)
        
        if self.right is not None:
            right_handle = self.right.substitute(bindings, pool)
        
        return intern(self.op, left_handle, right_handle)
    
    @classmethod
    def var(cls, name: str) -> 'Pattern':
        """Create a variable pattern."""
        return cls(is_variable=True, var_name=name)
    
    @classmethod
    def wildcard(cls) -> 'Pattern':
        """Create a wildcard pattern."""
        return cls(is_wildcard=True)
    
    @classmethod
    def op(cls, op_id: OpID, left: Optional['Pattern'] = None, right: Optional['Pattern'] = None) -> 'Pattern':
        """Create an operation pattern."""
        return cls(op=op_id, left=left, right=right)


@dataclass
class RewriteRule:
    """
    A rewrite rule for transforming expressions.
    """
    
    name: str
    lhs: Pattern
    rhs: Pattern
    direction: RewriteDirection = RewriteDirection.LEFT_TO_RIGHT
    priority: int = 0
    description: str = ""
    condition: Optional[Callable[[Dict[str, int], int], bool]] = None
    
    def apply(self, handle: int, pool) -> Optional[int]:
        """
        Apply the rewrite rule to an expression.
        
        Args:
            handle: The handle to rewrite.
            pool: The expression pool.
            
        Returns:
            The rewritten handle, or None if the rule doesn't apply.
        """
        # Match the LHS pattern
        bindings = self.lhs.match(handle, pool)
        if bindings is None:
            return None
        
        # Check the condition (if any)
        if self.condition is not None:
            if not self.condition(bindings, handle):
                return None
        
        # Substitute the RHS pattern
        return self.rhs.substitute(bindings, pool)
    
    def __repr__(self) -> str:
        """String representation."""
        arrow = self.direction.value
        return f"{self.name}: {self.lhs} {arrow} {self.rhs}"


class RewriteSystem:
    """
    Term rewriting system for simplifying expressions.
    Applies rewrite rules in order until no more rules apply.
    """
    
    def __init__(self):
        """Initialize the rewrite system."""
        self._rules: List[RewriteRule] = []
        self._rule_index: Dict[OpID, List[RewriteRule]] = {}
        self._stats = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0,
        }
        
        # Initialize with default rules
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize the default rewrite rules."""
        # Identity rules
        self.add_rule(RewriteRule(
            name="add_zero_left",
            lhs=Pattern.op(OpID.ADD, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.var("x"),
            description="x + 0 -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="add_zero_right",
            lhs=Pattern.op(OpID.ADD, Pattern.wildcard(), Pattern.var("x")),
            rhs=Pattern.var("x"),
            description="0 + x -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="mul_one_left",
            lhs=Pattern.op(OpID.MUL, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.var("x"),
            description="x * 1 -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="mul_one_right",
            lhs=Pattern.op(OpID.MUL, Pattern.wildcard(), Pattern.var("x")),
            rhs=Pattern.var("x"),
            description="1 * x -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="sub_zero",
            lhs=Pattern.op(OpID.SUB, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.var("x"),
            description="x - 0 -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="div_one",
            lhs=Pattern.op(OpID.DIV, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.var("x"),
            description="x / 1 -> x"
        ))
        
        # Self operations
        self.add_rule(RewriteRule(
            name="sub_self",
            lhs=Pattern.op(OpID.SUB, Pattern.var("x"), Pattern.var("x")),
            rhs=Pattern.op(OpID.CONST_ZERO),
            description="x - x -> 0"
        ))
        
        self.add_rule(RewriteRule(
            name="div_self",
            lhs=Pattern.op(OpID.DIV, Pattern.var("x"), Pattern.var("x")),
            rhs=Pattern.op(OpID.CONST_ONE),
            description="x / x -> 1"
        ))
        
        # Exponential/Logarithmic
        self.add_rule(RewriteRule(
            name="exp_log",
            lhs=Pattern.op(OpID.EXP, Pattern.op(OpID.LOG, Pattern.var("x"))),
            rhs=Pattern.var("x"),
            description="exp(log(x)) -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="log_exp",
            lhs=Pattern.op(OpID.LOG, Pattern.op(OpID.EXP, Pattern.var("x"))),
            rhs=Pattern.var("x"),
            description="log(exp(x)) -> x"
        ))
        
        # Power rules
        self.add_rule(RewriteRule(
            name="power_zero",
            lhs=Pattern.op(OpID.POWER, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.op(OpID.CONST_ONE),
            description="x^0 -> 1"
        ))
        
        self.add_rule(RewriteRule(
            name="power_one",
            lhs=Pattern.op(OpID.POWER, Pattern.var("x"), Pattern.wildcard()),
            rhs=Pattern.var("x"),
            description="x^1 -> x"
        ))
        
        self.add_rule(RewriteRule(
            name="power_self",
            lhs=Pattern.op(OpID.POWER, Pattern.var("x"), Pattern.op(OpID.CONST_ONE)),
            rhs=Pattern.var("x"),
            description="x^1 -> x"
        ))
        
        # Double negation
        self.add_rule(RewriteRule(
            name="neg_neg",
            lhs=Pattern.op(OpID.NEG, Pattern.op(OpID.NEG, Pattern.var("x"))),
            rhs=Pattern.var("x"),
            description="-(-x) -> x"
        ))
        
        # Inverse rules
        self.add_rule(RewriteRule(
            name="inverse_inverse",
            lhs=Pattern.op(OpID.INVERSE, Pattern.op(OpID.INVERSE, Pattern.var("x"))),
            rhs=Pattern.var("x"),
            description="1/(1/x) -> x"
        ))
        
        # Square/cube simplifications
        self.add_rule(RewriteRule(
            name="square_neg",
            lhs=Pattern.op(OpID.SQUARE, Pattern.op(OpID.NEG, Pattern.var("x"))),
            rhs=Pattern.op(OpID.SQUARE, Pattern.var("x")),
            description="(-x)^2 -> x^2"
        ))
        
        self.add_rule(RewriteRule(
            name="cube_neg",
            lhs=Pattern.op(OpID.CUBE, Pattern.op(OpID.NEG, Pattern.var("x"))),
            rhs=Pattern.op(OpID.NEG, Pattern.op(OpID.CUBE, Pattern.var("x"))),
            description="(-x)^3 -> -x^3"
        ))
        
        # Trigonometric simplifications
        self.add_rule(RewriteRule(
            name="sin_zero",
            lhs=Pattern.op(OpID.SIN, Pattern.op(OpID.CONST_ZERO)),
            rhs=Pattern.op(OpID.CONST_ZERO),
            description="sin(0) -> 0"
        ))
        
        self.add_rule(RewriteRule(
            name="cos_zero",
            lhs=Pattern.op(OpID.COS, Pattern.op(OpID.CONST_ZERO)),
            rhs=Pattern.op(OpID.CONST_ONE),
            description="cos(0) -> 1"
        ))
        
        self.add_rule(RewriteRule(
            name="tan_zero",
            lhs=Pattern.op(OpID.TAN, Pattern.op(OpID.CONST_ZERO)),
            rhs=Pattern.op(OpID.CONST_ZERO),
            description="tan(0) -> 0"
        ))
        
        # Activation function simplifications
        self.add_rule(RewriteRule(
            name="sigmoid_zero",
            lhs=Pattern.op(OpID.SIGMOID, Pattern.op(OpID.CONST_ZERO)),
            rhs=Pattern.op(OpID.CONST_ONE, Pattern.op(OpID.CONST_ZERO)),
            description="sigmoid(0) -> 0.5"
        ))
        
        self.add_rule(RewriteRule(
            name="relu_positive",
            lhs=Pattern.op(OpID.RELU, Pattern.var("x")),
            rhs=Pattern.var("x"),
            condition=lambda b, h: b.get('x', 0) > 0,
            description="relu(x) -> x for x > 0"
        ))
        
        self.add_rule(RewriteRule(
            name="relu_negative",
            lhs=Pattern.op(OpID.RELU, Pattern.var("x")),
            rhs=Pattern.op(OpID.CONST_ZERO),
            condition=lambda b, h: b.get('x', 0) <= 0,
            description="relu(x) -> 0 for x <= 0"
        ))
    
    def add_rule(self, rule: RewriteRule) -> None:
        """
        Add a rewrite rule to the system.
        
        Args:
            rule: The rule to add.
        """
        self._rules.append(rule)
        
        # Index the rule by its LHS operation
        if rule.lhs.op is not None:
            if rule.lhs.op not in self._rule_index:
                self._rule_index[rule.lhs.op] = []
            self._rule_index[rule.lhs.op].append(rule)
        
        # Sort rules by priority
        self._rules.sort(key=lambda r: r.priority, reverse=True)
    
    def apply_rules(self, handle: int, max_iterations: int = 100) -> int:
        """
        Apply rewrite rules to an expression until no more rules apply.
        
        Args:
            handle: The handle to rewrite.
            max_iterations: Maximum number of iterations.
            
        Returns:
            The simplified handle.
        """
        current = handle
        pool = get_pool()
        
        for _ in range(max_iterations):
            applied = False
            
            # Get the operation at the root
            node = pool.get_node(current)
            if node is None:
                break
            
            # Try to apply rules that match the root operation first
            root_op = node.op
            candidate_rules = self._rule_index.get(root_op, [])
            
            # Also consider rules without an LHS op (wildcard patterns)
            for rule in self._rules:
                if rule.lhs.op is None:
                    if rule not in candidate_rules:
                        candidate_rules.append(rule)
            
            # Apply rules in priority order
            for rule in candidate_rules:
                result = rule.apply(current, pool)
                if result is not None and result != current:
                    # Rule applied successfully
                    current = result
                    applied = True
                    self._stats['successful_applications'] += 1
                    break
            
            self._stats['total_applications'] += 1
            
            if not applied:
                break
        
        return current
    
    def normalize(self, handle: int) -> int:
        """
        Normalize an expression to its simplest form.
        
        Args:
            handle: The handle to normalize.
            
        Returns:
            The normalized handle.
        """
        return self.apply_rules(handle)
    
    def get_stats(self) -> Dict[str, int]:
        """Get rewrite system statistics."""
        return {
            'total_rules': len(self._rules),
            'total_applications': self._stats['total_applications'],
            'successful_applications': self._stats['successful_applications'],
            'failed_applications': self._stats['failed_applications'],
            'success_rate': (self._stats['successful_applications'] / 
                           (self._stats['total_applications'] + 1) * 100),
        }
    
    def print_stats(self) -> None:
        """Print rewrite system statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("Rewrite System Statistics")
        print("=" * 50)
        print(f"Total Rules:            {stats['total_rules']}")
        print(f"Total Applications:     {stats['total_applications']}")
        print(f"Successful Applications:{stats['successful_applications']}")
        print(f"Failed Applications:    {stats['failed_applications']}")
        print(f"Success Rate:           {stats['success_rate']:.2f}%")
        print("=" * 50)


# Global rewrite system
_REWRITE_SYSTEM: Optional[RewriteSystem] = None


def get_rewrite_system() -> RewriteSystem:
    """Get or create the global rewrite system."""
    global _REWRITE_SYSTEM
    if _REWRITE_SYSTEM is None:
        _REWRITE_SYSTEM = RewriteSystem()
    return _REWRITE_SYSTEM


def simplify(handle: int) -> int:
    """
    Simplify an expression using the global rewrite system.
    
    Args:
        handle: The handle to simplify.
        
    Returns:
        The simplified handle.
    """
    return get_rewrite_system().normalize(handle)