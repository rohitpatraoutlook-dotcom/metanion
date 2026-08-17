"""
Symbolic differentiator for the Metanion engine.
Performs automatic differentiation on symbolic expressions.
"""

from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
import weakref

from ..symbolic import (
    OpID, intern, lookup, get_pool, get_op_name, get_op_arity,
    is_differentiable_op, is_constant_op, get_depth, count_nodes_in_subtree
)
from ..symbolic import simplify, HandleTraversal
from ..algebra import get_rewrite_system
from ..exceptions import ExpressionError
from .derivative_rules import get_derivative_rules


@dataclass
class DifferentiationContext:
    """
    Context for symbolic differentiation.
    Tracks variables and differentiation state.
    """
    
    variable_handle: int  # The handle representing the variable to differentiate with respect to
    variable_name: str = "x"
    derivatives: Dict[int, int] = field(default_factory=dict)  # handle -> derivative handle
    visited: Set[int] = field(default_factory=set)
    max_depth: Optional[int] = None
    simplify_result: bool = True
    
    def add_derivative(self, handle: int, derivative_handle: int) -> None:
        """Add a derivative mapping for a handle."""
        self.derivatives[handle] = derivative_handle
    
    def get_derivative(self, handle: int) -> Optional[int]:
        """Get the derivative of a handle if it exists."""
        return self.derivatives.get(handle)
    
    def mark_visited(self, handle: int) -> None:
        """Mark a handle as visited during differentiation."""
        self.visited.add(handle)
    
    def is_visited(self, handle: int) -> bool:
        """Check if a handle has been visited."""
        return handle in self.visited


class SymbolicDifferentiator:
    """
    Main symbolic differentiator for the Metanion engine.
    Performs automatic differentiation using term rewriting and chain rule.
    """
    
    def __init__(self, simplify_after: bool = True):
        """
        Initialize the symbolic differentiator.
        
        Args:
            simplify_after: Whether to simplify derivative expressions.
        """
        self._rules = get_derivative_rules()
        self._rewrite_system = get_rewrite_system()
        self._simplify_after = simplify_after
        self._stats = {
            'differentiations': 0,
            'chain_rule_applications': 0,
            'simplifications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        self._cache: Dict[Tuple[int, int], int] = {}  # (handle, var_handle) -> derivative handle
    
    def differentiate(
        self,
        handle: int,
        variable_handle: int = -1,
        simplify_result: Optional[bool] = None,
        max_depth: Optional[int] = None
    ) -> int:
        """
        Differentiate an expression with respect to a variable.
        
        Args:
            handle: The expression to differentiate.
            variable_handle: The handle representing the variable.
                              If -1, uses the identity handle (ID).
            simplify_result: Whether to simplify the result.
            max_depth: Maximum depth for differentiation (prevents infinite recursion).
            
        Returns:
            The derivative expression handle.
            
        Raises:
            ExpressionError: If the expression cannot be differentiated.
        """
        # Get the variable handle if not provided
        if variable_handle == -1:
            # Use the default variable (identity)
            variable_handle = intern(OpID.IDENTITY)
        
        # Check cache
        cache_key = (handle, variable_handle)
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self._stats['cache_misses'] += 1
        self._stats['differentiations'] += 1
        
        # Create differentiation context
        simplify = simplify_result if simplify_result is not None else self._simplify_after
        context = DifferentiationContext(
            variable_handle=variable_handle,
            simplify_result=simplify,
            max_depth=max_depth
        )
        
        # Perform differentiation
        result = self._differentiate(handle, context)
        
        # Simplify the result
        if simplify:
            self._stats['simplifications'] += 1
            result = self._rewrite_system.normalize(result)
        
        # Cache the result
        self._cache[cache_key] = result
        
        return result
    
    def _differentiate(self, handle: int, context: DifferentiationContext) -> int:
        """
        Internal differentiation function with context.
        
        Args:
            handle: The expression to differentiate.
            context: The differentiation context.
            
        Returns:
            The derivative expression handle.
        """
        # Check if we've already differentiated this handle
        cached = context.get_derivative(handle)
        if cached is not None:
            return cached
        
        # Check if we've visited this handle (avoid infinite recursion)
        if context.is_visited(handle):
            raise ExpressionError(f"Circular dependency detected at handle {handle}")
        
        context.mark_visited(handle)
        
        # Get the expression node
        node = lookup(handle)
        if node is None:
            raise ExpressionError(f"Handle {handle} not found")
        
        # Check if this is the variable itself
        if handle == context.variable_handle:
            # d/dx(x) = 1
            result = intern(OpID.CONST_ONE)
            context.add_derivative(handle, result)
            return result
        
        # Check if this is a constant
        if is_constant_op(node.op):
            # d/dx(c) = 0
            result = intern(OpID.CONST_ZERO)
            context.add_derivative(handle, result)
            return result
        
        # Check if this is a composite expression
        if node.arity == 1:
            # Unary operation: chain rule
            result = self._differentiate_unary(handle, node, context)
        elif node.arity == 2:
            # Binary operation: chain rule for each child
            result = self._differentiate_binary(handle, node, context)
        elif node.arity == 0:
            # Nullary operation: constant
            result = intern(OpID.CONST_ZERO)
        else:
            raise ExpressionError(
                f"Unsupported arity {node.arity} for operation {get_op_name(node.op)}"
            )
        
        # Cache the result
        context.add_derivative(handle, result)
        
        return result
    
    def _differentiate_unary(self, handle: int, node: Any, context: DifferentiationContext) -> int:
        """
        Differentiate a unary operation.
        
        Args:
            handle: The expression handle.
            node: The expression node.
            context: The differentiation context.
            
        Returns:
            The derivative expression handle.
        """
        # Get the child handle
        child = node.left
        if child is None:
            raise ExpressionError(f"Unary operation {get_op_name(node.op)} has no child")
        
        # Differentiate the child
        child_derivative = self._differentiate(child, context)
        
        # Apply the derivative rule for this operation
        rule = self._rules.get_rule(node.op)
        if rule is None:
            if is_differentiable_op(node.op):
                raise ExpressionError(
                    f"No derivative rule defined for {get_op_name(node.op)}"
                )
            else:
                raise ExpressionError(
                    f"Operation {get_op_name(node.op)} is not differentiable"
                )
        
        # The rule expects (handle, derivative_handle, None) for unary ops
        result = rule.derivative_fn(handle, child_derivative, None)
        
        self._stats['chain_rule_applications'] += 1
        
        # Simplify immediately if requested
        if context.simplify_result:
            result = self._rewrite_system.normalize(result)
        
        return result
    
    def _differentiate_binary(self, handle: int, node: Any, context: DifferentiationContext) -> int:
        """
        Differentiate a binary operation.
        
        Args:
            handle: The expression handle.
            node: The expression node.
            context: The differentiation context.
            
        Returns:
            The derivative expression handle.
        """
        # Get the child handles
        left = node.left
        right = node.right
        
        if left is None or right is None:
            raise ExpressionError(f"Binary operation {get_op_name(node.op)} missing child")
        
        # Differentiate the children
        left_derivative = self._differentiate(left, context)
        right_derivative = self._differentiate(right, context)
        
        # Apply the derivative rule for this operation
        rule = self._rules.get_rule(node.op)
        if rule is None:
            if is_differentiable_op(node.op):
                raise ExpressionError(
                    f"No derivative rule defined for {get_op_name(node.op)}"
                )
            else:
                raise ExpressionError(
                    f"Operation {get_op_name(node.op)} is not differentiable"
                )
        
        # The rule expects (handle, left_derivative, right_derivative)
        result = rule.derivative_fn(handle, left_derivative, right_derivative)
        
        self._stats['chain_rule_applications'] += 1
        
        # Simplify immediately if requested
        if context.simplify_result:
            result = self._rewrite_system.normalize(result)
        
        return result
    
    def differentiate_n(self, handle: int, order: int, variable_handle: int = -1) -> int:
        """
        Differentiate an expression n times.
        
        Args:
            handle: The expression to differentiate.
            order: The order of differentiation.
            variable_handle: The variable handle.
            
        Returns:
            The nth derivative expression handle.
        """
        if order < 0:
            raise ValueError("Order must be non-negative")
        
        if order == 0:
            return handle
        
        current = handle
        for _ in range(order):
            current = self.differentiate(current, variable_handle)
        
        return current
    
    def get_jacobian(self, expressions: List[int], variables: List[int]) -> List[List[int]]:
        """
        Compute the Jacobian matrix of multiple expressions.
        
        Args:
            expressions: List of expression handles.
            variables: List of variable handles.
            
        Returns:
            Matrix of derivative handles (expressions x variables).
        """
        jacobian = []
        
        for expr in expressions:
            row = []
            for var in variables:
                deriv = self.differentiate(expr, var)
                row.append(deriv)
            jacobian.append(row)
        
        return jacobian
    
    def get_hessian(self, expression: int, variables: List[int]) -> List[List[int]]:
        """
        Compute the Hessian matrix of a single expression.
        
        Args:
            expression: The expression handle.
            variables: List of variable handles.
            
        Returns:
            Hessian matrix of derivative handles.
        """
        # First compute the gradient
        gradient = [self.differentiate(expression, var) for var in variables]
        
        # Then differentiate each gradient component
        hessian = []
        for grad in gradient:
            row = []
            for var in variables:
                deriv = self.differentiate(grad, var)
                row.append(deriv)
            hessian.append(row)
        
        return hessian
    
    def is_zero_derivative(self, handle: int, variable_handle: int = -1) -> bool:
        """
        Check if the derivative of an expression is zero.
        
        Args:
            handle: The expression handle.
            variable_handle: The variable handle.
            
        Returns:
            True if the derivative is zero, False otherwise.
        """
        derivative = self.differentiate(handle, variable_handle)
        node = lookup(derivative)
        if node is None:
            return False
        return node.op == OpID.CONST_ZERO
    
    def is_constant_wrt(self, handle: int, variable_handle: int = -1) -> bool:
        """
        Check if an expression is constant with respect to a variable.
        
        Args:
            handle: The expression handle.
            variable_handle: The variable handle.
            
        Returns:
            True if the expression is constant, False otherwise.
        """
        return self.is_zero_derivative(handle, variable_handle)
    
    def clear_cache(self) -> None:
        """Clear the differentiation cache."""
        self._cache.clear()
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get differentiation statistics."""
        return {
            'differentiations': self._stats['differentiations'],
            'chain_rule_applications': self._stats['chain_rule_applications'],
            'simplifications': self._stats['simplifications'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_size': len(self._cache),
            'hit_ratio': (self._stats['cache_hits'] / 
                         (self._stats['cache_hits'] + self._stats['cache_misses'] + 1) * 100),
        }
    
    def print_stats(self) -> None:
        """Print differentiation statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("Symbolic Differentiator Statistics")
        print("=" * 50)
        print(f"Differentiations:           {stats['differentiations']}")
        print(f"Chain Rule Applications:    {stats['chain_rule_applications']}")
        print(f"Simplifications:            {stats['simplifications']}")
        print(f"Cache Hits:                 {stats['cache_hits']}")
        print(f"Cache Misses:               {stats['cache_misses']}")
        print(f"Cache Size:                 {stats['cache_size']}")
        print(f"Hit Ratio:                  {stats['hit_ratio']:.2f}%")
        print("=" * 50)


# Global differentiator instance
_DIFFERENTIATOR: Optional[SymbolicDifferentiator] = None


def get_differentiator() -> SymbolicDifferentiator:
    """Get or create the global differentiator."""
    global _DIFFERENTIATOR
    if _DIFFERENTIATOR is None:
        _DIFFERENTIATOR = SymbolicDifferentiator()
    return _DIFFERENTIATOR


def differentiate(handle: int, variable_handle: int = -1) -> int:
    """
    Differentiate an expression with respect to a variable.
    
    Args:
        handle: The expression to differentiate.
        variable_handle: The variable handle (default: identity).
        
    Returns:
        The derivative expression handle.
    """
    return get_differentiator().differentiate(handle, variable_handle)


def differentiate_n(handle: int, order: int, variable_handle: int = -1) -> int:
    """
    Differentiate an expression n times.
    
    Args:
        handle: The expression to differentiate.
        order: The order of differentiation.
        variable_handle: The variable handle.
        
    Returns:
        The nth derivative expression handle.
    """
    return get_differentiator().differentiate_n(handle, order, variable_handle)


def is_constant_wrt(handle: int, variable_handle: int = -1) -> bool:
    """
    Check if an expression is constant with respect to a variable.
    
    Args:
        handle: The expression handle.
        variable_handle: The variable handle.
        
    Returns:
        True if the expression is constant, False otherwise.
    """
    return get_differentiator().is_constant_wrt(handle, variable_handle)