"""
Cost modeling for the Metanion engine.
Estimates and tracks computational costs of expressions.
"""

from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
import math

from ..symbolic import OpID, get_op_metadata, get_op_name, lookup
from ..symbolic import get_depth, count_nodes_in_subtree
from ..exceptions import ExpressionError


@dataclass
class CostProfile:
    """Cost profile for an expression or model."""
    
    total_time_ns: float = 0.0
    total_ops: int = 0
    max_depth: int = 0
    node_count: int = 0
    primitive_ops: Dict[OpID, int] = field(default_factory=dict)
    primitive_costs: Dict[OpID, float] = field(default_factory=dict)
    
    def add_primitive(self, op: OpID, cost_ns: float, count: int = 1) -> None:
        """Add cost for a primitive operation."""
        self.primitive_ops[op] = self.primitive_ops.get(op, 0) + count
        self.primitive_costs[op] = self.primitive_costs.get(op, 0) + cost_ns * count
        self.total_ops += count
        self.total_time_ns += cost_ns * count
    
    def get_total_time_ms(self) -> float:
        """Get total time in milliseconds."""
        return self.total_time_ns / 1_000_000.0
    
    def get_total_time_us(self) -> float:
        """Get total time in microseconds."""
        return self.total_time_ns / 1000.0
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by operation."""
        breakdown = {}
        for op, cost in self.primitive_costs.items():
            op_name = get_op_name(op)
            breakdown[op_name] = cost
        return breakdown
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CostProfile(ops={self.total_ops}, "
                f"time={self.get_total_time_ms():.3f}ms, "
                f"depth={self.max_depth}, "
                f"nodes={self.node_count})")


class CostModel:
    """
    Cost model for estimating expression execution time.
    Uses precomputed primitive costs and structural analysis.
    """
    
    def __init__(self):
        """Initialize the cost model."""
        # Base costs for primitive operations (in nanoseconds)
        self._base_costs: Dict[OpID, float] = {}
        self._initialize_base_costs()
        
        # Cache for computed costs
        self._cache: Dict[int, CostProfile] = {}
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'profiled_expressions': 0,
        }
    
    def _initialize_base_costs(self):
        """Initialize base costs for operations."""
        # Estimated costs in nanoseconds on typical hardware
        base_costs = {
            # Identity and constants
            OpID.IDENTITY: 0.1,
            OpID.CONST_ZERO: 0.1,
            OpID.CONST_ONE: 0.1,
            
            # Arithmetic (binary)
            OpID.ADD: 0.5,
            OpID.SUB: 0.5,
            OpID.MUL: 0.5,
            OpID.DIV: 2.0,
            OpID.POWER: 5.0,
            
            # Arithmetic (unary)
            OpID.NEG: 0.3,
            OpID.ABS: 0.4,
            OpID.SQRT: 3.0,
            OpID.CBRT: 4.0,
            OpID.INVERSE: 2.0,
            
            # Exponential
            OpID.EXP: 3.0,
            OpID.EXP2: 3.0,
            OpID.EXP10: 3.0,
            OpID.EXPM1: 3.0,
            
            # Logarithmic
            OpID.LOG: 3.0,
            OpID.LOG2: 3.0,
            OpID.LOG10: 3.0,
            OpID.LOG1P: 3.0,
            
            # Trigonometric
            OpID.SIN: 4.0,
            OpID.COS: 4.0,
            OpID.TAN: 5.0,
            OpID.COT: 5.0,
            OpID.SEC: 5.0,
            OpID.CSC: 5.0,
            
            # Inverse Trigonometric
            OpID.ASIN: 5.0,
            OpID.ACOS: 5.0,
            OpID.ATAN: 5.0,
            OpID.ACOT: 5.0,
            
            # Hyperbolic
            OpID.SINH: 5.0,
            OpID.COSH: 5.0,
            OpID.TANH: 5.0,
            OpID.COTH: 5.0,
            OpID.ASINH: 5.0,
            OpID.ACOSH: 5.0,
            OpID.ATANH: 5.0,
            
            # Activation
            OpID.SIGMOID: 6.0,
            OpID.RELU: 1.0,
            OpID.LEAKY_RELU: 1.0,
            OpID.GELU: 10.0,
            OpID.SWISH: 8.0,
            OpID.ELU: 5.0,
            OpID.SOFTPLUS: 5.0,
            
            # Special
            OpID.ERF: 15.0,
            OpID.GAMMA: 20.0,
            OpID.SIGMA: 10.0,
            OpID.BETA: 25.0,
            
            # Reduction
            OpID.SUM: 5.0,
            OpID.MEAN: 5.0,
            OpID.VAR: 10.0,
            OpID.STD: 10.0,
            OpID.MAX: 5.0,
            OpID.MIN: 5.0,
            
            # Logical
            OpID.WHERE: 2.0,
            OpID.GREATER: 0.5,
            OpID.LESS: 0.5,
            OpID.EQUAL: 0.5,
            OpID.AND: 0.3,
            OpID.OR: 0.3,
            OpID.NOT: 0.2,
            
            # Calculus
            OpID.DERIVATIVE: 10.0,
            OpID.SECOND_DERIVATIVE: 20.0,
            OpID.INTEGRAL: 30.0,
            
            # Statistical
            OpID.GAUSSIAN: 8.0,
            OpID.LOGISTIC: 6.0,
            
            # Custom
            OpID.CLIP: 1.5,
            OpID.SIGN: 0.5,
            OpID.ROUND: 1.0,
            OpID.FLOOR: 1.0,
            OpID.CEIL: 1.0,
            
            # Composite
            OpID.SQUARE: 0.6,
            OpID.CUBE: 0.7,
            OpID.LOGIT: 6.0,
            OpID.SOFTMAX: 10.0,
            
            # Gradient
            OpID.STOP_GRADIENT: 0.1,
            OpID.IDENTITY_GRAD: 0.1,
        }
        
        self._base_costs.update(base_costs)
    
    def set_primitive_cost(self, op: OpID, cost_ns: float) -> None:
        """Set the cost for a primitive operation."""
        self._base_costs[op] = cost_ns
        # Clear cache
        self._cache.clear()
    
    def get_primitive_cost(self, op: OpID) -> float:
        """Get the cost for a primitive operation."""
        if op in self._base_costs:
            return self._base_costs[op]
        # Default cost
        return 1.0
    
    def profile(self, handle: int) -> CostProfile:
        """
        Profile an expression and compute its cost.
        
        Args:
            handle: The expression handle.
            
        Returns:
            CostProfile for the expression.
        """
        # Check cache
        if handle in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[handle]
        
        self._stats['cache_misses'] += 1
        self._stats['profiled_expressions'] += 1
        
        profile = CostProfile()
        self._profile_expression(handle, profile)
        
        # Update depth and node count
        profile.max_depth = get_depth(handle, lookup)
        profile.node_count = count_nodes_in_subtree(handle, lookup)
        
        # Cache the profile
        self._cache[handle] = profile
        
        return profile
    
    def _profile_expression(self, handle: int, profile: CostProfile) -> None:
        """
        Recursively profile an expression.
        
        Args:
            handle: The expression handle.
            profile: The cost profile to update.
        """
        node = lookup(handle)
        if node is None:
            return
        
        # Get cost for this operation
        cost = self.get_primitive_cost(node.op)
        profile.add_primitive(node.op, cost)
        
        # Profile children
        for child in node.get_children():
            if child is not None:
                self._profile_expression(child, profile)
    
    def estimate_time(self, handle: int, batch_size: int = 1) -> float:
        """
        Estimate execution time for a batch.
        
        Args:
            handle: The expression handle.
            batch_size: Batch size.
            
        Returns:
            Estimated time in milliseconds.
        """
        profile = self.profile(handle)
        total_ns = profile.total_time_ns * batch_size
        return total_ns / 1_000_000.0  # Convert to ms
    
    def compare_expressions(self, h1: int, h2: int) -> Dict[str, Any]:
        """
        Compare the cost of two expressions.
        
        Args:
            h1: First expression.
            h2: Second expression.
            
        Returns:
            Comparison results.
        """
        p1 = self.profile(h1)
        p2 = self.profile(h2)
        
        return {
            'h1_time': p1.get_total_time_ms(),
            'h2_time': p2.get_total_time_ms(),
            'h1_nodes': p1.node_count,
            'h2_nodes': p2.node_count,
            'h1_depth': p1.max_depth,
            'h2_depth': p2.max_depth,
            'time_ratio': p1.get_total_time_ms() / (p2.get_total_time_ms() + 1e-10),
            'node_ratio': p1.node_count / (p2.node_count + 1e-10),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost model statistics."""
        return {
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_ratio': self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'] + 1),
            'profiled_expressions': self._stats['profiled_expressions'],
            'cache_size': len(self._cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the cost cache."""
        self._cache.clear()
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0


# Global cost model
_COST_MODEL: Optional[CostModel] = None


def get_cost_model() -> CostModel:
    """Get or create the global cost model."""
    global _COST_MODEL
    if _COST_MODEL is None:
        _COST_MODEL = CostModel()
    return _COST_MODEL


def profile_expression(handle: int) -> CostProfile:
    """Profile an expression."""
    return get_cost_model().profile(handle)


def estimate_time(handle: int, batch_size: int = 1) -> float:
    """Estimate execution time for an expression."""
    return get_cost_model().estimate_time(handle, batch_size)