"""
Genetic programming individual representation for Metanion.
Each individual is a Metanion Tensor with symbolic weights.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import copy
import random

from ..core import Tensor, DType
from ..symbolic import OpID, intern, lookup, get_pool, get_op_name
from ..symbolic import get_depth, count_nodes_in_subtree, simplify
from ..compile import compile_handle
from ..exceptions import ExpressionError, DepthLimitExceededError


@dataclass
class GPIndividual:
    """
    A single individual in the genetic programming population.
    Represents a complete Metanion model with symbolic weights.
    """
    
    # Core representation: flat list of handles (weights) and bias handle
    weight_handles: List[int] = field(default_factory=list)
    bias_handle: Optional[int] = None
    shape: Tuple[int, ...] = (1, 1)  # Input shape (features, output_dim)
    
    # Metadata
    fitness: float = float('inf')
    depth: int = 0
    node_count: int = 0
    inference_time: float = 0.0
    generation: int = 0
    id: int = field(default_factory=lambda: random.randint(0, 10**9))
    
    # Cached compiled function
    _compiled_func: Optional[callable] = None
    
    def __post_init__(self):
        """Initialize derived attributes."""
        self._update_stats()
    
    def _update_stats(self) -> None:
        """Update depth and node count statistics."""
        pool = get_pool()
        
        max_depth = 0
        total_nodes = 0
        
        # Check weights
        for handle in self.weight_handles:
            if handle is not None:
                depth = get_depth(handle, pool.get_node)
                nodes = count_nodes_in_subtree(handle, pool.get_node)
                max_depth = max(max_depth, depth)
                total_nodes += nodes
        
        # Check bias
        if self.bias_handle is not None:
            depth = get_depth(self.bias_handle, pool.get_node)
            nodes = count_nodes_in_subtree(self.bias_handle, pool.get_node)
            max_depth = max(max_depth, depth)
            total_nodes += nodes
        
        self.depth = max_depth
        self.node_count = total_nodes
    
    def evaluate(self, X: Tensor) -> Tensor:
        """
        Evaluate the individual on input data.
        
        Args:
            X: Input tensor.
            
        Returns:
            Output tensor.
        """
        # TODO: Implement full evaluation with weight tensor
        # For now, return a placeholder
        return Tensor.zeros(X.shape)
    
    def get_expression(self) -> str:
        """
        Get a string representation of the individual's expression.
        
        Returns:
            String representation.
        """
        pool = get_pool()
        parts = []
        
        for i, handle in enumerate(self.weight_handles):
            if handle is not None:
                node = lookup(handle)
                if node is not None:
                    parts.append(f"w{i}: {get_op_name(node.op)}")
        
        if self.bias_handle is not None:
            node = lookup(self.bias_handle)
            if node is not None:
                parts.append(f"bias: {get_op_name(node.op)}")
        
        return f"GPIndividual(id={self.id}, {', '.join(parts)})"
    
    def compile(self) -> callable:
        """
        Compile the individual to a callable function.
        
        Returns:
            Compiled function.
        """
        if self._compiled_func is None:
            # TODO: Compile the full model
            # For now, compile the first weight
            if self.weight_handles:
                self._compiled_func = compile_handle(self.weight_handles[0])
            else:
                self._compiled_func = lambda x: x
        return self._compiled_func
    
    def simplify(self) -> None:
        """
        Simplify the individual's expressions.
        """
        pool = get_pool()
        
        for i, handle in enumerate(self.weight_handles):
            if handle is not None:
                self.weight_handles[i] = simplify(handle)
        
        if self.bias_handle is not None:
            self.bias_handle = simplify(self.bias_handle)
        
        self._update_stats()
    
    def copy(self) -> 'GPIndividual':
        """
        Create a deep copy of the individual.
        
        Returns:
            A new GPIndividual instance.
        """
        return GPIndividual(
            weight_handles=copy.deepcopy(self.weight_handles),
            bias_handle=self.bias_handle,
            shape=self.shape,
            generation=self.generation,
            id=random.randint(0, 10**9)
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"GPIndividual(id={self.id}, depth={self.depth}, "
                f"nodes={self.node_count}, fitness={self.fitness:.6f})")


class IndividualFactory:
    """
    Factory for creating GP individuals with various initialization strategies.
    """
    
    @staticmethod
    def create_random(
        shape: Tuple[int, ...],
        max_depth: int = 5,
        op_set: Optional[List[OpID]] = None
    ) -> GPIndividual:
        """
        Create a random individual.
        
        Args:
            shape: Shape of the weight tensor.
            max_depth: Maximum depth of expressions.
            op_set: Set of operations to use (default: all).
            
        Returns:
            A new GPIndividual instance.
        """
        from ..symbolic import get_all_operation_ids, get_op_arity
        
        if op_set is None:
            op_set = get_all_operation_ids()
        
        weight_handles = []
        
        for _ in range(shape[0] * shape[1]):
            handle = IndividualFactory._generate_random_tree(max_depth, op_set)
            weight_handles.append(handle)
        
        # Generate random bias
        bias_handle = IndividualFactory._generate_random_tree(max_depth, op_set)
        
        return GPIndividual(
            weight_handles=weight_handles,
            bias_handle=bias_handle,
            shape=shape
        )
    
    @staticmethod
    def _generate_random_tree(
        max_depth: int,
        op_set: List[OpID],
        current_depth: int = 0
    ) -> int:
        """
        Generate a random expression tree.
        
        Args:
            max_depth: Maximum depth.
            op_set: Set of operations.
            current_depth: Current depth in the tree.
            
        Returns:
            Handle of the generated expression.
        """
        if current_depth >= max_depth:
            # Leaf node: identity or constant
            choices = [OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE]
            op = random.choice(choices)
            return intern(op)
        
        # Choose a random operation
        op = random.choice(op_set)
        arity = get_op_arity(op)
        
        if arity == 0:
            return intern(op)
        elif arity == 1:
            child = IndividualFactory._generate_random_tree(
                max_depth, op_set, current_depth + 1
            )
            return intern(op, child)
        elif arity == 2:
            left = IndividualFactory._generate_random_tree(
                max_depth, op_set, current_depth + 1
            )
            right = IndividualFactory._generate_random_tree(
                max_depth, op_set, current_depth + 1
            )
            return intern(op, left, right)
        else:
            raise ExpressionError(f"Unsupported arity: {arity}")
    
    @staticmethod
    def create_from_template(
        template: List[OpID],
        shape: Tuple[int, ...]
    ) -> GPIndividual:
        """
        Create an individual from a template of operations.
        
        Args:
            template: List of operations to use as weights.
            shape: Shape of the weight tensor.
            
        Returns:
            A new GPIndividual instance.
        """
        weight_handles = []
        
        for i in range(shape[0] * shape[1]):
            op = template[i % len(template)]
            # Create a simple expression from the template
            if get_op_arity(op) == 0:
                handle = intern(op)
            elif get_op_arity(op) == 1:
                handle = intern(op, intern(OpID.IDENTITY))
            else:
                handle = intern(op, intern(OpID.IDENTITY), intern(OpID.CONST_ONE))
            weight_handles.append(handle)
        
        # Simple bias
        bias_handle = intern(OpID.CONST_ZERO)
        
        return GPIndividual(
            weight_handles=weight_handles,
            bias_handle=bias_handle,
            shape=shape
        )
    
    @staticmethod
    def create_identity(shape: Tuple[int, ...]) -> GPIndividual:
        """
        Create an identity individual (f(x) = x).
        
        Args:
            shape: Shape of the weight tensor.
            
        Returns:
            A new GPIndividual instance.
        """
        identity_handle = intern(OpID.IDENTITY)
        weight_handles = [identity_handle] * (shape[0] * shape[1])
        
        return GPIndividual(
            weight_handles=weight_handles,
            bias_handle=intern(OpID.CONST_ZERO),
            shape=shape
        )