"""
Mutation operators for genetic programming.
"""

from typing import List, Optional, Tuple, Dict, Any
import random

from .individual import GPIndividual
from ..symbolic import OpID, intern, lookup, get_pool
from ..symbolic import get_op_arity, get_op_name, get_all_operation_ids
from ..exceptions import ExpressionError


class MutationOperator:
    """
    Base class for mutation operators.
    """
    
    def __init__(self, max_depth: int = 10, op_set: Optional[List[OpID]] = None):
        """
        Initialize the mutation operator.
        
        Args:
            max_depth: Maximum depth of expressions after mutation.
            op_set: Set of operations to use.
        """
        self.max_depth = max_depth
        self.op_set = op_set or get_all_operation_ids()
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """
        Mutate an individual.
        
        Args:
            individual: The individual to mutate.
            
        Returns:
            The mutated individual.
        """
        raise NotImplementedError("Subclasses must implement mutate()")


class PointMutation(MutationOperator):
    """
    Point mutation - changes a single operation.
    """
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """Perform point mutation."""
        offspring = individual.copy()
        
        # Get all handles from the individual
        handles = []
        for i, h in enumerate(offspring.weight_handles):
            if h is not None:
                handles.append((i, h))
        
        if offspring.bias_handle is not None:
            handles.append(('bias', offspring.bias_handle))
        
        if not handles:
            return offspring
        
        # Select a random handle to mutate
        idx, handle = random.choice(handles)
        
        # Get the node
        node = lookup(handle)
        if node is None:
            return offspring
        
        # Select a new operation of the same arity
        arity = get_op_arity(node.op)
        candidates = [op for op in self.op_set if get_op_arity(op) == arity]
        
        if not candidates:
            return offspring
        
        # Choose a new operation
        new_op = random.choice(candidates)
        
        # Create the new handle
        if arity == 0:
            new_handle = intern(new_op)
        elif arity == 1:
            # Keep the child
            new_handle = intern(new_op, node.left)
        elif arity == 2:
            # Keep the children
            new_handle = intern(new_op, node.left, node.right)
        else:
            return offspring
        
        # Replace the handle
        if idx == 'bias':
            offspring.bias_handle = new_handle
        else:
            offspring.weight_handles[idx] = new_handle
        
        # Simplify the offspring
        offspring.simplify()
        
        return offspring


class SubtreeMutation(MutationOperator):
    """
    Subtree mutation - replaces a subtree with a random one.
    """
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """Perform subtree mutation."""
        offspring = individual.copy()
        
        # Get all handles from the individual
        handles = []
        for i, h in enumerate(offspring.weight_handles):
            if h is not None:
                handles.append((i, h))
        
        if offspring.bias_handle is not None:
            handles.append(('bias', offspring.bias_handle))
        
        if not handles:
            return offspring
        
        # Select a random handle to mutate
        idx, handle = random.choice(handles)
        
        # Generate a new random subtree
        depth = random.randint(1, self.max_depth)
        new_handle = self._generate_random_tree(depth)
        
        # Replace the handle
        if idx == 'bias':
            offspring.bias_handle = new_handle
        else:
            offspring.weight_handles[idx] = new_handle
        
        # Simplify the offspring
        offspring.simplify()
        
        return offspring
    
    def _generate_random_tree(self, max_depth: int) -> int:
        """
        Generate a random expression tree.
        
        Args:
            max_depth: Maximum depth of the tree.
            
        Returns:
            Handle of the generated expression.
        """
        if max_depth <= 0:
            # Leaf node
            op = random.choice([OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE])
            return intern(op)
        
        # Choose a random operation
        op = random.choice(self.op_set)
        arity = get_op_arity(op)
        
        if arity == 0:
            return intern(op)
        elif arity == 1:
            child = self._generate_random_tree(max_depth - 1)
            return intern(op, child)
        elif arity == 2:
            left = self._generate_random_tree(max_depth - 1)
            right = self._generate_random_tree(max_depth - 1)
            return intern(op, left, right)
        else:
            raise ExpressionError(f"Unsupported arity: {arity}")


class ShrinkMutation(MutationOperator):
    """
    Shrink mutation - replaces a subtree with one of its children.
    """
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """Perform shrink mutation."""
        offspring = individual.copy()
        
        # Get all handles from the individual
        handles = []
        for i, h in enumerate(offspring.weight_handles):
            if h is not None:
                handles.append((i, h))
        
        if offspring.bias_handle is not None:
            handles.append(('bias', offspring.bias_handle))
        
        if not handles:
            return offspring
        
        # Select a random handle to mutate
        idx, handle = random.choice(handles)
        
        # Get the node
        node = lookup(handle)
        if node is None or node.arity == 0:
            return offspring
        
        # Choose a random child to keep
        children = node.get_children()
        if not children:
            return offspring
        
        child = random.choice(children)
        
        # Replace the handle with the child
        if idx == 'bias':
            offspring.bias_handle = child
        else:
            offspring.weight_handles[idx] = child
        
        # Simplify the offspring
        offspring.simplify()
        
        return offspring


class HoistMutation(MutationOperator):
    """
    Hoist mutation - replaces a subtree with a subtree from the same tree.
    """
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """Perform hoist mutation."""
        # TODO: Implement hoist mutation
        return individual.copy()


class GaussianMutation(MutationOperator):
    """
    Gaussian mutation - adds Gaussian noise to numeric parts.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        op_set: Optional[List[OpID]] = None,
        sigma: float = 0.1
    ):
        """
        Initialize Gaussian mutation.
        
        Args:
            max_depth: Maximum depth of expressions after mutation.
            op_set: Set of operations to use.
            sigma: Standard deviation of Gaussian noise.
        """
        super().__init__(max_depth, op_set)
        self.sigma = sigma
    
    def mutate(self, individual: GPIndividual) -> GPIndividual:
        """Perform Gaussian mutation."""
        # TODO: Implement Gaussian mutation for numeric constants
        return individual.copy()