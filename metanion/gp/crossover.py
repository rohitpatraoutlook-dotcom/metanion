"""
Crossover operators for genetic programming.
"""

from typing import List, Optional, Tuple, Dict, Any, Set
import random

from .individual import GPIndividual
from ..symbolic import intern, lookup, get_pool, get_op_arity
from ..symbolic import get_depth, count_nodes_in_subtree, simplify
from ..exceptions import ExpressionError


class CrossoverOperator:
    """
    Base class for crossover operators.
    """
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 100):
        """
        Initialize the crossover operator.
        
        Args:
            max_depth: Maximum depth of resulting expressions.
            max_nodes: Maximum number of nodes in resulting expressions.
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
    
    def crossover(
        self,
        parent1: GPIndividual,
        parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent.
            parent2: Second parent.
            
        Returns:
            Two offspring individuals.
        """
        raise NotImplementedError("Subclasses must implement crossover()")
    
    def _is_valid(self, handle: int) -> bool:
        """
        Check if an expression is valid (within depth and node limits).
        
        Args:
            handle: The expression handle.
            
        Returns:
            True if valid, False otherwise.
        """
        pool = get_pool()
        depth = get_depth(handle, pool.get_node)
        nodes = count_nodes_in_subtree(handle, pool.get_node)
        return depth <= self.max_depth and nodes <= self.max_nodes


class SubtreeCrossover(CrossoverOperator):
    """
    Subtree crossover - swaps subtrees between parents.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        max_nodes: int = 100,
        uniform: bool = False
    ):
        """
        Initialize subtree crossover.
        
        Args:
            max_depth: Maximum depth of resulting expressions.
            max_nodes: Maximum number of nodes in resulting expressions.
            uniform: If True, choose crossover points uniformly.
        """
        super().__init__(max_depth, max_nodes)
        self.uniform = uniform
    
    def crossover(
        self,
        parent1: GPIndividual,
        parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform subtree crossover."""
        # Create copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Get all handles from both parents
        handles1 = self._get_all_handles(offspring1)
        handles2 = self._get_all_handles(offspring2)
        
        if not handles1 or not handles2:
            return offspring1, offspring2
        
        # Select crossover points
        if self.uniform:
            point1 = random.choice(handles1)
            point2 = random.choice(handles2)
        else:
            # Bias towards internal nodes (more likely to be selected)
            point1 = self._select_subtree(handles1)
            point2 = self._select_subtree(handles2)
        
        # Check if crossover is valid
        if not self._is_valid_crossover(point1, point2, offspring1, offspring2):
            return offspring1, offspring2
        
        # Perform crossover
        pool = get_pool()
        
        # Replace point1 in offspring1 with point2 from parent2
        # and point2 in offspring2 with point1 from parent1
        try:
            # This is simplified - actual implementation would need to traverse
            # the expression trees and replace subtrees
            pass
        except Exception:
            # If crossover fails, return parents
            return offspring1, offspring2
        
        # Simplify the offspring
        offspring1.simplify()
        offspring2.simplify()
        
        # Check validity
        if not self._is_valid(offspring1.weight_handles[0]) or not self._is_valid(offspring2.weight_handles[0]):
            return parent1.copy(), parent2.copy()
        
        return offspring1, offspring2
    
    def _get_all_handles(self, individual: GPIndividual) -> List[int]:
        """
        Get all handles from an individual.
        
        Args:
            individual: The individual.
            
        Returns:
            List of handles.
        """
        handles = []
        for handle in individual.weight_handles:
            if handle is not None:
                handles.append(handle)
        if individual.bias_handle is not None:
            handles.append(individual.bias_handle)
        return handles
    
    def _select_subtree(self, handles: List[int]) -> int:
        """
        Select a subtree biased towards internal nodes.
        
        Args:
            handles: List of handles.
            
        Returns:
            Selected handle.
        """
        # Weight selection towards internal nodes (non-leaf)
        weighted = []
        for h in handles:
            node = lookup(h)
            if node is not None and node.arity > 0:
                weighted.extend([h] * 3)  # Internal nodes get higher weight
            else:
                weighted.append(h)
        
        return random.choice(weighted) if weighted else random.choice(handles)
    
    def _is_valid_crossover(
        self,
        point1: int,
        point2: int,
        offspring1: GPIndividual,
        offspring2: GPIndividual
    ) -> bool:
        """
        Check if a crossover is valid (type compatibility).
        
        Args:
            point1: Crossover point in offspring1.
            point2: Crossover point in offspring2.
            offspring1: First offspring.
            offspring2: Second offspring.
            
        Returns:
            True if valid, False otherwise.
        """
        # Check type compatibility
        node1 = lookup(point1)
        node2 = lookup(point2)
        
        if node1 is None or node2 is None:
            return False
        
        # Check arity compatibility
        arity1 = get_op_arity(node1.op)
        arity2 = get_op_arity(node2.op)
        
        # For simplicity, only allow crossover of same arity
        return arity1 == arity2


class OnePointCrossover(CrossoverOperator):
    """
    One-point crossover - crossover at a single point.
    """
    
    def crossover(
        self,
        parent1: GPIndividual,
        parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform one-point crossover."""
        # TODO: Implement one-point crossover
        return parent1.copy(), parent2.copy()


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover - each gene is independently chosen from parents.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        max_nodes: int = 100,
        probability: float = 0.5
    ):
        """
        Initialize uniform crossover.
        
        Args:
            max_depth: Maximum depth of resulting expressions.
            max_nodes: Maximum number of nodes in resulting expressions.
            probability: Probability of taking from first parent.
        """
        super().__init__(max_depth, max_nodes)
        self.probability = probability
    
    def crossover(
        self,
        parent1: GPIndividual,
        parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform uniform crossover."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover weight handles
        for i in range(min(len(offspring1.weight_handles), len(offspring2.weight_handles))):
            if random.random() < self.probability:
                # Swap weights
                temp = offspring1.weight_handles[i]
                offspring1.weight_handles[i] = offspring2.weight_handles[i]
                offspring2.weight_handles[i] = temp
        
        # Crossover bias
        if random.random() < self.probability:
            temp = offspring1.bias_handle
            offspring1.bias_handle = offspring2.bias_handle
            offspring2.bias_handle = temp
        
        # Simplify the offspring
        offspring1.simplify()
        offspring2.simplify()
        
        return offspring1, offspring2