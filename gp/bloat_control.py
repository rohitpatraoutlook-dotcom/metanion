"""
Bloat control mechanisms for genetic programming.
Prevents uncontrolled growth of expression trees.
"""

from typing import List, Optional, Tuple, Dict, Any, Set
import random
from dataclasses import dataclass, field

from .individual import GPIndividual
from ..symbolic import get_depth, count_nodes_in_subtree, simplify


@dataclass
class BloatControl:
    """
    Configuration for bloat control.
    """
    
    max_depth: int = 10
    max_nodes: int = 100
    parsimony_pressure: float = 0.01
    use_age_layers: bool = True
    use_parsimony: bool = True
    use_double_elitism: bool = True
    age_layers: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    def __post_init__(self):
        """Sort age layers."""
        self.age_layers.sort()


class BloatController:
    """
    Controls bloat in the GP population.
    Implements multiple strategies to limit expression growth.
    """
    
    def __init__(self, config: Optional[BloatControl] = None):
        """
        Initialize the bloat controller.
        
        Args:
            config: Bloat control configuration.
        """
        self.config = config or BloatControl()
        self._stats = {
            'truncations': 0,
            'simplifications': 0,
            'age_resets': 0,
        }
    
    def apply_control(self, population: List[GPIndividual]) -> List[GPIndividual]:
        """
        Apply bloat control to the population.
        
        Args:
            population: The population to control.
            
        Returns:
            The controlled population.
        """
        controlled = []
        
        for individual in population:
            # Apply simplifications
            if self.config.use_parsimony:
                individual = self._apply_parsimony(individual)
            
            # Check if individual exceeds limits
            if self._is_too_large(individual):
                # Try to shrink it
                individual = self._shrink_individual(individual)
            
            # Check again after shrinking
            if self._is_too_large(individual):
                # If still too large, truncate
                individual = self._truncate_individual(individual)
                self._stats['truncations'] += 1
            
            # Apply age layers
            if self.config.use_age_layers:
                individual = self._apply_age_layers(individual)
            
            controlled.append(individual)
        
        return controlled
    
    def _is_too_large(self, individual: GPIndividual) -> bool:
        """
        Check if an individual exceeds size limits.
        
        Args:
            individual: The individual to check.
            
        Returns:
            True if the individual is too large.
        """
        return (individual.depth > self.config.max_depth or 
                individual.node_count > self.config.max_nodes)
    
    def _apply_parsimony(self, individual: GPIndividual) -> GPIndividual:
        """
        Apply parsimony pressure by simplifying.
        
        Args:
            individual: The individual to simplify.
            
        Returns:
            The simplified individual.
        """
        # Simplify the individual
        individual.simplify()
        self._stats['simplifications'] += 1
        
        return individual
    
    def _shrink_individual(self, individual: GPIndividual) -> GPIndividual:
        """
        Shrink an individual by replacing subtrees.
        
        Args:
            individual: The individual to shrink.
            
        Returns:
            The shrunken individual.
        """
        # Try to shrink by replacing deep subtrees
        offspring = individual.copy()
        
        # Get all handles
        handles = []
        for i, h in enumerate(offspring.weight_handles):
            if h is not None:
                handles.append((i, h))
        
        if offspring.bias_handle is not None:
            handles.append(('bias', offspring.bias_handle))
        
        # Sort by depth (deepest first)
        pool = None  # We'll use the global pool
        handles.sort(key=lambda x: get_depth(x[1], pool), reverse=True)
        
        # Try to shrink the deepest handles
        for idx, handle in handles[:min(5, len(handles))]:
            # Replace with identity or constant
            choices = [
                intern(OpID.IDENTITY),
                intern(OpID.CONST_ZERO),
                intern(OpID.CONST_ONE),
                intern(OpID.ADD, handle, intern(OpID.CONST_ZERO)),
            ]
            new_handle = random.choice(choices)
            
            if idx == 'bias':
                offspring.bias_handle = new_handle
            else:
                offspring.weight_handles[idx] = new_handle
        
        # Simplify
        offspring.simplify()
        
        return offspring
    
    def _truncate_individual(self, individual: GPIndividual) -> GPIndividual:
        """
        Truncate an individual to within limits.
        
        Args:
            individual: The individual to truncate.
            
        Returns:
            The truncated individual.
        """
        offspring = individual.copy()
        
        # Get all handles
        handles = []
        for i, h in enumerate(offspring.weight_handles):
            if h is not None:
                handles.append((i, h))
        
        if offspring.bias_handle is not None:
            handles.append(('bias', offspring.bias_handle))
        
        # Sort by depth (deepest first)
        pool = None
        handles.sort(key=lambda x: get_depth(x[1], pool), reverse=True)
        
        # Replace handles until within limits
        for idx, handle in handles:
            # Replace with identity
            new_handle = intern(OpID.IDENTITY)
            
            if idx == 'bias':
                offspring.bias_handle = new_handle
            else:
                offspring.weight_handles[idx] = new_handle
            
            # Check if within limits
            offspring._update_stats()
            if not self._is_too_large(offspring):
                break
        
        # Simplify
        offspring.simplify()
        
        return offspring
    
    def _apply_age_layers(self, individual: GPIndividual) -> GPIndividual:
        """
        Apply age layers to the individual.
        
        Args:
            individual: The individual.
            
        Returns:
            The individual with age layers applied.
        """
        # Check if individual is in a new age layer
        for layer in self.config.age_layers:
            if individual.generation > 0 and individual.generation % layer == 0:
                # Reset part of the individual
                if random.random() < 0.1:
                    self._stats['age_resets'] += 1
                    # Replace a random weight with identity
                    idx = random.randint(0, len(individual.weight_handles) - 1)
                    individual.weight_handles[idx] = intern(OpID.IDENTITY)
                    individual.simplify()
                    break
        
        return individual
    
    def get_stats(self) -> Dict[str, int]:
        """Get bloat control statistics."""
        return self._stats
    
    def print_stats(self) -> None:
        """Print bloat control statistics."""
        stats = self.get_stats()
        print("=" * 40)
        print("Bloat Control Statistics")
        print("=" * 40)
        print(f"Truncations:      {stats['truncations']}")
        print(f"Simplifications:  {stats['simplifications']}")
        print(f"Age Resets:       {stats['age_resets']}")
        print("=" * 40)


class ParetoBloatControl(BloatController):
    """
    Pareto-based bloat control.
    Uses Pareto optimization to balance size and fitness.
    """
    
    def __init__(
        self,
        config: Optional[BloatControl] = None,
        size_weight: float = 0.5
    ):
        """
        Initialize Pareto bloat control.
        
        Args:
            config: Bloat control configuration.
            size_weight: Weight for size in Pareto fitness.
        """
        super().__init__(config)
        self.size_weight = size_weight
    
    def apply_control(self, population: List[GPIndividual]) -> List[GPIndividual]:
        """
        Apply Pareto-based bloat control.
        
        Args:
            population: The population to control.
            
        Returns:
            The controlled population.
        """
        # First apply standard controls
        controlled = super().apply_control(population)
        
        # Then apply Pareto selection
        # Keep individuals that are not Pareto-dominated
        # in terms of (fitness, size)
        pareto_individuals = []
        
        for i, ind1 in enumerate(controlled):
            dominated = False
            for j, ind2 in enumerate(controlled):
                if i != j:
                    # Check if ind2 dominates ind1
                    if (ind2.fitness <= ind1.fitness and 
                        ind2.node_count <= ind1.node_count and
                        (ind2.fitness < ind1.fitness or ind2.node_count < ind1.node_count)):
                        dominated = True
                        break
            if not dominated:
                pareto_individuals.append(ind1)
        
        return pareto_individuals