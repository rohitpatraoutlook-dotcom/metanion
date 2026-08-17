"""
Population initialization strategies for genetic programming.
"""

from typing import List, Optional, Tuple, Dict, Any
import random
from enum import Enum

from .individual import GPIndividual, IndividualFactory
from ..symbolic import OpID, get_all_operation_ids, get_op_arity


class InitializationMethod(Enum):
    """Methods for initializing the GP population."""
    RAMPED_HALF_HALF = "ramped_half_half"
    FULL = "full"
    GROW = "grow"
    IDENTITY = "identity"
    RANDOM = "random"
    SEEDED = "seeded"


class PopulationInitializer:
    """
    Handles initialization of GP populations with various strategies.
    """
    
    def __init__(
        self,
        pop_size: int,
        shape: Tuple[int, ...],
        max_depth: int = 5,
        op_set: Optional[List[OpID]] = None,
        method: InitializationMethod = InitializationMethod.RAMPED_HALF_HALF,
        seed_individuals: Optional[List[GPIndividual]] = None
    ):
        """
        Initialize the population initializer.
        
        Args:
            pop_size: Size of the population.
            shape: Shape of the weight tensor.
            max_depth: Maximum depth of expressions.
            op_set: Set of operations to use.
            method: Initialization method.
            seed_individuals: Seed individuals for seeded initialization.
        """
        self.pop_size = pop_size
        self.shape = shape
        self.max_depth = max_depth
        self.op_set = op_set or get_all_operation_ids()
        self.method = method
        self.seed_individuals = seed_individuals or []
    
    def initialize(self) -> List[GPIndividual]:
        """
        Initialize the population.
        
        Returns:
            List of GPIndividual instances.
        """
        if self.method == InitializationMethod.FULL:
            return self._initialize_full()
        elif self.method == InitializationMethod.GROW:
            return self._initialize_grow()
        elif self.method == InitializationMethod.RAMPED_HALF_HALF:
            return self._initialize_ramped_half_half()
        elif self.method == InitializationMethod.IDENTITY:
            return self._initialize_identity()
        elif self.method == InitializationMethod.SEEDED:
            return self._initialize_seeded()
        else:  # RANDOM
            return self._initialize_random()
    
    def _initialize_full(self) -> List[GPIndividual]:
        """Initialize with full trees (all leaves at max depth)."""
        population = []
        
        for _ in range(self.pop_size):
            # Generate random full tree
            individual = IndividualFactory.create_random(
                self.shape,
                self.max_depth,
                self.op_set
            )
            population.append(individual)
        
        return population
    
    def _initialize_grow(self) -> List[GPIndividual]:
        """Initialize with grow trees (variable depth)."""
        population = []
        
        for _ in range(self.pop_size):
            # Generate random grow tree (max depth but variable)
            individual = IndividualFactory.create_random(
                self.shape,
                self.max_depth,
                self.op_set
            )
            population.append(individual)
        
        return population
    
    def _initialize_ramped_half_half(self) -> List[GPIndividual]:
        """
        Initialize with ramped half-half method.
        Half the population with full trees, half with grow trees,
        varying depths from 1 to max_depth.
        """
        population = []
        depth_range = range(1, self.max_depth + 1)
        
        for depth in depth_range:
            for _ in range(self.pop_size // (2 * len(depth_range))):
                # Full tree
                individual = IndividualFactory.create_random(
                    self.shape,
                    depth,
                    self.op_set
                )
                population.append(individual)
            
            for _ in range(self.pop_size // (2 * len(depth_range))):
                # Grow tree
                individual = IndividualFactory.create_random(
                    self.shape,
                    depth,
                    self.op_set
                )
                population.append(individual)
        
        # Fill remaining with random
        while len(population) < self.pop_size:
            individual = IndividualFactory.create_random(
                self.shape,
                self.max_depth,
                self.op_set
            )
            population.append(individual)
        
        return population
    
    def _initialize_identity(self) -> List[GPIndividual]:
        """Initialize all individuals as identity functions."""
        population = []
        
        for _ in range(self.pop_size):
            individual = IndividualFactory.create_identity(self.shape)
            population.append(individual)
        
        return population
    
    def _initialize_seeded(self) -> List[GPIndividual]:
        """Initialize with seed individuals plus random ones."""
        population = []
        
        # Add seed individuals
        for seed in self.seed_individuals:
            population.append(seed.copy())
        
        # Fill remaining with random
        while len(population) < self.pop_size:
            individual = IndividualFactory.create_random(
                self.shape,
                self.max_depth,
                self.op_set
            )
            population.append(individual)
        
        return population
    
    def _initialize_random(self) -> List[GPIndividual]:
        """Initialize with completely random individuals."""
        population = []
        
        for _ in range(self.pop_size):
            individual = IndividualFactory.create_random(
                self.shape,
                self.max_depth,
                self.op_set
            )
            population.append(individual)
        
        return population
    
    def get_methods(self) -> List[InitializationMethod]:
        """Get all available initialization methods."""
        return list(InitializationMethod)