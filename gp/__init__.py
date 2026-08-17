"""
Genetic programming engine for the Metanion engine.
"""

from .individual import GPIndividual, IndividualFactory
from .initialization import PopulationInitializer, InitializationMethod
from .selection import (
    TournamentSelection,
    RouletteSelection,
    RankSelection,
    ElitismSelection,
    StochasticUniversalSampling,
)
from .crossover import (
    SubtreeCrossover,
    OnePointCrossover,
    UniformCrossover,
)
from .mutation import (
    PointMutation,
    SubtreeMutation,
    ShrinkMutation,
    HoistMutation,
    GaussianMutation,
)
from .fitness import (
    FitnessEvaluator,
    MultiObjectiveFitness,
    ParetoFitness,
)
from .population import PopulationManager
from .bloat_control import (
    BloatControl,
    BloatController,
    ParetoBloatControl,
)

__all__ = [
    # Individual
    'GPIndividual',
    'IndividualFactory',
    
    # Initialization
    'PopulationInitializer',
    'InitializationMethod',
    
    # Selection
    'TournamentSelection',
    'RouletteSelection',
    'RankSelection',
    'ElitismSelection',
    'StochasticUniversalSampling',
    
    # Crossover
    'SubtreeCrossover',
    'OnePointCrossover',
    'UniformCrossover',
    
    # Mutation
    'PointMutation',
    'SubtreeMutation',
    'ShrinkMutation',
    'HoistMutation',
    'GaussianMutation',
    
    # Fitness
    'FitnessEvaluator',
    'MultiObjectiveFitness',
    'ParetoFitness',
    
    # Population
    'PopulationManager',
    
    # Bloat Control
    'BloatControl',
    'BloatController',
    'ParetoBloatControl',
]