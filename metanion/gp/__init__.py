"""Genetic Programming module for Metanion."""

try:
    from .individual import GPIndividual, IndividualFactory
    from .initialization import PopulationInitializer, InitializationMethod
    from .selection import TournamentSelection, RouletteSelection, RankSelection, ElitismSelection
    from .crossover import SubtreeCrossover, OnePointCrossover, UniformCrossover
    from .mutation import PointMutation, SubtreeMutation, ShrinkMutation, HoistMutation, GaussianMutation
    from .fitness import FitnessEvaluator, MultiObjectiveFitness, ParetoFitness
    from .population import PopulationManager
    from .bloat_control import BloatControl, BloatController, ParetoBloatControl
except ImportError as e:
    print(f"GP module import error: {e}")
    # Define placeholder classes
    GPIndividual = None
    IndividualFactory = None
    PopulationInitializer = None
    InitializationMethod = None
    TournamentSelection = None
    RouletteSelection = None
    RankSelection = None
    ElitismSelection = None
    SubtreeCrossover = None
    OnePointCrossover = None
    UniformCrossover = None
    PointMutation = None
    SubtreeMutation = None
    ShrinkMutation = None
    HoistMutation = None
    GaussianMutation = None
    FitnessEvaluator = None
    MultiObjectiveFitness = None
    ParetoFitness = None
    PopulationManager = None
    BloatControl = None
    BloatController = None
    ParetoBloatControl = None

__all__ = [
    'GPIndividual', 'IndividualFactory',
    'PopulationInitializer', 'InitializationMethod',
    'TournamentSelection', 'RouletteSelection', 'RankSelection', 'ElitismSelection',
    'SubtreeCrossover', 'OnePointCrossover', 'UniformCrossover',
    'PointMutation', 'SubtreeMutation', 'ShrinkMutation', 'HoistMutation', 'GaussianMutation',
    'FitnessEvaluator', 'MultiObjectiveFitness', 'ParetoFitness',
    'PopulationManager',
    'BloatControl', 'BloatController', 'ParetoBloatControl',
]
