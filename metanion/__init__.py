__version__ = "3.5.9"
__author__ = "Metanion Community"

from .api import Metanion
from .autometanion import AutoMetanion
from .knowledge_manager import SmartKnowledgeManager, SmartTrainer
from .gp import GPIndividual, PopulationInitializer, InitializationMethod, RobustFitness, PopulationManager
from .symbolic import OpID, intern, lookup, get_pool, reset_pool
from .compile import compile_handle

__all__ = [
    '__version__', '__author__',
    'Metanion', 'AutoMetanion',
    'SmartKnowledgeManager', 'SmartTrainer',
    'GPIndividual', 'PopulationInitializer', 'InitializationMethod',
    'RobustFitness', 'PopulationManager',
    'OpID', 'intern', 'lookup', 'get_pool', 'reset_pool',
    'compile_handle',
]
