"""
Global configuration for the Metanion Engine.
All tunable parameters are centralized here.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetanionConfig:
    """
    Configuration dataclass for the Metanion engine.
    All parameters have sensible defaults.
    """
    
    # --- Expression Limits ---
    max_depth: int = 8
    """Maximum depth of any expression tree."""
    
    max_nodes: int = 256
    """Maximum number of nodes in any expression."""
    
    # --- Memory Limits ---
    arena_size_mb: int = 100
    """Pre-allocated memory arena size in megabytes."""
    
    max_handles: int = 1_000_000
    """Maximum number of unique expressions in the pool."""
    
    # --- Time & Performance ---
    max_inference_time_ms: float = 10.0
    """Maximum allowed inference time per batch in milliseconds."""
    
    jit_cache_size: int = 1000
    """Maximum number of compiled functions to cache."""
    
    # --- GP Hyperparameters ---
    population_size: int = 100
    """Number of individuals in the GP population."""
    
    generations: int = 50
    """Number of generations to evolve."""
    
    crossover_rate: float = 0.8
    """Probability of crossover between selected parents."""
    
    mutation_rate: float = 0.2
    """Probability of mutation after crossover."""
    
    tournament_size: int = 3
    """Number of individuals in tournament selection."""
    
    elitism_count: int = 5
    """Number of best individuals to preserve each generation."""
    
    # --- Regularization Weights ---
    lambda_depth: float = 0.01
    """Penalty per node depth in fitness."""
    
    lambda_time: float = 0.001
    """Penalty per millisecond in fitness."""
    
    # --- Data ---
    batch_size: int = 32
    """Batch size for training."""
    
    validation_split: float = 0.2
    """Fraction of data to use for validation."""
    
    # --- Randomness ---
    random_seed: Optional[int] = 42
    """Seed for reproducibility (None = no seed)."""
    
    # --- Logging ---
    verbose: bool = True
    """Print progress during training."""
    
    log_interval: int = 10
    """Log every N generations."""
    
    checkpoint_interval: int = 25
    """Save checkpoint every N generations."""
    
    # --- Hardware ---
    use_numba: bool = False
    """Use Numba JIT if available."""
    
    use_multiprocessing: bool = True
    """Use multiprocessing for parallel fitness evaluation."""
    
    num_workers: int = 4
    """Number of worker processes for parallel evaluation."""
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.lambda_depth < 0:
            raise ValueError("lambda_depth must be >= 0")
        if self.lambda_time < 0:
            raise ValueError("lambda_time must be >= 0")


# Global singleton configuration
DEFAULT_CONFIG = MetanionConfig()

# Active configuration (can be reassigned)
ACTIVE_CONFIG = DEFAULT_CONFIG