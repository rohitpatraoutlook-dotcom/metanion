"""
Complete Metanion model with training and evaluation capabilities.
"""

from typing import Optional, List, Tuple, Dict, Any, Union, Callable
import time
import numpy as np
from dataclasses import dataclass, field

from ..core import Tensor, DType, Shape
from ..gp import (
    GPIndividual, PopulationManager, FitnessEvaluator,
    BloatController, BloatControl, ParetoBloatControl
)
from ..gp import InitializationMethod, PopulationInitializer
from ..compile import compile_handle
from ..symbolic import simplify, intern, OpID
from .metanion_stack import MetanionStack, StackConfig
from .metanion_layer import MetanionLayer
from ..exceptions import ModelError, TrainingError


@dataclass
class ModelConfig:
    """Configuration for the Metanion model."""
    
    # Architecture
    layer_sizes: List[int] = field(default_factory=lambda: [1, 10, 1])
    use_bias: bool = True
    max_depth: int = 5
    
    # Training
    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism_count: int = 5
    
    # Regularization
    lambda_depth: float = 0.01
    lambda_time: float = 0.001
    max_time_ms: float = 10.0
    
    # Data
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Randomness
    random_seed: Optional[int] = 42
    
    # Hardware
    use_numba: bool = False
    use_multiprocessing: bool = True
    num_workers: int = 4
    
    # Logging
    verbose: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 25


class MetanionModel:
    """
    Complete Metanion model with training and evaluation.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the Metanion model.
        
        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        
        # Set random seed
        if self.config.random_seed is not None:
            import random
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Build model
        self.stack = MetanionStack(
            layer_sizes=self.config.layer_sizes,
            use_bias=self.config.use_bias,
            max_depth=self.config.max_depth,
            dtype=DType.FLOAT64
        )
        
        # Training state
        self._is_trained = False
        self._training_history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_depth': [],
            'avg_depth': [],
        }
        self._best_individual = None
        
        # Fitness evaluator
        self._evaluator = None
    
    def fit(
        self,
        X: Union[Tensor, np.ndarray, List],
        y: Union[Tensor, np.ndarray, List],
        X_val: Optional[Union[Tensor, np.ndarray, List]] = None,
        y_val: Optional[Union[Tensor, np.ndarray, List]] = None,
        epochs: Optional[int] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            X: Training input data.
            y: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
            epochs: Number of generations to train.
            verbose: Whether to print progress.
            
        Returns:
            Training history.
        """
        # Convert data to Tensor
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = self._to_tensor(X_val)
            y_val_tensor = self._to_tensor(y_val)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        # Set epochs
        epochs = epochs or self.config.generations
        verbose = verbose if verbose is not None else self.config.verbose
        
        # Initialize fitness evaluator
        self._evaluator = FitnessEvaluator(
            X_tensor, y_tensor,
            X_val_tensor, y_val_tensor,
            lambda_depth=self.config.lambda_depth,
            lambda_time=self.config.lambda_time,
            max_time_ms=self.config.max_time_ms,
            use_validation=X_val_tensor is not None
        )
        
        # Initialize population from model architecture
        population = self._create_population()
        
        # Initialize population manager
        manager = PopulationManager(
            population=population,
            population_size=self.config.population_size,
            evaluator=self._evaluator,
            elitism_count=self.config.elitism_count,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            tournament_size=self.config.tournament_size,
            max_depth=self.config.max_depth
        )
        
        # Bloat control
        bloat_config = BloatControl(
            max_depth=self.config.max_depth * 2,
            max_nodes=100,
            parsimony_pressure=self.config.lambda_depth,
            use_age_layers=True,
            use_parsimony=True
        )
        bloat_controller = BloatController(bloat_config)
        
        # Training loop
        start_time = time.time()
        
        for gen in range(epochs):
            # Evolve one generation
            manager._evolve_one_generation()
            
            # Apply bloat control
            controlled_pop = bloat_controller.apply_control(manager.population)
            manager.population = controlled_pop
            
            # Update history
            stats = manager.get_stats()
            self._training_history['generations'].append(gen)
            self._training_history['best_fitness'].append(stats['best_fitness'])
            self._training_history['avg_fitness'].append(stats['avg_fitness'])
            self._training_history['best_depth'].append(stats['best_depth'])
            self._training_history['avg_depth'].append(stats['avg_depth'])
            
            # Log progress
            if verbose and (gen + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Generation {gen + 1}/{epochs} | "
                      f"Best Fitness: {stats['best_fitness']:.6f} | "
                      f"Avg Fitness: {stats['avg_fitness']:.6f} | "
                      f"Best Depth: {stats['best_depth']} | "
                      f"Time: {elapsed:.2f}s")
        
        # Get best individual
        best = manager.get_best()
        if best is not None:
            self._best_individual = best
            self._update_model_from_individual(best)
            self._is_trained = True
        
        elapsed = time.time() - start_time
        
        # Final stats
        final_stats = manager.get_stats()
        final_stats['training_time'] = elapsed
        final_stats['total_generations'] = epochs
        
        if verbose:
            print(f"Training completed in {elapsed:.2f}s")
            print(f"Best fitness: {final_stats['best_fitness']:.6f}")
            print(f"Best depth: {final_stats['best_depth']}")
        
        return self._training_history
    
    def _create_population(self) -> List[GPIndividual]:
        """
        Create a population from the model architecture.
        
        Returns:
            List of GP individuals.
        """
        population = []
        
        # For each layer, create weights from expressions
        for layer in self.stack.layers:
            # Get weight handles from the layer
            handles = []
            for i in range(layer.out_features):
                for j in range(layer.in_features):
                    handles.append(layer._get_weight_handle(i, j))
            
            # Create an individual from these handles
            individual = GPIndividual(
                weight_handles=handles,
                bias_handle=layer.bias_handles[0] if layer.use_bias else None,
                shape=(layer.in_features, layer.out_features)
            )
            population.append(individual)
        
        return population
    
    def _update_model_from_individual(self, individual: GPIndividual) -> None:
        """
        Update the model from a GP individual.
        
        Args:
            individual: The GP individual.
        """
        # Update weights
        idx = 0
        for layer in self.stack.layers:
            for i in range(layer.out_features):
                for j in range(layer.in_features):
                    if idx < len(individual.weight_handles):
                        layer._set_weight_handle(i, j, individual.weight_handles[idx])
                        idx += 1
            
            if layer.use_bias and individual.bias_handle is not None:
                for i in range(layer.out_features):
                    if i < len(layer.bias_handles):
                        layer.bias_handles[i] = individual.bias_handle
    
    def predict(self, X: Union[Tensor, np.ndarray, List]) -> Tensor:
        """
        Make predictions on new data.
        
        Args:
            X: Input data.
            
        Returns:
            Predictions.
        """
        if not self._is_trained:
            raise ModelError("Model has not been trained yet")
        
        X_tensor = self._to_tensor(X)
        return self.stack.forward(X_tensor)
    
    def evaluate(self, X: Union[Tensor, np.ndarray, List], y: Union[Tensor, np.ndarray, List]) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            X: Input data.
            y: Target data.
            
        Returns:
            Mean squared error.
        """
        predictions = self.predict(X)
        y_tensor = self._to_tensor(y)
        
        # Compute MSE
        pred_np = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
        true_np = y_tensor.numpy() if hasattr(y_tensor, 'numpy') else np.array(y_tensor)
        
        return float(np.mean((pred_np - true_np) ** 2))
    
    def get_expression(self) -> str:
        """
        Get string representation of the model.
        
        Returns:
            String representation.
        """
        parts = []
        for i, layer in enumerate(self.stack.layers):
            parts.append(f"Layer {i}: {layer}")
        return "\n".join(parts)
    
    def simplify(self) -> None:
        """Simplify all expressions in the model."""
        self.stack.simplify()
        if self._best_individual is not None:
            self._best_individual.simplify()
    
    def _to_tensor(self, data: Union[Tensor, np.ndarray, List]) -> Tensor:
        """Convert data to Tensor."""
        if isinstance(data, Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return Tensor(data.tolist())
        elif isinstance(data, list):
            return Tensor(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'is_trained': self._is_trained,
            'num_layers': self.stack.num_layers,
            'total_depth': self.stack.get_depth(),
            'total_nodes': self.stack.get_node_count(),
            'training_generations': len(self._training_history['generations']),
            'best_fitness': self._training_history['best_fitness'][-1] if self._training_history['best_fitness'] else None,
            'model_size': sum(layer.get_node_count() for layer in self.stack.layers),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MetanionModel(layers={self.stack.num_layers}, "
                f"trained={self._is_trained}, "
                f"depth={self.stack.get_depth()}, "
                f"nodes={self.stack.get_node_count()})")