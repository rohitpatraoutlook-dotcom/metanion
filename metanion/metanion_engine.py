"""
Main Metanion engine entry point.
Provides a unified interface for the entire framework.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import random
import numpy as np
import time

from .config import MetanionConfig, ACTIVE_CONFIG, DEFAULT_CONFIG
from .core import Tensor, DType
from .symbolic import OpID, intern, get_pool, reset_pool, simplify
from .symbolic import get_all_operation_ids, get_op_name
from .compile import compile_handle, get_compiler
from .algebra import get_rewrite_system
from .calculus import get_differentiator, differentiate
from .model import MetanionModel, ModelConfig
from .data import Dataset, DataLoader, StatisticsInjector
from .utils import TreePrinter, get_cost_model, get_time_profiler
from .io import BinaryEncoder, BinaryDecoder, CheckpointManager
from .runtime import get_jit_cache, get_gc_controller
from .exceptions import EngineError


class MetanionEngine:
    """
    Main entry point for the Metanion engine.
    Provides a unified API for all functionality.
    """
    
    def __init__(self, config: Optional[MetanionConfig] = None):
        """
        Initialize the Metanion engine.
        
        Args:
            config: Engine configuration.
        """
        self.config = config or DEFAULT_CONFIG
        
        # Set random seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Initialize components
        self._model: Optional[MetanionModel] = None
        self._tree_printer = TreePrinter()
        self._checkpoint_manager: Optional[CheckpointManager] = None
        
        # Statistics
        self._stats = {
            'start_time': time.time(),
            'models_trained': 0,
            'predictions_made': 0,
            'expressions_simplified': 0,
            'compilations': 0,
        }
    
    def create_model(
        self,
        layer_sizes: List[int],
        use_bias: bool = True,
        max_depth: int = 5
    ) -> MetanionModel:
        """
        Create a new Metanion model.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output].
            use_bias: Whether to use bias in all layers.
            max_depth: Maximum depth of expressions.
            
        Returns:
            A new MetanionModel instance.
        """
        config = ModelConfig(
            layer_sizes=layer_sizes,
            use_bias=use_bias,
            max_depth=max_depth,
            population_size=self.config.population_size,
            generations=self.config.generations,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            tournament_size=self.config.tournament_size,
            elitism_count=self.config.elitism_count,
            lambda_depth=self.config.lambda_depth,
            lambda_time=self.config.lambda_time,
            max_time_ms=self.config.max_inference_time_ms,
            batch_size=self.config.batch_size,
            random_seed=self.config.random_seed,
            verbose=self.config.verbose,
            log_interval=self.config.log_interval,
            checkpoint_interval=self.config.checkpoint_interval
        )
        
        self._model = MetanionModel(config)
        self._stats['models_trained'] += 1
        
        return self._model
    
    def train(
        self,
        X: Union[Tensor, np.ndarray, List],
        y: Union[Tensor, np.ndarray, List],
        X_val: Optional[Union[Tensor, np.ndarray, List]] = None,
        y_val: Optional[Union[Tensor, np.ndarray, List]] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model on data.
        
        Args:
            X: Training input data.
            y: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
            epochs: Number of training epochs.
            
        Returns:
            Training history.
        """
        if self._model is None:
            raise EngineError("No model created. Call create_model() first.")
        
        return self._model.fit(X, y, X_val, y_val, epochs)
    
    def predict(self, X: Union[Tensor, np.ndarray, List]) -> Tensor:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data.
            
        Returns:
            Predictions as Tensor.
        """
        if self._model is None:
            raise EngineError("No model available. Train or load a model first.")
        
        self._stats['predictions_made'] += 1
        return self._model.predict(X)
    
    def evaluate(self, X: Union[Tensor, np.ndarray, List], y: Union[Tensor, np.ndarray, List]) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            X: Input data.
            y: Target data.
            
        Returns:
            Mean squared error.
        """
        if self._model is None:
            raise EngineError("No model available. Train or load a model first.")
        
        return self._model.evaluate(X, y)
    
    def simplify_expression(self, handle: int) -> int:
        """
        Simplify an expression.
        
        Args:
            handle: The expression handle.
            
        Returns:
            Simplified handle.
        """
        result = simplify(handle)
        self._stats['expressions_simplified'] += 1
        return result
    
    def differentiate_expression(self, handle: int, variable_handle: int = -1) -> int:
        """
        Differentiate an expression.
        
        Args:
            handle: The expression handle.
            variable_handle: The variable handle.
            
        Returns:
            Derivative handle.
        """
        return differentiate(handle, variable_handle)
    
    def compile_expression(self, handle: int) -> Callable:
        """
        Compile an expression to a callable function.
        
        Args:
            handle: The expression handle.
            
        Returns:
            Compiled function.
        """
        self._stats['compilations'] += 1
        return compile_handle(handle)
    
    def print_expression(self, handle: int, format: str = "lisp", var_name: str = "x") -> str:
        """
        Print an expression in a readable format.
        
        Args:
            handle: The expression handle.
            format: Output format (text, lisp, latex, json, dot).
            var_name: Name of the input variable.
            
        Returns:
            String representation.
        """
        return self._tree_printer.print_tree(handle, format, var_name)
    
    def create_dataset(
        self,
        X: Union[Tensor, np.ndarray, List],
        y: Optional[Union[Tensor, np.ndarray, List]] = None,
        normalize: bool = False,
        shuffle: bool = True
    ) -> Dataset:
        """
        Create a dataset.
        
        Args:
            X: Input data.
            y: Target data (optional).
            normalize: Whether to normalize the data.
            shuffle: Whether to shuffle the data.
            
        Returns:
            Dataset instance.
        """
        from .data import DatasetConfig
        config = DatasetConfig(
            dtype=DType.FLOAT64,
            normalize=normalize,
            shuffle=shuffle,
            random_seed=self.config.random_seed,
            batch_size=self.config.batch_size
        )
        return Dataset(X, y, config)
    
    def save_model(self, filepath: str, checkpoint_id: Optional[str] = None) -> str:
        """
        Save the current model to a file.
        
        Args:
            filepath: Path to save the model.
            checkpoint_id: Optional checkpoint ID.
            
        Returns:
            Checkpoint ID.
        """
        if self._model is None:
            raise EngineError("No model to save.")
        
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(
                checkpoint_dir=os.path.dirname(filepath) or 'checkpoints',
                max_checkpoints=5,
                save_interval=self.config.checkpoint_interval
            )
        
        return self._checkpoint_manager.save_checkpoint(
            self._model,
            self._model._training_history.get('generations', [None])[-1] if self._model._training_history.get('generations') else 0,
            checkpoint_id=checkpoint_id
        )
    
    def load_model(self, filepath: str) -> MetanionModel:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file.
            
        Returns:
            Loaded model.
        """
        decoder = BinaryDecoder()
        self._model = decoder.decode_from_file(filepath)
        return self._model
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dictionary of statistics.
        """
        uptime = time.time() - self._stats['start_time']
        
        stats = {
            'uptime_seconds': uptime,
            'models_trained': self._stats['models_trained'],
            'predictions_made': self._stats['predictions_made'],
            'expressions_simplified': self._stats['expressions_simplified'],
            'compilations': self._stats['compilations'],
        }
        
        # Add component stats
        if self._model is not None:
            stats['model_stats'] = self._model.get_stats()
        
        return stats
    
    def print_stats(self) -> None:
        """Print engine statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("Metanion Engine Statistics")
        print("=" * 50)
        print(f"Uptime:                 {stats['uptime_seconds']:.2f}s")
        print(f"Models Trained:         {stats['models_trained']}")
        print(f"Predictions Made:       {stats['predictions_made']}")
        print(f"Expressions Simplified: {stats['expressions_simplified']}")
        print(f"Compilations:           {stats['compilations']}")
        if 'model_stats' in stats:
            print("-" * 50)
            print("Model Statistics:")
            for key, value in stats['model_stats'].items():
                print(f"  {key}: {value}")
        print("=" * 50)
    
    def reset(self) -> None:
        """Reset the engine."""
        reset_pool()
        get_compiler().clear_cache()
        get_jit_cache().clear()
        get_gc_controller().reset_stats()
        self._model = None
        self._stats = {
            'start_time': time.time(),
            'models_trained': 0,
            'predictions_made': 0,
            'expressions_simplified': 0,
            'compilations': 0,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MetanionEngine(config={self.config})"


# Global engine instance
_ENGINE: Optional[MetanionEngine] = None


def get_engine() -> MetanionEngine:
    """Get or create the global engine instance."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MetanionEngine()
    return _ENGINE


# Convenience functions
def create_model(layer_sizes: List[int], **kwargs) -> MetanionModel:
    """Create a new model using the global engine."""
    return get_engine().create_model(layer_sizes, **kwargs)


def train(X, y, X_val=None, y_val=None, epochs=None):
    """Train using the global engine."""
    return get_engine().train(X, y, X_val, y_val, epochs)


def predict(X):
    """Predict using the global engine."""
    return get_engine().predict(X)