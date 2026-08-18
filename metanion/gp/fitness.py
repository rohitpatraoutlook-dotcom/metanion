"""
Fitness evaluation for genetic programming.
Evaluates individuals on training data with multiple objectives.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
import time
import math

from .individual import GPIndividual
from ..core import Tensor, DType
from ..compile import compile_handle
from ..exceptions import EvaluationError


class FitnessEvaluator:
    """
    Evaluates fitness of GP individuals.
    Supports multiple objectives: accuracy, complexity, time.
    """
    
    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Optional[Tensor] = None,
        y_val: Optional[Tensor] = None,
        lambda_depth: float = 0.01,
        lambda_time: float = 0.001,
        max_time_ms: float = 10.0,
        use_validation: bool = True,
        minimize: bool = True
    ):
        """
        Initialize the fitness evaluator.
        
        Args:
            X_train: Training input data.
            y_train: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
            lambda_depth: Penalty per node depth.
            lambda_time: Penalty per millisecond.
            max_time_ms: Maximum allowed inference time.
            use_validation: Use validation data if available.
            minimize: True if minimizing fitness.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.lambda_depth = lambda_depth
        self.lambda_time = lambda_time
        self.max_time_ms = max_time_ms
        self.use_validation = use_validation and X_val is not None and y_val is not None
        self.minimize = minimize
        
        # Cache for evaluated individuals
        self._cache: Dict[int, float] = {}
        self._evaluation_count = 0
    
    def evaluate(self, individual: GPIndividual) -> float:
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Fitness value (lower is better if minimize=True).
        """
        # Check cache
        if individual.id in self._cache:
            return self._cache[individual.id]
        
        self._evaluation_count += 1
        
        try:
            # Measure prediction error
            mse = self._compute_mse(individual)
            
            # Measure complexity
            depth_penalty = self.lambda_depth * individual.depth
            node_penalty = self.lambda_depth * 0.1 * individual.node_count
            
            # Measure time
            time_ms = self._measure_time(individual)
            time_penalty = self.lambda_time * time_ms
            
            # Time penalty for exceeding limit
            if time_ms > self.max_time_ms:
                time_penalty += (time_ms - self.max_time_ms) * 10.0
            
            # Combine into fitness
            fitness = mse + depth_penalty + node_penalty + time_penalty
            
            # Store fitness
            individual.fitness = fitness
            individual.inference_time = time_ms
            
            # Cache the result
            self._cache[individual.id] = fitness
            
            return fitness
            
        except Exception as e:
            # If evaluation fails, return worst fitness
            individual.fitness = float('inf')
            self._cache[individual.id] = float('inf')
            return float('inf')
    
    def _compute_mse(self, individual: GPIndividual) -> float:
        """
        Compute mean squared error on the training/validation data.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Mean squared error.
        """
        # Choose data
        if self.use_validation:
            X = self.X_val
            y = self.y_val
        else:
            X = self.X_train
            y = self.y_train
        
        # Get the compiled function
        try:
            func = individual.compile()
        except Exception:
            return float('inf')
        
        # Evaluate on data
        try:
            # Convert to numpy for evaluation
            X_np = X.numpy() if hasattr(X, 'numpy') else X
            y_np = y.numpy() if hasattr(y, 'numpy') else y
            
            # Flatten if needed
            if len(X_np.shape) > 2:
                X_np = X_np.reshape(X_np.shape[0], -1)
            if len(y_np.shape) > 1:
                y_np = y_np.flatten()
            
            # Predict
            y_pred = []
            for i in range(X_np.shape[0]):
                try:
                    pred = func([X_np[i, 0]])
                    y_pred.append(pred)
                except Exception:
                    y_pred.append(0.0)
            
            y_pred = np.array(y_pred)
            
            # Compute MSE
            mse = np.mean((y_np[:len(y_pred)] - y_pred) ** 2)
            
            # Check for NaN or Inf
            if np.isnan(mse) or np.isinf(mse):
                return float('inf')
            
            return float(mse)
            
        except Exception:
            return float('inf')
    
    def _measure_time(self, individual: GPIndividual) -> float:
        """
        Measure inference time of the individual.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Inference time in milliseconds.
        """
        try:
            func = individual.compile()
            
            # Warm up
            X_np = self.X_train.numpy() if hasattr(self.X_train, 'numpy') else self.X_train
            if len(X_np.shape) > 2:
                X_np = X_np.reshape(X_np.shape[0], -1)
            
            # Measure time
            start = time.perf_counter()
            for i in range(min(100, X_np.shape[0])):
                try:
                    func([X_np[i, 0]])
                except Exception:
                    pass
            end = time.perf_counter()
            
            # Return time in milliseconds
            time_ms = (end - start) * 1000.0 / min(100, X_np.shape[0])
            
            return time_ms
            
        except Exception:
            return float('inf')
    
    def evaluate_population(self, population: List[GPIndividual]) -> None:
        """
        Evaluate all individuals in a population.
        
        Args:
            population: List of individuals to evaluate.
        """
        for individual in population:
            self.evaluate(individual)
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self._evaluation_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            'evaluation_count': self._evaluation_count,
            'cache_size': len(self._cache),
            'use_validation': self.use_validation,
        }


class MultiObjectiveFitness(FitnessEvaluator):
    """
    Multi-objective fitness evaluator.
    Returns multiple fitness values for Pareto optimization.
    """
    
    def evaluate(self, individual: GPIndividual) -> Tuple[float, float, float]:
        """
        Evaluate individual on multiple objectives.
        
        Returns:
            Tuple of (mse, complexity, time).
        """
        # Compute MSE
        mse = self._compute_mse(individual)
        
        # Compute complexity
        complexity = individual.depth + 0.1 * individual.node_count
        
        # Compute time
        time_ms = self._measure_time(individual)
        
        return mse, complexity, time_ms
    
    def pareto_dominates(
        self,
        ind1: GPIndividual,
        ind2: GPIndividual
    ) -> bool:
        """
        Check if ind1 Pareto dominates ind2.
        
        Args:
            ind1: First individual.
            ind2: Second individual.
            
        Returns:
            True if ind1 dominates ind2.
        """
        # Get objectives
        obj1 = self.evaluate(ind1)
        obj2 = self.evaluate(ind2)
        
        # Check if ind1 is better in all objectives
        better = False
        for o1, o2 in zip(obj1, obj2):
            if self.minimize:
                if o1 > o2:
                    return False
                if o1 < o2:
                    better = True
            else:
                if o1 < o2:
                    return False
                if o1 > o2:
                    better = True
        
        return better


class ParetoFitness:
    """
    Pareto-based fitness evaluation for multi-objective optimization.
    """
    
    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Optional[Tensor] = None,
        y_val: Optional[Tensor] = None,
        lambda_depth: float = 0.01,
        lambda_time: float = 0.001,
        use_validation: bool = True
    ):
        """
        Initialize Pareto fitness evaluator.
        
        Args:
            X_train: Training input data.
            y_train: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
            lambda_depth: Penalty per node depth.
            lambda_time: Penalty per millisecond.
            use_validation: Use validation data if available.
        """
        self.fitness = MultiObjectiveFitness(
            X_train, y_train, X_val, y_val,
            lambda_depth, lambda_time, 10.0, use_validation
        )
        self._frontier: List[int] = []
        self._rank_cache: Dict[int, int] = {}
    
    def evaluate(self, individual: GPIndividual) -> float:
        """
        Evaluate individual using Pareto ranking.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Pareto rank (lower is better).
        """
        # First compute objectives
        objectives = self.fitness.evaluate(individual)
        
        # Check if any objective is inf
        if any(math.isinf(o) for o in objectives):
            return float('inf')
        
        # Compute Pareto rank
        rank = self._compute_pareto_rank(individual)
        individual.fitness = rank
        
        return rank
    
    def _compute_pareto_rank(self, individual: GPIndividual) -> int:
        """
        Compute Pareto rank for an individual.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Pareto rank (number of individuals that dominate it).
        """
        # Check cache
        if individual.id in self._rank_cache:
            return self._rank_cache[individual.id]
        
        # Count how many individuals dominate this one
        # This is simplified - in practice, you'd compare against the population
        rank = 0
        
        # Store rank
        self._rank_cache[individual.id] = rank
        return rank
    
    def update_frontier(self, population: List[GPIndividual]) -> None:
        """
        Update the Pareto frontier.
        
        Args:
            population: Population to find frontier from.
        """
        self._frontier = []
        
        # Find non-dominated individuals
        for i, ind1 in enumerate(population):
            dominated = False
            for j, ind2 in enumerate(population):
                if i != j:
                    if self.fitness.pareto_dominates(ind2, ind1):
                        dominated = True
                        break
            if not dominated:
                self._frontier.append(ind1.id)
    
    def get_frontier(self) -> List[int]:
        """Get the current Pareto frontier."""
        return self._frontier