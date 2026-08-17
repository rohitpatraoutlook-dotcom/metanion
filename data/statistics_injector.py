"""
Statistics injection for data preprocessing.
Injects dataset statistics as frozen handles in the expression pool.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np

from ..core import Tensor
from ..symbolic import OpID, intern, get_pool
from ..exceptions import DataError


class StatisticsInjector:
    """
    Injects dataset statistics as frozen handles in the expression pool.
    Makes statistics available as constants for symbolic expressions.
    """
    
    def __init__(self, dataset):
        """
        Initialize the statistics injector.
        
        Args:
            dataset: The dataset to compute statistics from.
        """
        self.dataset = dataset
        self._stat_handles: Dict[str, int] = {}
        self._injected = False
        
        # Get data as numpy for statistics
        self.X_np = self._to_numpy(dataset.X)
        self.y_np = self._to_numpy(dataset.y) if dataset.y is not None else None
    
    def _to_numpy(self, data: Union[Tensor, np.ndarray, List]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, Tensor):
            return data.numpy() if hasattr(data, 'numpy') else np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise DataError(f"Unsupported data type: {type(data)}")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary of statistics.
        """
        stats = {}
        
        # Input statistics
        if len(self.X_np.shape) >= 1:
            stats['X_mean'] = np.mean(self.X_np, axis=0)
            stats['X_std'] = np.std(self.X_np, axis=0)
            stats['X_min'] = np.min(self.X_np, axis=0)
            stats['X_max'] = np.max(self.X_np, axis=0)
            stats['X_median'] = np.median(self.X_np, axis=0)
            stats['X_q25'] = np.percentile(self.X_np, 25, axis=0)
            stats['X_q75'] = np.percentile(self.X_np, 75, axis=0)
        
        # Target statistics
        if self.y_np is not None and len(self.y_np.shape) >= 1:
            stats['y_mean'] = np.mean(self.y_np, axis=0)
            stats['y_std'] = np.std(self.y_np, axis=0)
            stats['y_min'] = np.min(self.y_np, axis=0)
            stats['y_max'] = np.max(self.y_np, axis=0)
        
        # Dataset size
        stats['n_samples'] = self.X_np.shape[0]
        stats['input_dim'] = self.X_np.shape[1] if len(self.X_np.shape) > 1 else 1
        stats['output_dim'] = self.y_np.shape[1] if self.y_np is not None and len(self.y_np.shape) > 1 else 1
        
        return stats
    
    def inject_statistics(self) -> Dict[str, int]:
        """
        Inject statistics into the expression pool.
        
        Returns:
            Dictionary mapping statistic names to handles.
        """
        if self._injected:
            return self._stat_handles
        
        stats = self.compute_statistics()
        
        for name, value in stats.items():
            if isinstance(value, np.ndarray):
                # For arrays, convert to list of values
                if value.size == 0:
                    continue
                # Store each value separately
                for i, val in enumerate(value.flatten()):
                    handle = self._inject_constant(float(val))
                    self._stat_handles[f"{name}_{i}"] = handle
            elif isinstance(value, (int, float)):
                handle = self._inject_constant(float(value))
                self._stat_handles[name] = handle
        
        self._injected = True
        return self._stat_handles
    
    def _inject_constant(self, value: float) -> int:
        """
        Inject a constant value into the expression pool.
        
        Args:
            value: The value to inject.
            
        Returns:
            Handle of the injected constant.
        """
        # For simple values, try to represent as operations
        # e^x / x pattern to create constants
        pool = get_pool()
        
        # For common constants, use direct handles
        if value == 0.0:
            return intern(OpID.CONST_ZERO)
        elif value == 1.0:
            return intern(OpID.CONST_ONE)
        
        # For other values, create a more complex expression
        # We'll use a simple approximation: value + epsilon
        # This is a placeholder - in practice, you'd want better constant generation
        return intern(OpID.CONST_ZERO)  # Placeholder
    
    def get_stat_handle(self, stat_name: str) -> Optional[int]:
        """
        Get the handle for a statistic.
        
        Args:
            stat_name: The name of the statistic.
            
        Returns:
            Handle of the statistic, or None if not found.
        """
        if not self._injected:
            self.inject_statistics()
        
        return self._stat_handles.get(stat_name)
    
    def get_all_stat_handles(self) -> Dict[str, int]:
        """
        Get all statistic handles.
        
        Returns:
            Dictionary mapping statistic names to handles.
        """
        if not self._injected:
            self.inject_statistics()
        
        return self._stat_handles.copy()
    
    def get_normalization_handles(self) -> Dict[str, int]:
        """
        Get handles for normalization statistics.
        
        Returns:
            Dictionary of normalization handles.
        """
        if not self._injected:
            self.inject_statistics()
        
        return {
            'mean': self._stat_handles.get('X_mean_0'),
            'std': self._stat_handles.get('X_std_0'),
            'min': self._stat_handles.get('X_min_0'),
            'max': self._stat_handles.get('X_max_0'),
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Returns:
            Dictionary of dataset information.
        """
        if not self._injected:
            self.inject_statistics()
        
        return {
            'n_samples': self._stat_handles.get('n_samples'),
            'input_dim': self._stat_handles.get('input_dim'),
            'output_dim': self._stat_handles.get('output_dim'),
            'has_target': self.y_np is not None,
        }