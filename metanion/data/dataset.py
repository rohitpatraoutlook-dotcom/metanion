"""
Dataset handling for the Metanion engine.
Provides flexible dataset management and preprocessing.
"""

from typing import Optional, List, Tuple, Dict, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
import numpy as np
from collections.abc import Sequence

from ..core import Tensor, DType
from ..exceptions import DataError


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    
    dtype: DType = DType.FLOAT64
    normalize: bool = False
    normalize_method: str = "standard"  # standard, minmax, robust
    shuffle: bool = True
    random_seed: Optional[int] = 42
    batch_size: int = 32
    num_workers: int = 0  # For future multiprocessing support


class Dataset(Sequence):
    """
    Flexible dataset class for Metanion.
    Handles data loading, preprocessing, and batching.
    """
    
    def __init__(
        self,
        X: Union[Tensor, np.ndarray, List],
        y: Optional[Union[Tensor, np.ndarray, List]] = None,
        config: Optional[DatasetConfig] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            X: Input data.
            y: Target data (optional).
            config: Dataset configuration.
        """
        self.config = config or DatasetConfig()
        
        # Convert to Tensor
        self.X = self._to_tensor(X)
        self.y = self._to_tensor(y) if y is not None else None
        
        # Validate shapes
        self._validate_shapes()
        
        # Store statistics for normalization
        self.X_mean = None
        self.X_std = None
        self.X_min = None
        self.X_max = None
        
        # Apply normalization if configured
        if self.config.normalize:
            self._compute_statistics()
        
        # Compute dataset size
        self._n_samples = self.X.shape[0] if len(self.X.shape) > 0 else 1
        self._input_dim = self.X.shape[1] if len(self.X.shape) > 1 else 1
        self._output_dim = self.y.shape[1] if self.y is not None and len(self.y.shape) > 1 else 1
        
        # Shuffle indices
        self._indices = list(range(self._n_samples))
        if self.config.shuffle and self.config.random_seed is not None:
            import random
            random.seed(self.config.random_seed)
            random.shuffle(self._indices)
    
    def _to_tensor(self, data: Union[Tensor, np.ndarray, List]) -> Tensor:
        """Convert data to Tensor."""
        if isinstance(data, Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return Tensor(data.tolist(), dtype=self.config.dtype)
        elif isinstance(data, list):
            return Tensor(data, dtype=self.config.dtype)
        else:
            raise DataError(f"Unsupported data type: {type(data)}")
    
    def _validate_shapes(self) -> None:
        """Validate data shapes."""
        if len(self.X.shape) < 1:
            raise DataError("Input data must have at least 1 dimension")
        
        if self.y is not None:
            if len(self.y.shape) < 1:
                raise DataError("Target data must have at least 1 dimension")
            if self.y.shape[0] != self.X.shape[0]:
                raise DataError(
                    f"Number of samples mismatch: X={self.X.shape[0]}, y={self.y.shape[0]}"
                )
    
    def _compute_statistics(self) -> None:
        """Compute normalization statistics."""
        X_np = self.X.numpy() if hasattr(self.X, 'numpy') else np.array(self.X)
        
        if self.config.normalize_method == "standard":
            self.X_mean = np.mean(X_np, axis=0)
            self.X_std = np.std(X_np, axis=0) + 1e-8
        elif self.config.normalize_method == "minmax":
            self.X_min = np.min(X_np, axis=0)
            self.X_max = np.max(X_np, axis=0) + 1e-8
        elif self.config.normalize_method == "robust":
            self.X_mean = np.median(X_np, axis=0)
            self.X_std = np.percentile(X_np, 75, axis=0) - np.percentile(X_np, 25, axis=0) + 1e-8
    
    def __len__(self) -> int:
        """Get the number of samples."""
        return self._n_samples
    
    def __getitem__(self, idx: Union[int, slice]) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Get a sample or slice of the dataset.
        
        Args:
            idx: Index or slice.
            
        Returns:
            Tuple of (input, target) or slices.
        """
        if isinstance(idx, slice):
            indices = self._indices[idx]
            X_batch = self._get_batch(indices)
            y_batch = self._get_batch_y(indices)
            return X_batch, y_batch
        else:
            idx = self._indices[idx]
            X_sample = self._get_sample(idx)
            y_sample = self._get_sample_y(idx)
            return X_sample, y_sample
    
    def _get_sample(self, idx: int) -> Tensor:
        """Get a single input sample."""
        X_np = self.X.numpy() if hasattr(self.X, 'numpy') else np.array(self.X)
        sample = X_np[idx]
        
        # Apply normalization
        if self.config.normalize and self.X_mean is not None:
            if self.config.normalize_method == "standard":
                sample = (sample - self.X_mean) / self.X_std
            elif self.config.normalize_method == "minmax":
                sample = (sample - self.X_min) / (self.X_max - self.X_min)
            elif self.config.normalize_method == "robust":
                sample = (sample - self.X_mean) / self.X_std
        
        return Tensor(sample.tolist(), dtype=self.config.dtype)
    
    def _get_batch(self, indices: List[int]) -> Tensor:
        """Get a batch of input samples."""
        X_np = self.X.numpy() if hasattr(self.X, 'numpy') else np.array(self.X)
        batch = X_np[indices]
        
        # Apply normalization
        if self.config.normalize and self.X_mean is not None:
            if self.config.normalize_method == "standard":
                batch = (batch - self.X_mean) / self.X_std
            elif self.config.normalize_method == "minmax":
                batch = (batch - self.X_min) / (self.X_max - self.X_min)
            elif self.config.normalize_method == "robust":
                batch = (batch - self.X_mean) / self.X_std
        
        return Tensor(batch.tolist(), dtype=self.config.dtype)
    
    def _get_sample_y(self, idx: int) -> Optional[Tensor]:
        """Get a single target sample."""
        if self.y is None:
            return None
        y_np = self.y.numpy() if hasattr(self.y, 'numpy') else np.array(self.y)
        return Tensor(y_np[idx].tolist(), dtype=self.config.dtype)
    
    def _get_batch_y(self, indices: List[int]) -> Optional[Tensor]:
        """Get a batch of target samples."""
        if self.y is None:
            return None
        y_np = self.y.numpy() if hasattr(self.y, 'numpy') else np.array(self.y)
        return Tensor(y_np[indices].tolist(), dtype=self.config.dtype)
    
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Get a random batch from the dataset.
        
        Args:
            batch_size: Batch size (default: config.batch_size).
            
        Returns:
            Tuple of (input_batch, target_batch).
        """
        batch_size = batch_size or self.config.batch_size
        batch_size = min(batch_size, self._n_samples)
        
        # Select random indices
        import random
        indices = random.sample(self._indices, batch_size)
        
        X_batch = self._get_batch(indices)
        y_batch = self._get_batch_y(indices)
        
        return X_batch, y_batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'n_samples': self._n_samples,
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'normalized': self.config.normalize,
            'normalize_method': self.config.normalize_method,
            'shuffled': self.config.shuffle,
        }


class DataLoader:
    """
    Data loader for iterating over datasets in batches.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset: The dataset to load.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            random_seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size or dataset.config.batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed or dataset.config.random_seed
        
        # Set up indices
        self._reset_indices()
    
    def _reset_indices(self) -> None:
        """Reset and shuffle indices."""
        self._indices = list(range(len(self.dataset)))
        if self.shuffle and self.random_seed is not None:
            import random
            random.seed(self.random_seed)
            random.shuffle(self._indices)
        self._current_idx = 0
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        """Iterate over batches."""
        self._reset_indices()
        return self
    
    def __next__(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the next batch."""
        if self._current_idx >= len(self._indices):
            raise StopIteration
        
        end_idx = min(self._current_idx + self.batch_size, len(self._indices))
        batch_indices = self._indices[self._current_idx:end_idx]
        self._current_idx = end_idx
        
        X_batch = self.dataset._get_batch(batch_indices)
        y_batch = self.dataset._get_batch_y(batch_indices)
        
        return X_batch, y_batch
    
    def __len__(self) -> int:
        """Get the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size