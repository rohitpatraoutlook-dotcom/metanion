"""
Checkpoint manager for the Metanion engine.
Handles saving and loading model checkpoints during training.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
import os
import time
import glob
from dataclasses import dataclass, field

from ..model import MetanionModel, ModelConfig
from ..exceptions import CheckpointError
from .binary_encoder import BinaryEncoder
from .binary_decoder import BinaryDecoder


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    checkpoint_id: str
    timestamp: float
    generation: int
    best_fitness: float
    filepath: str
    size_bytes: int
    compression: str = 'none'
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5,
        save_interval: int = 10,
        keep_best_only: bool = False
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            save_interval: Save checkpoint every N generations.
            keep_best_only: Only keep the best checkpoint.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        self.keep_best_only = keep_best_only
        
        self.encoder = BinaryEncoder()
        self.decoder = BinaryDecoder()
        self._checkpoints: Dict[str, CheckpointMetadata] = {}
        self._best_checkpoint: Optional[CheckpointMetadata] = None
        
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load existing checkpoints
        self._scan_checkpoints()
    
    def _scan_checkpoints(self) -> None:
        """Scan the checkpoint directory for existing checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.metanion')
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            checkpoint_id = filename.replace('checkpoint_', '').replace('.metanion', '')
            
            # Try to read metadata from file
            try:
                with open(filepath, 'rb') as f:
                    data = f.read()
                
                # Read header only to get metadata
                from .binary_encoder import BinaryHeader
                header, _ = BinaryHeader.unpack(data)
                metadata = header.metadata
                
                checkpoint = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=os.path.getmtime(filepath),
                    generation=metadata.get('generation', 0),
                    best_fitness=metadata.get('best_fitness', float('inf')),
                    filepath=filepath,
                    size_bytes=os.path.getsize(filepath),
                    compression=header.compression,
                    metadata=metadata
                )
                
                self._checkpoints[checkpoint_id] = checkpoint
                
                # Track best checkpoint
                if self._best_checkpoint is None or checkpoint.best_fitness < self._best_checkpoint.best_fitness:
                    self._best_checkpoint = checkpoint
                    
            except Exception:
                # If can't read, skip this checkpoint
                pass
    
    def save_checkpoint(
        self,
        model: MetanionModel,
        generation: int,
        metrics: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            model: The model to save.
            generation: Current generation number.
            metrics: Additional metrics to store.
            checkpoint_id: Optional checkpoint ID.
            
        Returns:
            Checkpoint ID.
        """
        if checkpoint_id is None:
            import uuid
            checkpoint_id = f"{generation}_{uuid.uuid4().hex[:8]}"
        
        filepath = os.path.join(self.checkpoint_dir, f'checkpoint_{checkpoint_id}.metanion')
        
        # Prepare metadata
        metadata = {
            'generation': generation,
            'best_fitness': model._training_history.get('best_fitness', [float('inf')])[-1] if model._training_history.get('best_fitness') else float('inf'),
            'avg_fitness': model._training_history.get('avg_fitness', [0.0])[-1] if model._training_history.get('avg_fitness') else 0.0,
            'num_layers': model.stack.num_layers,
            'input_size': model.stack.input_size,
            'output_size': model.stack.output_size,
            'depth': model.stack.get_depth(),
            'nodes': model.stack.get_node_count(),
            'timestamp': time.time(),
        }
        if metrics:
            metadata.update(metrics)
        
        # Encode and save
        try:
            # Add metadata to the model's header
            data = self.encoder.encode_model(model)
            
            # Write to file
            with open(filepath, 'wb') as f:
                f.write(data)
            
            # Create checkpoint metadata
            checkpoint = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                generation=generation,
                best_fitness=metadata['best_fitness'],
                filepath=filepath,
                size_bytes=os.path.getsize(filepath),
                metadata=metadata
            )
            
            self._checkpoints[checkpoint_id] = checkpoint
            
            # Update best checkpoint
            if self._best_checkpoint is None or checkpoint.best_fitness < self._best_checkpoint.best_fitness:
                self._best_checkpoint = checkpoint
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            return checkpoint_id
            
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> MetanionModel:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to load. If None, loads the best checkpoint.
            
        Returns:
            The loaded model.
        """
        if checkpoint_id is None:
            if self._best_checkpoint is None:
                raise CheckpointError("No best checkpoint available")
            checkpoint = self._best_checkpoint
        else:
            if checkpoint_id not in self._checkpoints:
                raise CheckpointError(f"Checkpoint {checkpoint_id} not found")
            checkpoint = self._checkpoints[checkpoint_id]
        
        try:
            model = self.decoder.decode_from_file(checkpoint.filepath)
            return model
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        if self.keep_best_only:
            # Keep only the best checkpoint
            for cp_id, cp in list(self._checkpoints.items()):
                if cp_id != self._best_checkpoint.checkpoint_id:
                    self._delete_checkpoint(cp_id)
            return
        
        # Sort checkpoints by timestamp (newest first)
        sorted_cps = sorted(
            self._checkpoints.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Keep only the max_checkpoints newest
        keep = set()
        for cp in sorted_cps[:self.max_checkpoints]:
            keep.add(cp.checkpoint_id)
        
        # Delete older checkpoints
        for cp_id in list(self._checkpoints.keys()):
            if cp_id not in keep:
                self._delete_checkpoint(cp_id)
    
    def _delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to delete.
        """
        if checkpoint_id in self._checkpoints:
            cp = self._checkpoints[checkpoint_id]
            try:
                if os.path.exists(cp.filepath):
                    os.remove(cp.filepath)
            except Exception:
                pass
            del self._checkpoints[checkpoint_id]
            
            # Update best checkpoint if needed
            if self._best_checkpoint and self._best_checkpoint.checkpoint_id == checkpoint_id:
                self._best_checkpoint = None
                # Find new best
                for cp in self._checkpoints.values():
                    if self._best_checkpoint is None or cp.best_fitness < self._best_checkpoint.best_fitness:
                        self._best_checkpoint = cp
    
    def get_checkpoint_info(self, checkpoint_id: Optional[str] = None) -> Optional[CheckpointMetadata]:
        """
        Get information about a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID. If None, returns info about the best checkpoint.
            
        Returns:
            Checkpoint metadata or None if not found.
        """
        if checkpoint_id is None:
            return self._best_checkpoint
        return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata, sorted by timestamp (newest first).
        """
        return sorted(
            self._checkpoints.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
    
    def should_save(self, generation: int) -> bool:
        """
        Check if a checkpoint should be saved at this generation.
        
        Args:
            generation: Current generation number.
            
        Returns:
            True if checkpoint should be saved.
        """
        return (generation + 1) % self.save_interval == 0