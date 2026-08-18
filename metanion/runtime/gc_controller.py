"""
Garbage collection controller for the Metanion engine.
Manages Python's garbage collector and memory cleanup.
"""

from typing import Optional, Dict, Any, List
import gc
import threading
import time
from dataclasses import dataclass, field

from ..config import ACTIVE_CONFIG


@dataclass
class GCStats:
    """Statistics for garbage collection."""
    
    collections: int = 0
    collected_objects: int = 0
    uncollectable_objects: int = 0
    collection_time_ms: float = 0.0
    memory_freed_bytes: int = 0
    generation_counts: List[int] = field(default_factory=list)


class GCController:
    """
    Controls garbage collection for the Metanion engine.
    """
    
    def __init__(self):
        """Initialize the GC controller."""
        self._gc_enabled = True
        self._auto_collect = True
        self._collect_threshold = 1000  # Number of allocations before collection
        self._allocation_count = 0
        self._last_collection_time = 0.0
        self._stats = GCStats()
        self._lock = threading.RLock()
        
        # Store GC thresholds
        self._original_thresholds = gc.get_threshold()
        
        # Set optimized thresholds
        gc.set_threshold(700, 10, 10)
        
        # Disable GC during critical operations if needed
        self._gc_disabled = False
    
    def enable_gc(self) -> None:
        """Enable garbage collection."""
        self._gc_enabled = True
        if not gc.isenabled():
            gc.enable()
    
    def disable_gc(self) -> None:
        """Disable garbage collection."""
        self._gc_enabled = False
        if gc.isenabled():
            gc.disable()
    
    def collect(self, generation: int = 2) -> GCStats:
        """
        Run garbage collection.
        
        Args:
            generation: Generation to collect (0, 1, 2).
            
        Returns:
            GC statistics.
        """
        if not self._gc_enabled:
            return self._stats
        
        start_time = time.perf_counter_ns()
        
        # Collect garbage
        collected = gc.collect(generation)
        uncollectable = len(gc.garbage)
        
        # Clean up uncollectable objects
        if uncollectable > 0:
            for obj in gc.garbage:
                try:
                    # Attempt to break reference cycles
                    if hasattr(obj, '__del__'):
                        obj.__del__ = lambda: None
                except:
                    pass
            gc.garbage.clear()
        
        end_time = time.perf_counter_ns()
        
        with self._lock:
            self._stats.collections += 1
            self._stats.collected_objects += collected
            self._stats.uncollectable_objects += uncollectable
            self._stats.collection_time_ms += (end_time - start_time) / 1_000_000.0
            self._stats.generation_counts = gc.get_count()
        
        self._last_collection_time = time.time()
        
        return self._stats
    
    def auto_collect(self, force: bool = False) -> None:
        """
        Perform automatic collection if needed.
        
        Args:
            force: Force collection regardless of threshold.
        """
        if not self._auto_collect:
            return
        
        with self._lock:
            self._allocation_count += 1
        
        if force or self._allocation_count >= self._collect_threshold:
            with self._lock:
                self._allocation_count = 0
            self.collect()
    
    def set_threshold(self, threshold: int) -> None:
        """
        Set the collection threshold.
        
        Args:
            threshold: Number of allocations before collection.
        """
        self._collect_threshold = threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GC statistics."""
        with self._lock:
            return {
                'collections': self._stats.collections,
                'collected_objects': self._stats.collected_objects,
                'uncollectable_objects': self._stats.uncollectable_objects,
                'collection_time_ms': self._stats.collection_time_ms,
                'memory_freed_bytes': self._stats.memory_freed_bytes,
                'generation_counts': self._stats.generation_counts,
                'gc_enabled': self._gc_enabled,
                'auto_collect': self._auto_collect,
                'threshold': self._collect_threshold,
                'allocation_count': self._allocation_count,
            }
    
    def print_stats(self) -> None:
        """Print GC statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("Garbage Collection Statistics")
        print("=" * 50)
        print(f"Collections:            {stats['collections']}")
        print(f"Collected Objects:      {stats['collected_objects']}")
        print(f"Uncollectable Objects:  {stats['uncollectable_objects']}")
        print(f"Collection Time:        {stats['collection_time_ms']:.2f} ms")
        print(f"GC Enabled:             {stats['gc_enabled']}")
        print(f"Auto Collect:           {stats['auto_collect']}")
        print(f"Threshold:              {stats['threshold']}")
        print(f"Allocations:            {stats['allocation_count']}")
        print(f"Gen Counts:             {stats['generation_counts']}")
        print("=" * 50)
    
    def reset_stats(self) -> None:
        """Reset GC statistics."""
        with self._lock:
            self._stats = GCStats()
            self._allocation_count = 0
    
    def __enter__(self):
        """Context manager entry - disable GC for critical section."""
        self.disable_gc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - re-enable GC."""
        self.enable_gc()
        # Run collection after critical section
        self.collect()


# Global GC controller
_GC_CONTROLLER: Optional[GCController] = None


def get_gc_controller() -> GCController:
    """Get or create the global GC controller."""
    global _GC_CONTROLLER
    if _GC_CONTROLLER is None:
        _GC_CONTROLLER = GCController()
    return _GC_CONTROLLER


def collect_garbage(generation: int = 2) -> None:
    """Convenience function to collect garbage."""
    get_gc_controller().collect(generation)


def suppress_gc() -> GCController:
    """Context manager to suppress garbage collection."""
    return get_gc_controller()