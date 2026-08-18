"""
JIT cache manager for the Metanion engine.
Manages caching of compiled functions with LRU eviction.
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from collections import OrderedDict
import threading
import time

from ..config import ACTIVE_CONFIG
from ..exceptions import CacheError


class JITCacheManager:
    """
    Manages the JIT compilation cache with LRU eviction.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize the JIT cache manager.
        
        Args:
            max_size: Maximum number of cached functions.
        """
        self.max_size = max_size or ACTIVE_CONFIG.jit_cache_size
        self._cache: OrderedDict[int, Tuple[Callable, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_compilations': 0,
        }
        self._hit_times: List[float] = []
        self._miss_times: List[float] = []
    
    def get(self, handle: int) -> Optional[Callable]:
        """
        Get a cached function by handle.
        
        Args:
            handle: The expression handle.
            
        Returns:
            The cached function, or None if not found.
        """
        with self._lock:
            if handle in self._cache:
                # Move to end (most recently used)
                func, timestamp = self._cache.pop(handle)
                self._cache[handle] = (func, time.time())
                self._stats['hits'] += 1
                return func
            
            self._stats['misses'] += 1
            return None
    
    def put(self, handle: int, func: Callable) -> None:
        """
        Cache a compiled function.
        
        Args:
            handle: The expression handle.
            func: The compiled function.
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[handle] = (func, time.time())
            self._stats['total_compilations'] += 1
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if self._cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            self._stats['evictions'] += 1
    
    def invalidate(self, handle: int) -> None:
        """
        Invalidate a cached function.
        
        Args:
            handle: The expression handle.
        """
        with self._lock:
            if handle in self._cache:
                del self._cache[handle]
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._stats['hits'] = 0
            self._stats['misses'] = 0
            self._stats['evictions'] = 0
            self._stats['total_compilations'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_ratio': self._stats['hits'] / (total_requests + 1),
                'evictions': self._stats['evictions'],
                'total_compilations': self._stats['total_compilations'],
                'usage_percent': (len(self._cache) / self.max_size) * 100,
            }
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("JIT Cache Statistics")
        print("=" * 50)
        print(f"Cache Size:         {stats['cache_size']}/{stats['max_size']}")
        print(f"Usage:              {stats['usage_percent']:.1f}%")
        print(f"Hits:               {stats['hits']}")
        print(f"Misses:             {stats['misses']}")
        print(f"Hit Ratio:          {stats['hit_ratio']:.2%}")
        print(f"Evictions:          {stats['evictions']}")
        print(f"Total Compilations: {stats['total_compilations']}")
        print("=" * 50)


# Global JIT cache manager
_JIT_CACHE: Optional[JITCacheManager] = None


def get_jit_cache() -> JITCacheManager:
    """Get or create the global JIT cache manager."""
    global _JIT_CACHE
    if _JIT_CACHE is None:
        _JIT_CACHE = JITCacheManager()
    return _JIT_CACHE