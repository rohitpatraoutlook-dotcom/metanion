"""
Time profiling utilities for the Metanion engine.
Measures actual execution time with high precision.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
import time
import statistics
from dataclasses import dataclass, field
import threading


@dataclass
class TimeProfile:
    """Results of a time profiling measurement."""
    
    mean_ns: float = 0.0
    std_ns: float = 0.0
    min_ns: float = 0.0
    max_ns: float = 0.0
    median_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0
    n_samples: int = 0
    total_time_ns: float = 0.0
    
    def get_mean_ms(self) -> float:
        """Get mean time in milliseconds."""
        return self.mean_ns / 1_000_000.0
    
    def get_mean_us(self) -> float:
        """Get mean time in microseconds."""
        return self.mean_ns / 1000.0
    
    def get_total_ms(self) -> float:
        """Get total time in milliseconds."""
        return self.total_time_ns / 1_000_000.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TimeProfile(mean={self.get_mean_ms():.3f}ms, "
                f"std={self.std_ns / 1000:.3f}us, "
                f"n={self.n_samples}, "
                f"total={self.get_total_ms():.3f}ms)")


class TimeProfiler:
    """
    High-precision time profiler for measuring execution time.
    """
    
    def __init__(self, warmup_iterations: int = 10, max_iterations: int = 1000):
        """
        Initialize the time profiler.
        
        Args:
            warmup_iterations: Number of warmup iterations.
            max_iterations: Maximum number of measurement iterations.
        """
        self.warmup_iterations = warmup_iterations
        self.max_iterations = max_iterations
        self._cache: Dict[str, TimeProfile] = {}
        self._stats = {
            'measurements': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def profile(
        self,
        func: Callable,
        *args,
        name: Optional[str] = None,
        iterations: Optional[int] = None,
        warmup: Optional[int] = None,
        **kwargs
    ) -> TimeProfile:
        """
        Profile a function's execution time.
        
        Args:
            func: The function to profile.
            *args: Arguments to pass to the function.
            name: Name for caching (optional).
            iterations: Number of measurement iterations.
            warmup: Number of warmup iterations.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            TimeProfile with measurement results.
        """
        # Generate cache key
        cache_key = name or f"{func.__name__}_{id(func)}"
        
        # Check cache
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self._stats['cache_misses'] += 1
        
        # Set iterations
        iterations = iterations or self.max_iterations
        warmup = warmup or self.warmup_iterations
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Measure time
        times: List[float] = []
        for _ in range(min(iterations, self.max_iterations)):
            start = time.perf_counter_ns()
            func(*args, **kwargs)
            end = time.perf_counter_ns()
            times.append(float(end - start))
        
        # Compute statistics
        profile = TimeProfile(
            mean_ns=statistics.mean(times),
            std_ns=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ns=min(times),
            max_ns=max(times),
            median_ns=statistics.median(times),
            p95_ns=statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            p99_ns=statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
            n_samples=len(times),
            total_time_ns=sum(times),
        )
        
        # Cache the result
        self._cache[cache_key] = profile
        self._stats['measurements'] += 1
        
        return profile
    
    def profile_expression(
        self,
        handle: int,
        X_data: List[float],
        iterations: Optional[int] = None
    ) -> TimeProfile:
        """
        Profile a symbolic expression's execution time.
        
        Args:
            handle: The expression handle.
            X_data: Input data for evaluation.
            iterations: Number of measurement iterations.
            
        Returns:
            TimeProfile with measurement results.
        """
        from ..compile import compile_handle
        
        # Compile the expression
        func = compile_handle(handle)
        
        # Profile the compiled function
        def eval_func():
            func(X_data)
        
        return self.profile(eval_func, name=f"expr_{handle}", iterations=iterations)
    
    def compare_expressions(
        self,
        h1: int,
        h2: int,
        X_data: List[float],
        iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare the execution time of two expressions.
        
        Args:
            h1: First expression handle.
            h2: Second expression handle.
            X_data: Input data for evaluation.
            iterations: Number of measurement iterations.
            
        Returns:
            Comparison results.
        """
        p1 = self.profile_expression(h1, X_data, iterations)
        p2 = self.profile_expression(h2, X_data, iterations)
        
        return {
            'expr1_time_ms': p1.get_mean_ms(),
            'expr2_time_ms': p2.get_mean_ms(),
            'expr1_samples': p1.n_samples,
            'expr2_samples': p2.n_samples,
            'time_ratio': p1.get_mean_ms() / (p2.get_mean_ms() + 1e-10),
            'speedup': p2.get_mean_ms() / (p1.get_mean_ms() + 1e-10),
            'expr1_stats': p1,
            'expr2_stats': p2,
        }
    
    def clear_cache(self) -> None:
        """Clear the profiling cache."""
        self._cache.clear()
        self._stats['measurements'] = 0
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        return {
            'measurements': self._stats['measurements'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_ratio': self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'] + 1),
            'cache_size': len(self._cache),
        }


# Global time profiler
_TIME_PROFILER: Optional[TimeProfiler] = None


def get_time_profiler() -> TimeProfiler:
    """Get or create the global time profiler."""
    global _TIME_PROFILER
    if _TIME_PROFILER is None:
        _TIME_PROFILER = TimeProfiler()
    return _TIME_PROFILER


def profile_time(func: Callable, *args, **kwargs) -> TimeProfile:
    """Profile a function's execution time."""
    return get_time_profiler().profile(func, *args, **kwargs)