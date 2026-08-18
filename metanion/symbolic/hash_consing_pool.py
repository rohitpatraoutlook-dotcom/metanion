"""
Hash-consing expression pool for the Metanion engine.
"""

from typing import Optional, Dict, List, Tuple, Any
from ..exceptions import PoolError, HandleNotFoundError
from ..config import ACTIVE_CONFIG


class HashConsingPool:
    """Global expression pool with hash-consing."""

    def __init__(self, max_handles: int = 1000000):
        self._max_handles = max_handles
        self._pool: Dict[Tuple[Any, Optional[int], Optional[int]], int] = {}
        self._reverse: Dict[int, Tuple[Any, Optional[int], Optional[int]]] = {}
        self._next_handle = 1
        self._initialize_constants()

    def _initialize_constants(self):
        """Initialize the pool with basic constants."""
        # Identity - handle 1, no children needed
        self._pool[(1, None, None)] = 1  # OpID.IDENTITY = 1
        self._reverse[1] = (1, None, None)

        # Constants
        self._pool[(2, None, None)] = 2  # OpID.CONST_ZERO = 2
        self._reverse[2] = (2, None, None)

        self._pool[(3, None, None)] = 3  # OpID.CONST_ONE = 3
        self._reverse[3] = (3, None, None)

        self._next_handle = 4

    def intern(self, op: int, left: Optional[int] = None, right: Optional[int] = None) -> int:
        """Intern an expression and return its handle."""
        # Validate arity (skip validation for initialization)
        if left is None and right is None:
            # Nullary operation - just use the key directly
            key = (op, None, None)
            if key in self._pool:
                return self._pool[key]

            if len(self._pool) >= self._max_handles:
                raise PoolError(f"Pool capacity exceeded: {self._max_handles}")

            handle = self._next_handle
            self._next_handle += 1
            self._pool[key] = handle
            self._reverse[handle] = key
            return handle

        # Binary operation
        if left is not None and right is not None:
            key = (op, left, right)
            if key in self._pool:
                return self._pool[key]

            if len(self._pool) >= self._max_handles:
                raise PoolError(f"Pool capacity exceeded: {self._max_handles}")

            handle = self._next_handle
            self._next_handle += 1
            self._pool[key] = handle
            self._reverse[handle] = key
            return handle

        # Unary operation
        if left is not None and right is None:
            key = (op, left, None)
            if key in self._pool:
                return self._pool[key]

            if len(self._pool) >= self._max_handles:
                raise PoolError(f"Pool capacity exceeded: {self._max_handles}")

            handle = self._next_handle
            self._next_handle += 1
            self._pool[key] = handle
            self._reverse[handle] = key
            return handle

        raise PoolError(f"Invalid arguments: op={op}, left={left}, right={right}")

    def get_node(self, handle: int) -> Optional[Tuple]:
        """Get the node for a handle."""
        if handle in self._reverse:
            return self._reverse[handle]
        return None

    def exists(self, handle: int) -> bool:
        """Check if a handle exists."""
        return handle in self._reverse

    def get_handle(self, op: int, left: Optional[int] = None, right: Optional[int] = None) -> Optional[int]:
        """Get handle for an expression if it exists."""
        key = (op, left, right)
        return self._pool.get(key)

    def get_all_handles(self) -> List[int]:
        """Get all handles in the pool."""
        return list(self._reverse.keys())

    def clear(self):
        """Clear the pool."""
        self._pool.clear()
        self._reverse.clear()
        self._next_handle = 1
        self._initialize_constants()


# Global pool instance
_POOL: Optional[HashConsingPool] = None


def get_pool() -> HashConsingPool:
    """Get or create the global pool."""
    global _POOL
    if _POOL is None:
        _POOL = HashConsingPool(ACTIVE_CONFIG.max_handles)
    return _POOL


def reset_pool() -> None:
    """Reset the global pool."""
    global _POOL
    _POOL = None


def intern(op: int, left: Optional[int] = None, right: Optional[int] = None) -> int:
    """Convenience function to intern an expression."""
    return get_pool().intern(op, left, right)


def lookup(handle: int) -> Optional[Tuple]:
    """Convenience function to look up a handle."""
    return get_pool().get_node(handle)
