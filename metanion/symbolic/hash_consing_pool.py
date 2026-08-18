"""
Hash-consing expression pool for the Metanion engine.
"""

from typing import Optional, Dict, List, Tuple, Any
from .op_enum import OpID, get_op_arity, get_op_name
from ..exceptions import PoolError


class HashConsingPool:
    def __init__(self, max_handles: int = 1000000):
        self._max_handles = max_handles
        self._pool: Dict[Any, int] = {}
        self._reverse: Dict[int, Any] = {}
        self._next_handle = 1
        self._initialize_constants()

    def _initialize_constants(self):
        # Identity (still used as default variable)
        self._pool[(OpID.IDENTITY, None, None)] = 1
        self._reverse[1] = (OpID.IDENTITY, None, None)
        # Zero
        self._pool[(OpID.CONST_ZERO, None, None)] = 2
        self._reverse[2] = (OpID.CONST_ZERO, None, None)
        # One
        self._pool[(OpID.CONST_ONE, None, None)] = 3
        self._reverse[3] = (OpID.CONST_ONE, None, None)
        self._next_handle = 4

    def intern(self, op: OpID, left: Optional[int] = None, right: Optional[int] = None, value: Optional[float] = None, index: Optional[int] = None) -> int:
        if op == OpID.CONST:
            if value is None:
                raise ValueError("CONST requires a value")
            key = (op, value)
        elif op == OpID.VAR:
            if index is None:
                raise ValueError("VAR requires an index")
            key = (op, index)
        else:
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

    def get_node(self, handle: int) -> Optional[Any]:
        if handle in self._reverse:
            return self._reverse[handle]
        return None

    def exists(self, handle: int) -> bool:
        return handle in self._reverse

    def get_handle(self, op: OpID, left: Optional[int] = None, right: Optional[int] = None, value: Optional[float] = None, index: Optional[int] = None) -> Optional[int]:
        if op == OpID.CONST:
            if value is None:
                return None
            key = (op, value)
        elif op == OpID.VAR:
            if index is None:
                return None
            key = (op, index)
        else:
            key = (op, left, right)
        return self._pool.get(key)

    def get_all_handles(self) -> List[int]:
        return list(self._reverse.keys())

    def clear(self):
        self._pool.clear()
        self._reverse.clear()
        self._next_handle = 1
        self._initialize_constants()


_POOL: Optional[HashConsingPool] = None

def get_pool() -> HashConsingPool:
    global _POOL
    if _POOL is None:
        _POOL = HashConsingPool()
    return _POOL

def reset_pool() -> None:
    global _POOL
    _POOL = None

def intern(op: OpID, left: Optional[int] = None, right: Optional[int] = None, value: Optional[float] = None, index: Optional[int] = None) -> int:
    return get_pool().intern(op, left, right, value, index)

def lookup(handle: int) -> Optional[Any]:
    return get_pool().get_node(handle)
