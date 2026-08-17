"""
Hash-consing expression pool for the Metanion engine.
Provides unique integer handles for every unique expression.
Implements structural sharing and efficient lookup.
"""

from typing import Optional, Dict, List, Tuple, Set, Any, Callable
from collections import defaultdict
import weakref
import threading

from .op_enum import OpID, get_op_arity, get_op_name
from .expression_node import ExpressionNode, ExpressionNodeFactory
from ..exceptions import PoolError, PoolOverflowError, HandleNotFoundError
from ..config import ACTIVE_CONFIG


class HashConsingPool:
    """
    Global expression pool with hash-consing.
    Every unique expression gets a unique integer handle.
    Expressions are stored in a dictionary keyed by hash.
    
    Thread-safe with a read-write lock.
    """
    
    def __init__(self, max_handles: int = 1_000_000):
        """
        Initialize the expression pool.
        
        Args:
            max_handles: Maximum number of unique expressions allowed.
        """
        self._max_handles = max_handles
        
        # Maps hash -> (op, left, right)
        self._hash_to_node: Dict[int, Tuple[OpID, Optional[int], Optional[int]]] = {}
        
        # Maps hash -> handle
        self._hash_to_handle: Dict[int, int] = {}
        
        # Maps handle -> hash
        self._handle_to_hash: Dict[int, int] = {}
        
        # Maps handle -> depth (cached)
        self._handle_to_depth: Dict[int, int] = {}
        
        # Maps handle -> node count (cached)
        self._handle_to_node_count: Dict[int, int] = {}
        
        # Reverse mapping: (op, left, right) -> hash (for fast lookup)
        self._node_to_hash: Dict[Tuple[OpID, Optional[int], Optional[int]], int] = {}
        
        # Statistics
        self._stats = {
            'total_handles': 0,
            'hash_collisions': 0,
            'lookup_hits': 0,
            'lookup_misses': 0,
            'intern_calls': 0,
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize with basic constants
        self._initialize_constants()
    
    def _initialize_constants(self):
        """Initialize the pool with basic constant expressions."""
        # Variable placeholder (identity)
        self.intern(OpID.IDENTITY, None, None)
        
        # Constants
        self.intern(OpID.CONST_ZERO, None, None)
        self.intern(OpID.CONST_ONE, None, None)
    
    def _compute_hash(self, op: OpID, left: Optional[int], right: Optional[int]) -> int:
        """
        Compute a deterministic hash for an expression node.
        
        Args:
            op: Operation ID.
            left: Left child handle (or None).
            right: Right child handle (or None).
            
        Returns:
            Hash value.
        """
        # Use Python's built-in hash for tuples
        # But we need deterministic hashing across runs
        hash_value = hash((op, left or 0, right or 0))
        return hash_value
    
    def _get_node_key(self, op: OpID, left: Optional[int], right: Optional[int]) -> Tuple[OpID, Optional[int], Optional[int]]:
        """Get the canonical node key."""
        return (op, left, right)
    
    def intern(self, op: OpID, left: Optional[int] = None, right: Optional[int] = None) -> int:
        """
        Intern an expression node and return its handle.
        If the expression already exists, return the existing handle.
        
        Args:
            op: Operation ID.
            left: Left child handle (or None for nullary).
            right: Right child handle (or None for unary).
            
        Returns:
            Unique integer handle.
            
        Raises:
            PoolOverflowError: If pool exceeds max handles.
        """
        with self._lock:
            self._stats['intern_calls'] += 1
            
            # Validate arity
            arity = get_op_arity(op)
            if arity == 0:
                if left is not None or right is not None:
                    raise PoolError(f"Nullary operation {get_op_name(op)} cannot have children")
            elif arity == 1:
                if left is None:
                    raise PoolError(f"Unary operation {get_op_name(op)} requires left child")
                if right is not None:
                    raise PoolError(f"Unary operation {get_op_name(op)} cannot have right child")
            elif arity == 2:
                if left is None or right is None:
                    raise PoolError(f"Binary operation {get_op_name(op)} requires both children")
            
            # Compute hash
            node_key = self._get_node_key(op, left, right)
            
            # Check if this node already exists
            if node_key in self._node_to_hash:
                node_hash = self._node_to_hash[node_key]
                handle = self._hash_to_handle[node_hash]
                self._stats['lookup_hits'] += 1
                return handle
            
            self._stats['lookup_misses'] += 1
            
            # Check if we have room
            if len(self._hash_to_handle) >= self._max_handles:
                raise PoolOverflowError(
                    f"Pool has reached maximum capacity of {self._max_handles} handles. "
                    f"Consider increasing max_handles in config."
                )
            
            # Create new handle
            handle = len(self._hash_to_handle) + 1  # 1-indexed, 0 is reserved
            
            # Compute hash
            node_hash = self._compute_hash(op, left, right)
            
            # Handle hash collisions
            if node_hash in self._hash_to_handle:
                # Collision - use linear probing
                self._stats['hash_collisions'] += 1
                original_hash = node_hash
                probe = 1
                while node_hash in self._hash_to_handle:
                    node_hash = self._compute_hash(op ^ probe, left, right)
                    probe += 1
                    if probe > 1000:
                        raise PoolError("Hash collision resolution failed after 1000 attempts")
            
            # Store the node
            self._hash_to_node[node_hash] = (op, left, right)
            self._hash_to_handle[node_hash] = handle
            self._handle_to_hash[handle] = node_hash
            self._node_to_hash[node_key] = node_hash
            
            # Update statistics
            self._stats['total_handles'] += 1
            
            return handle
    
    def get_node(self, handle: int) -> Optional[ExpressionNode]:
        """
        Get the expression node for a handle.
        
        Args:
            handle: The handle to look up.
            
        Returns:
            ExpressionNode instance, or None if not found.
        """
        with self._lock:
            if handle not in self._handle_to_hash:
                return None
            
            node_hash = self._handle_to_hash[handle]
            if node_hash not in self._hash_to_node:
                return None
            
            op, left, right = self._hash_to_node[node_hash]
            
            # Create ExpressionNode
            if get_op_arity(op) == 0:
                return ExpressionNode(op)
            elif get_op_arity(op) == 1:
                return ExpressionNode(op, left, None)
            else:
                return ExpressionNode(op, left, right)
    
    def get_op(self, handle: int) -> Optional[OpID]:
        """Get the operation ID for a handle."""
        node = self.get_node(handle)
        return node.op if node is not None else None
    
    def get_children(self, handle: int) -> Tuple[Optional[int], Optional[int]]:
        """Get the children of a handle."""
        node = self.get_node(handle)
        if node is None:
            return (None, None)
        return (node.left, node.right)
    
    def get_depth(self, handle: int) -> int:
        """
        Get the depth of the expression tree rooted at handle.
        Uses memoization for performance.
        
        Args:
            handle: Root handle.
            
        Returns:
            Maximum depth (number of nodes along longest path).
        """
        with self._lock:
            if handle in self._handle_to_depth:
                return self._handle_to_depth[handle]
            
            node = self.get_node(handle)
            if node is None:
                return 0
            
            if node.arity == 0:
                depth = 1
            else:
                max_child_depth = 0
                for child in node.get_children():
                    if child is not None:
                        child_depth = self.get_depth(child)
                        max_child_depth = max(max_child_depth, child_depth)
                depth = 1 + max_child_depth
            
            self._handle_to_depth[handle] = depth
            return depth
    
    def get_node_count(self, handle: int) -> int:
        """
        Get the total number of nodes in the expression tree.
        Uses memoization for performance.
        
        Args:
            handle: Root handle.
            
        Returns:
            Total node count.
        """
        with self._lock:
            if handle in self._handle_to_node_count:
                return self._handle_to_node_count[handle]
            
            node = self.get_node(handle)
            if node is None:
                return 0
            
            if node.arity == 0:
                count = 1
            else:
                count = 1
                for child in node.get_children():
                    if child is not None:
                        count += self.get_node_count(child)
            
            self._handle_to_node_count[handle] = count
            return count
    
    def exists(self, handle: int) -> bool:
        """Check if a handle exists in the pool."""
        with self._lock:
            return handle in self._handle_to_hash
    
    def get_handle(self, op: OpID, left: Optional[int] = None, right: Optional[int] = None) -> Optional[int]:
        """
        Get the handle for an expression node without interning it.
        
        Args:
            op: Operation ID.
            left: Left child handle.
            right: Right child handle.
            
        Returns:
            Handle if exists, None otherwise.
        """
        with self._lock:
            node_key = self._get_node_key(op, left, right)
            if node_key in self._node_to_hash:
                node_hash = self._node_to_hash[node_key]
                return self._hash_to_handle.get(node_hash)
            return None
    
    def get_handle_for_node(self, node: ExpressionNode) -> Optional[int]:
        """Get the handle for an ExpressionNode."""
        return self.get_handle(node.op, node.left, node.right)
    
    def intern_node(self, node: ExpressionNode) -> int:
        """Intern an ExpressionNode and return its handle."""
        return self.intern(node.op, node.left, node.right)
    
    def get_all_handles(self) -> List[int]:
        """Get all handles in the pool."""
        with self._lock:
            return list(self._handle_to_hash.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            stats = {
                'total_handles': self._stats['total_handles'],
                'unique_expressions': len(self._hash_to_handle),
                'hash_collisions': self._stats['hash_collisions'],
                'lookup_hits': self._stats['lookup_hits'],
                'lookup_misses': self._stats['lookup_misses'],
                'intern_calls': self._stats['intern_calls'],
                'hit_ratio': 0.0,
                'memory_estimate': 0,
            }
            
            if self._stats['lookup_hits'] + self._stats['lookup_misses'] > 0:
                stats['hit_ratio'] = self._stats['lookup_hits'] / (
                    self._stats['lookup_hits'] + self._stats['lookup_misses']
                )
            
            # Estimate memory usage (rough)
            stats['memory_estimate'] = (
                len(self._hash_to_handle) * 8 +  # handles
                len(self._hash_to_node) * 32 +   # nodes
                len(self._handle_to_hash) * 8 +  # mappings
                len(self._node_to_hash) * 32     # node keys
            )
            
            return stats
    
    def clear_cache(self):
        """Clear the depth and node count caches."""
        with self._lock:
            self._handle_to_depth.clear()
            self._handle_to_node_count.clear()
    
    def print_stats(self):
        """Print pool statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print("Hash-Consing Pool Statistics")
        print("=" * 50)
        print(f"Total Handles:        {stats['total_handles']}")
        print(f"Unique Expressions:   {stats['unique_expressions']}")
        print(f"Hash Collisions:      {stats['hash_collisions']}")
        print(f"Lookup Hits:          {stats['lookup_hits']}")
        print(f"Lookup Misses:        {stats['lookup_misses']}")
        print(f"Hit Ratio:            {stats['hit_ratio']:.2%}")
        print(f"Intern Calls:         {stats['intern_calls']}")
        print(f"Memory Estimate:      {stats['memory_estimate'] / 1024:.2f} KB")
        print("=" * 50)
    
    def __len__(self) -> int:
        """Get the number of unique expressions in the pool."""
        return len(self._hash_to_handle)
    
    def __contains__(self, handle: int) -> bool:
        """Check if a handle exists."""
        return self.exists(handle)


# Global pool instance
_POOL: Optional[HashConsingPool] = None


def get_pool() -> HashConsingPool:
    """Get or create the global expression pool."""
    global _POOL
    if _POOL is None:
        _POOL = HashConsingPool(ACTIVE_CONFIG.max_handles)
    return _POOL


def reset_pool() -> None:
    """Reset the global pool."""
    global _POOL
    _POOL = None


def intern(op: OpID, left: Optional[int] = None, right: Optional[int] = None) -> int:
    """Convenience function to intern an expression."""
    return get_pool().intern(op, left, right)


def lookup(handle: int) -> Optional[ExpressionNode]:
    """Convenience function to look up a handle."""
    return get_pool().get_node(handle)