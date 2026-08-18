"""
Memory arena for Metanion - simple pre-allocated memory manager.
"""

from typing import Dict, Optional, Set
from ..exceptions import MemoryAllocationError


class MemoryArena:
    """Simple memory arena with pre-allocated buffer."""
    
    def __init__(self, size_mb: int = 10):
        """
        Initialize memory arena.
        
        Args:
            size_mb: Size in megabytes.
        """
        self.size_bytes = size_mb * 1024 * 1024
        self.buffer = bytearray(self.size_bytes)
        self._allocations: Dict[int, int] = {}  # offset -> size
        self._next_offset: int = 0
        self._max_allocated: int = 0
    
    def allocate(self, size: int, alignment: int = 8) -> int:
        """
        Allocate a block of memory.
        
        Args:
            size: Number of bytes to allocate.
            alignment: Byte alignment.
            
        Returns:
            Offset of allocated block.
        """
        # Align size
        size = ((size + alignment - 1) // alignment) * alignment
        
        # Simple bump allocator
        offset = self._next_offset
        if offset + size > self.size_bytes:
            raise MemoryAllocationError(
                f"Out of memory: requested {size} bytes, "
                f"available {self.size_bytes - offset} bytes"
            )
        
        self._allocations[offset] = size
        self._next_offset += size
        self._max_allocated = max(self._max_allocated, self._next_offset)
        
        return offset
    
    def free(self, offset: int) -> None:
        """
        Free allocated memory.
        
        Args:
            offset: Offset to free.
        """
        if offset in self._allocations:
            del self._allocations[offset]
    
    def read(self, offset: int, size: int) -> bytes:
        """
        Read bytes from arena.
        
        Args:
            offset: Starting offset.
            size: Number of bytes to read.
            
        Returns:
            Bytes object.
        """
        if offset < 0 or offset + size > self.size_bytes:
            raise MemoryAllocationError(f"Invalid read range: {offset}:{offset+size}")
        
        # Check if offset is allocated
        if offset not in self._allocations:
            # Allow reading from allocated blocks only
            # Find if offset is within any allocation
            allocated = False
            for alloc_offset, alloc_size in self._allocations.items():
                if alloc_offset <= offset < alloc_offset + alloc_size:
                    allocated = True
                    break
            if not allocated:
                raise MemoryAllocationError(f"Cannot read from unallocated offset {offset}")
        
        return bytes(self.buffer[offset:offset+size])
    
    def write(self, offset: int, data: bytes) -> None:
        """
        Write bytes to arena.
        
        Args:
            offset: Starting offset.
            data: Bytes to write.
        """
        if offset < 0 or offset + len(data) > self.size_bytes:
            raise MemoryAllocationError(f"Invalid write range: {offset}:{offset+len(data)}")
        
        # Check if offset is allocated
        if offset not in self._allocations:
            # Find if offset is within any allocation
            allocated = False
            for alloc_offset, alloc_size in self._allocations.items():
                if alloc_offset <= offset < alloc_offset + alloc_size:
                    allocated = True
                    break
            if not allocated:
                raise MemoryAllocationError(f"Cannot write to unallocated offset {offset}")
        
        self.buffer[offset:offset+len(data)] = data
    
    def get_buffer(self, offset: int, size: int) -> memoryview:
        """
        Get memoryview of allocated block.
        
        Args:
            offset: Starting offset.
            size: Size of block.
            
        Returns:
            Memoryview object.
        """
        if offset not in self._allocations:
            raise MemoryAllocationError(f"Offset {offset} is not allocated")
        
        if self._allocations[offset] < size:
            raise MemoryAllocationError(
                f"Requested {size} bytes but block has {self._allocations[offset]} bytes"
            )
        
        return memoryview(self.buffer)[offset:offset+size]
    
    def clear(self) -> None:
        """Clear all allocations."""
        self._allocations.clear()
        self._next_offset = 0
        self._max_allocated = 0
    
    def stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            'total_mb': self.size_bytes / (1024 * 1024),
            'used_mb': self._next_offset / (1024 * 1024),
            'used_percent': (self._next_offset / self.size_bytes) * 100,
            'allocation_count': len(self._allocations),
        }


# Global arena instance
_ARENA: Optional[MemoryArena] = None


def get_arena() -> MemoryArena:
    """Get or create the global memory arena."""
    global _ARENA
    if _ARENA is None:
        _ARENA = MemoryArena(10)  # 10 MB default
    return _ARENA


def reset_arena() -> None:
    """Reset the global arena."""
    global _ARENA
    if _ARENA is not None:
        _ARENA.clear()
    _ARENA = None
