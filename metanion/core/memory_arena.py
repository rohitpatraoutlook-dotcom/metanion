"""
Memory arena for efficient tensor allocation.
Pre-allocates a large bytearray and manages allocations via offsets.
"""

import sys
from typing import Optional, Dict, Set
from collections import defaultdict

from ..exceptions import MemoryAllocationError
from ..config import ACTIVE_CONFIG


class MemoryArena:
    """
    Simple memory arena with mark-sweep allocation.
    Allocates a fixed-size bytearray and hands out slices.
    """
    
    def __init__(self, size_mb: int = 100):
        """
        Initialize the memory arena.
        
        Args:
            size_mb: Size of the arena in megabytes.
        """
        self.size_bytes = size_mb * 1024 * 1024
        self.buffer = bytearray(self.size_bytes)
        self._allocated: Dict[int, int] = {}  # offset -> size
        self._free_blocks: Set[int] = set()   # offsets of free blocks
        
        # Initially, the entire buffer is free
        self._free_blocks.add(0)
        self._block_sizes: Dict[int, int] = {0: self.size_bytes}
        
        # Statistics
        self.total_allocated = 0
        self.peak_allocated = 0
        self.allocation_count = 0
        
    def allocate(self, size: int, alignment: int = 8) -> int:
        """
        Allocate a block of memory.
        
        Args:
            size: Number of bytes to allocate.
            alignment: Byte alignment (must be power of 2).
            
        Returns:
            Offset (byte position) of the allocated block.
            
        Raises:
            MemoryAllocationError: If memory is exhausted.
        """
        if size <= 0:
            raise ValueError("size must be > 0")
        
        # Align size
        size = ((size + alignment - 1) // alignment) * alignment
        
        # Find a free block that fits
        for offset in sorted(self._free_blocks):
            block_size = self._block_sizes[offset]
            if block_size >= size:
                # Allocate from this block
                self._free_blocks.remove(offset)
                del self._block_sizes[offset]
                
                # Record allocation
                self._allocated[offset] = size
                
                # If there's leftover space, create a new free block
                if block_size > size:
                    new_offset = offset + size
                    new_size = block_size - size
                    self._free_blocks.add(new_offset)
                    self._block_sizes[new_offset] = new_size
                
                # Update statistics
                self.total_allocated += size
                if self.total_allocated > self.peak_allocated:
                    self.peak_allocated = self.total_allocated
                self.allocation_count += 1
                
                return offset
        
        # Out of memory
        used_mb = self.total_allocated / (1024 * 1024)
        total_mb = self.size_bytes / (1024 * 1024)
        raise MemoryAllocationError(
            f"Memory exhausted. Used: {used_mb:.2f}MB / {total_mb:.2f}MB. "
            f"Requested: {size} bytes. Try increasing arena_size_mb."
        )
    
    def free(self, offset: int) -> None:
        """
        Free an allocated block.
        
        Args:
            offset: Offset of the block to free.
            
        Raises:
            MemoryAllocationError: If offset is not allocated.
        """
        if offset not in self._allocated:
            raise MemoryAllocationError(f"Offset {offset} is not allocated.")
        
        size = self._allocated.pop(offset)
        self.total_allocated -= size
        
        # Add block back to free list
        self._free_blocks.add(offset)
        self._block_sizes[offset] = size
        
        # Coalesce adjacent free blocks
        self._coalesce(offset)
    
    def _coalesce(self, offset: int) -> None:
        """
        Coalesce adjacent free blocks to reduce fragmentation.
        
        Args:
            offset: Offset of the free block to coalesce.
        """
        if offset not in self._free_blocks:
            return
        
        size = self._block_sizes[offset]
        
        # Check right neighbor
        right_offset = offset + size
        if right_offset in self._free_blocks:
            # Merge with right block
            right_size = self._block_sizes.pop(right_offset)
            self._free_blocks.remove(right_offset)
            size += right_size
            self._block_sizes[offset] = size
        
        # Check left neighbor
        # Find block that ends at offset
        for candidate in list(self._free_blocks):
            if candidate == offset:
                continue
            if candidate + self._block_sizes[candidate] == offset:
                # Merge left block into this one
                left_size = self._block_sizes.pop(candidate)
                self._free_blocks.remove(candidate)
                del self._block_sizes[candidate]
                # Update the new start
                new_offset = candidate
                new_size = left_size + size
                self._free_blocks.add(new_offset)
                self._block_sizes[new_offset] = new_size
                # Remove old block
                self._free_blocks.remove(offset)
                del self._block_sizes[offset]
                # Recursively coalesce from new offset
                self._coalesce(new_offset)
                return
        
        # If no left merge, just keep the current block
        self._block_sizes[offset] = size
    
    def read(self, offset: int, size: int) -> bytes:
        """
        Read bytes from the arena.
        
        Args:
            offset: Starting offset.
            size: Number of bytes to read.
            
        Returns:
            Bytes object with the data.
        """
        if offset < 0 or offset + size > self.size_bytes:
            raise MemoryAllocationError(f"Invalid read range: {offset}:{offset+size}")
        
        # Check if offset is allocated
        if offset not in self._allocated:
            raise MemoryAllocationError(f"Cannot read from unallocated offset {offset}")
        
        return bytes(self.buffer[offset:offset+size])
    
    def write(self, offset: int, data: bytes) -> None:
        """
        Write bytes to the arena.
        
        Args:
            offset: Starting offset.
            data: Bytes to write.
        """
        if offset < 0 or offset + len(data) > self.size_bytes:
            raise MemoryAllocationError(f"Invalid write range: {offset}:{offset+len(data)}")
        
        if offset not in self._allocated:
            raise MemoryAllocationError(f"Cannot write to unallocated offset {offset}")
        
        self.buffer[offset:offset+len(data)] = data
    
    def get_buffer(self, offset: int, size: int) -> memoryview:
        """
        Get a memoryview of an allocated block.
        
        Args:
            offset: Starting offset.
            size: Size of the block.
            
        Returns:
            memoryview object.
        """
        if offset not in self._allocated:
            raise MemoryAllocationError(f"Offset {offset} is not allocated.")
        
        if self._allocated[offset] < size:
            raise MemoryAllocationError(
                f"Requested {size} bytes but block has {self._allocated[offset]} bytes."
            )
        
        return memoryview(self.buffer)[offset:offset+size]
    
    def get_allocated_blocks(self) -> Dict[int, int]:
        """Return a copy of the allocated blocks mapping."""
        return dict(self._allocated)
    
    def clear(self) -> None:
        """Clear all allocations (reset the arena)."""
        self._allocated.clear()
        self._free_blocks.clear()
        self._block_sizes.clear()
        
        self._free_blocks.add(0)
        self._block_sizes[0] = self.size_bytes
        
        self.total_allocated = 0
        self.peak_allocated = 0
        self.allocation_count = 0
    
    def stats(self) -> Dict[str, float]:
        """
        Return memory statistics.
        
        Returns:
            Dictionary with memory usage stats.
        """
        total_mb = self.size_bytes / (1024 * 1024)
        used_mb = self.total_allocated / (1024 * 1024)
        peak_mb = self.peak_allocated / (1024 * 1024)
        free_mb = (self.size_bytes - self.total_allocated) / (1024 * 1024)
        
        fragmentation = 0
        if self._free_blocks:
            free_blocks = sorted(self._free_blocks)
            # If there are many small blocks, fragmentation is high
            fragmentation = len(free_blocks) / (self.size_bytes / (1024 * 1024))
        
        return {
            'total_mb': total_mb,
            'used_mb': used_mb,
            'peak_mb': peak_mb,
            'free_mb': free_mb,
            'used_percent': (used_mb / total_mb) * 100,
            'allocation_count': self.allocation_count,
            'fragmentation_score': min(fragmentation, 1.0),
        }


# Global arena instance
_ARENA: Optional[MemoryArena] = None


def get_arena() -> MemoryArena:
    """Get or create the global memory arena."""
    global _ARENA
    if _ARENA is None:
        _ARENA = MemoryArena(ACTIVE_CONFIG.arena_size_mb)
    return _ARENA


def reset_arena() -> None:
    """Reset the global arena (clears all allocations)."""
    global _ARENA
    if _ARENA is not None:
        _ARENA.clear()