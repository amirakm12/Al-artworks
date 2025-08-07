"""
Memory Manager for Athena 3D Avatar
Optimizes PyTorch models for 12GB RAM constraint
"""

import psutil
import gc
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time

class MemoryPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class MemoryBlock:
    """Represents a memory block with metadata"""
    id: str
    size_gb: float
    priority: MemoryPriority
    last_accessed: float
    is_pinned: bool = False
    device: str = "cpu"

class MemoryManager:
    """Advanced memory manager for PyTorch models"""
    
    def __init__(self, max_ram_gb: float = 12.0, safety_margin_gb: float = 1.0):
        self.max_ram_gb = max_ram_gb
        self.safety_margin_gb = safety_margin_gb
        self.available_ram_gb = max_ram_gb - safety_margin_gb
        
        # Memory tracking
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.total_allocated_gb = 0.0
        
        # Performance tracking
        self.memory_usage_history: List[Tuple[float, float]] = []
        self.gc_times: List[float] = []
        
        # Configuration
        self.enable_automatic_gc = True
        self.gc_threshold_gb = 0.5  # Trigger GC when 0.5GB over limit
        self.max_history_size = 1000
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def initialize(self):
        """Initialize the memory manager"""
        try:
            # Set PyTorch memory management
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends, 'flash_attention'):
                torch.backends.flash_attention.enabled = True
            
            # Set memory fraction for GPU
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
                torch.cuda.empty_cache()
            
            self.logger.info(f"Memory manager initialized with {self.available_ram_gb:.1f}GB available")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}")
            return False
    
    def allocate_memory(self, block_id: str, size_gb: float, 
                       priority: MemoryPriority = MemoryPriority.MEDIUM,
                       device: str = "cpu") -> bool:
        """Allocate memory block with priority management"""
        try:
            # Check if allocation is possible
            if not self.can_allocate(size_gb):
                # Try to free memory
                if not self.free_memory_for_allocation(size_gb, priority):
                    self.logger.warning(f"Cannot allocate {size_gb:.2f}GB for {block_id}")
                    return False
            
            # Create memory block
            memory_block = MemoryBlock(
                id=block_id,
                size_gb=size_gb,
                priority=priority,
                last_accessed=time.time(),
                device=device
            )
            
            self.memory_blocks[block_id] = memory_block
            self.total_allocated_gb += size_gb
            
            self.logger.info(f"Allocated {size_gb:.2f}GB for {block_id} (Total: {self.total_allocated_gb:.2f}GB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to allocate memory for {block_id}: {e}")
            return False
    
    def can_allocate(self, size_gb: float) -> bool:
        """Check if memory allocation is possible"""
        current_usage = self.get_current_usage()
        return (current_usage + size_gb) <= self.available_ram_gb
    
    def free_memory_for_allocation(self, required_gb: float, 
                                 new_priority: MemoryPriority) -> bool:
        """Free memory by evicting lower priority blocks"""
        try:
            # Sort blocks by priority and last accessed time
            blocks_to_evict = []
            total_freed = 0.0
            
            # Get blocks sorted by priority (lowest first) and access time
            sorted_blocks = sorted(
                self.memory_blocks.values(),
                key=lambda x: (x.priority.value, x.last_accessed)
            )
            
            for block in sorted_blocks:
                if block.priority.value > new_priority.value:
                    blocks_to_evict.append(block)
                    total_freed += block.size_gb
                    
                    if total_freed >= required_gb:
                        break
            
            # Evict blocks
            for block in blocks_to_evict:
                self.free_memory_block(block.id)
            
            # Force garbage collection
            if self.enable_automatic_gc:
                self.force_garbage_collection()
            
            return total_freed >= required_gb
            
        except Exception as e:
            self.logger.error(f"Failed to free memory: {e}")
            return False
    
    def free_memory_block(self, block_id: str) -> bool:
        """Free a specific memory block"""
        try:
            if block_id in self.memory_blocks:
                block = self.memory_blocks[block_id]
                self.total_allocated_gb -= block.size_gb
                del self.memory_blocks[block_id]
                
                self.logger.info(f"Freed {block.size_gb:.2f}GB from {block_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to free memory block {block_id}: {e}")
            return False
    
    def update_memory_access(self, block_id: str):
        """Update last access time for memory block"""
        if block_id in self.memory_blocks:
            self.memory_blocks[block_id].last_accessed = time.time()
    
    def get_current_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            # Get system memory usage
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024**3)
            
            # Add PyTorch GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                used_gb += gpu_memory
            
            return used_gb
            
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        current_usage = self.get_current_usage()
        return max(0.0, self.max_ram_gb - current_usage)
    
    def force_garbage_collection(self):
        """Force garbage collection and clear PyTorch cache"""
        try:
            start_time = time.time()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force Python garbage collection
            collected = gc.collect()
            
            # Record GC time
            gc_time = time.time() - start_time
            self.gc_times.append(gc_time)
            
            # Keep only recent GC times
            if len(self.gc_times) > 100:
                self.gc_times.pop(0)
            
            self.logger.info(f"Garbage collection completed in {gc_time:.3f}s, collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Garbage collection failed: {e}")
    
    def optimize_pytorch_memory(self):
        """Optimize PyTorch memory usage"""
        try:
            # Set memory efficient settings
            torch.backends.cudnn.benchmark = True
            
            # Enable memory efficient attention
            if hasattr(torch.backends, 'flash_attention'):
                torch.backends.flash_attention.enabled = True
            
            # Set memory fraction
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("PyTorch memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"PyTorch memory optimization failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        try:
            stats = {
                'total_ram_gb': self.max_ram_gb,
                'available_ram_gb': self.get_available_memory(),
                'current_usage_gb': self.get_current_usage(),
                'allocated_blocks_gb': self.total_allocated_gb,
                'block_count': len(self.memory_blocks),
                'avg_gc_time_ms': np.mean(self.gc_times) * 1000 if self.gc_times else 0.0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    def monitor_memory_usage(self):
        """Monitor memory usage and trigger optimizations if needed"""
        try:
            current_usage = self.get_current_usage()
            self.memory_usage_history.append((time.time(), current_usage))
            
            # Keep only recent history
            if len(self.memory_usage_history) > self.max_history_size:
                self.memory_usage_history.pop(0)
            
            # Check if memory usage is high
            if current_usage > (self.max_ram_gb - self.gc_threshold_gb):
                self.logger.warning(f"High memory usage detected: {current_usage:.2f}GB")
                self.force_garbage_collection()
            
        except Exception as e:
            self.logger.error(f"Memory monitoring failed: {e}")
    
    def cleanup(self):
        """Cleanup memory manager resources"""
        try:
            # Free all memory blocks
            for block_id in list(self.memory_blocks.keys()):
                self.free_memory_block(block_id)
            
            # Force final garbage collection
            self.force_garbage_collection()
            
            # Clear history
            self.memory_usage_history.clear()
            self.gc_times.clear()
            
            self.logger.info("Memory manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Memory manager cleanup failed: {e}")