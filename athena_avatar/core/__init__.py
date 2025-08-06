"""
Core components for Athena 3D Avatar
Memory management and model optimization
"""

from .memory_manager import MemoryManager, MemoryPriority, MemoryBlock
from .model_optimizer import ModelOptimizer, OptimizationLevel, ModelConfig

__all__ = [
    'MemoryManager',
    'MemoryPriority', 
    'MemoryBlock',
    'ModelOptimizer',
    'OptimizationLevel',
    'ModelConfig'
]