"""
Device Utilities for ChatGPT+ Clone
Centralized GPU detection and device management for optimal performance
"""

import torch
import logging
import psutil
import platform
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceInfo:
    """Device information and capabilities"""
    name: str
    type: str  # 'cuda', 'mps', 'cpu'
    memory_total: Optional[int] = None  # in MB
    memory_available: Optional[int] = None  # in MB
    compute_capability: Optional[str] = None  # for CUDA
    is_available: bool = True

class DeviceManager:
    """Centralized device management for AI models"""
    
    def __init__(self):
        self.device_info = self._detect_devices()
        self.current_device = self._select_best_device()
        self.fallback_devices = self._get_fallback_devices()
        
        logger.info(f"Device Manager initialized")
        logger.info(f"Available devices: {[d.name for d in self.device_info.values()]}")
        logger.info(f"Selected device: {self.current_device}")
    
    def _detect_devices(self) -> Dict[str, DeviceInfo]:
        """Detect all available devices"""
        devices = {}
        
        # CPU (always available)
        devices['cpu'] = DeviceInfo(
            name="CPU",
            type="cpu",
            memory_total=psutil.virtual_memory().total // (1024 * 1024),  # MB
            memory_available=psutil.virtual_memory().available // (1024 * 1024),  # MB
            is_available=True
        )
        
        # CUDA GPU
        if torch.cuda.is_available():
            try:
                cuda_device = torch.device('cuda')
                gpu_props = torch.cuda.get_device_properties(cuda_device)
                
                devices['cuda'] = DeviceInfo(
                    name=f"CUDA GPU ({gpu_props.name})",
                    type="cuda",
                    memory_total=gpu_props.total_memory // (1024 * 1024),  # MB
                    memory_available=torch.cuda.get_device_properties(cuda_device).total_memory // (1024 * 1024),  # MB
                    compute_capability=f"{gpu_props.major}.{gpu_props.minor}",
                    is_available=True
                )
                
                logger.info(f"CUDA GPU detected: {gpu_props.name}")
                logger.info(f"CUDA Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                
            except Exception as e:
                logger.warning(f"CUDA detection failed: {e}")
                devices['cuda'] = DeviceInfo(
                    name="CUDA GPU (Error)",
                    type="cuda",
                    is_available=False
                )
        
        # Apple Silicon GPU (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                devices['mps'] = DeviceInfo(
                    name="Apple Silicon GPU (MPS)",
                    type="mps",
                    is_available=True
                )
                logger.info("Apple Silicon GPU (MPS) detected")
                
            except Exception as e:
                logger.warning(f"MPS detection failed: {e}")
                devices['mps'] = DeviceInfo(
                    name="Apple Silicon GPU (Error)",
                    type="mps",
                    is_available=False
                )
        
        return devices
    
    def _select_best_device(self) -> torch.device:
        """Select the best available device based on priority"""
        device_priority = ['cuda', 'mps', 'cpu']
        
        for device_type in device_priority:
            if device_type in self.device_info:
                device_info = self.device_info[device_type]
                if device_info.is_available:
                    if device_type == 'cuda':
                        return torch.device('cuda')
                    elif device_type == 'mps':
                        return torch.device('mps')
                    else:
                        return torch.device('cpu')
        
        # Fallback to CPU
        return torch.device('cpu')
    
    def _get_fallback_devices(self) -> list:
        """Get list of fallback devices in order of preference"""
        fallbacks = []
        
        if 'cuda' in self.device_info and self.device_info['cuda'].is_available:
            fallbacks.append(torch.device('cuda'))
        
        if 'mps' in self.device_info and self.device_info['mps'].is_available:
            fallbacks.append(torch.device('mps'))
        
        fallbacks.append(torch.device('cpu'))
        
        return fallbacks
    
    def get_best_device(self) -> torch.device:
        """Get the best available device"""
        return self.current_device
    
    def get_device_info(self, device_type: str = None) -> Optional[DeviceInfo]:
        """Get information about a specific device"""
        if device_type:
            return self.device_info.get(device_type)
        return self.device_info.get(self.current_device.type)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for all devices"""
        memory_info = {}
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu'] = {
            'total': cpu_memory.total // (1024 * 1024),  # MB
            'available': cpu_memory.available // (1024 * 1024),  # MB
            'used': cpu_memory.used // (1024 * 1024),  # MB
            'percent': cpu_memory.percent
        }
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                memory_info['cuda'] = {
                    'allocated': gpu_memory['allocated_bytes.all.current'] // (1024 * 1024),  # MB
                    'reserved': gpu_memory['reserved_bytes.all.current'] // (1024 * 1024),  # MB
                    'total': torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),  # MB
                }
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")
        
        return memory_info
    
    def optimize_for_model(self, model_name: str, model_size: str = "base") -> torch.device:
        """Optimize device selection for specific model requirements"""
        
        # Model-specific optimizations
        if "whisper" in model_name.lower():
            # Whisper works well on GPU but can run on CPU
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        elif "llm" in model_name.lower() or "gpt" in model_name.lower():
            # Large language models benefit greatly from GPU
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        
        elif "tts" in model_name.lower():
            # TTS can work on GPU or CPU
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        
        else:
            # Default to best available device
            return self.get_best_device()
    
    def safe_model_load(self, model, device: torch.device = None) -> torch.device:
        """Safely load a model to the specified device with fallback"""
        if device is None:
            device = self.get_best_device()
        
        try:
            model.to(device)
            logger.info(f"Model loaded successfully on {device}")
            return device
            
        except Exception as e:
            logger.warning(f"Failed to load model on {device}: {e}")
            
            # Try fallback devices
            for fallback_device in self.fallback_devices:
                if fallback_device != device:
                    try:
                        model.to(fallback_device)
                        logger.info(f"Model loaded on fallback device: {fallback_device}")
                        return fallback_device
                    except Exception as fallback_error:
                        logger.warning(f"Fallback to {fallback_device} failed: {fallback_error}")
                        continue
            
            # Last resort: CPU
            try:
                model.to(torch.device('cpu'))
                logger.info("Model loaded on CPU as last resort")
                return torch.device('cpu')
            except Exception as cpu_error:
                logger.error(f"Failed to load model on any device: {cpu_error}")
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'current_device': str(self.current_device),
            'device_info': self.device_info,
            'memory_usage': self.get_memory_usage(),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        if torch.cuda.is_available():
            stats['cuda_device_count'] = torch.cuda.device_count()
            stats['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return stats

# Global device manager instance
device_manager = DeviceManager()

def get_best_device() -> torch.device:
    """Get the best available device (convenience function)"""
    return device_manager.get_best_device()

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    return device_manager.get_performance_stats()

def optimize_for_model(model_name: str, model_size: str = "base") -> torch.device:
    """Optimize device selection for specific model"""
    return device_manager.optimize_for_model(model_name, model_size)

def safe_model_load(model, device: torch.device = None) -> torch.device:
    """Safely load a model with fallback"""
    return device_manager.safe_model_load(model, device)

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.device_manager = device_manager
    
    def start_timing(self):
        """Start performance timing"""
        import time
        self.start_time = time.time()
    
    def end_timing(self, operation: str):
        """End performance timing and log results"""
        import time
        if self.start_time:
            duration = time.time() - self.start_time
            memory_info = self.device_manager.get_memory_usage()
            
            logger.info(f"Performance - {operation}: {duration:.2f}s")
            logger.info(f"Memory usage: {memory_info}")
            
            self.start_time = None
    
    def log_device_usage(self):
        """Log current device usage"""
        stats = self.device_manager.get_performance_stats()
        logger.info(f"Device usage: {stats}")

# Convenience functions for common operations
def is_gpu_available() -> bool:
    """Check if any GPU is available"""
    return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

def get_gpu_memory_info() -> Dict[str, int]:
    """Get GPU memory information in MB"""
    if torch.cuda.is_available():
        return {
            'total': torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
            'allocated': torch.cuda.memory_allocated() // (1024 * 1024),
            'cached': torch.cuda.memory_reserved() // (1024 * 1024)
        }
    return {}

def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

def set_memory_fraction(fraction: float):
    """Set GPU memory fraction (0.0 to 1.0)"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"GPU memory fraction set to {fraction}")

# Initialize and log device information
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Device Detection Results ===")
    print(f"Best device: {get_best_device()}")
    print(f"GPU available: {is_gpu_available()}")
    
    device_info = get_device_info()
    print(f"Platform: {device_info['platform']}")
    print(f"Torch version: {device_info['torch_version']}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {device_info['cuda_device_count']}")
        print(f"CUDA device name: {device_info['cuda_device_name']}")
    
    memory_info = device_manager.get_memory_usage()
    print(f"Memory usage: {memory_info}")