"""
GPU Device Manager - Advanced GPU Detection & Optimization
Handles CUDA, MPS, and CPU device selection with memory management
"""

import torch
import logging
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("DeviceManager")

@dataclass
class GPUInfo:
    """GPU device information"""
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    is_available: bool = True

class DeviceManager:
    """Advanced GPU device detection and management"""
    
    def __init__(self):
        self.device = self.detect_optimal_device()
        self.gpu_info = self._get_gpu_info()
        self.memory_tracker = MemoryTracker()
        
        logger.info(f"DeviceManager initialized on device: {self.device}")
        logger.info(f"GPU Info: {self.gpu_info}")
    
    def detect_optimal_device(self) -> torch.device:
        """Detect and select the optimal device with priority: CUDA > MPS > CPU"""
        
        # Check CUDA GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA GPU(s)")
            
            # Select GPU with most memory
            best_gpu = self._select_best_cuda_gpu()
            device = torch.device(f"cuda:{best_gpu}")
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info(f"Selected CUDA GPU {best_gpu}")
            return device
        
        # Check Apple Silicon GPU (MPS)
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple MPS GPU detected")
                return torch.device("mps")
        except ImportError:
            logger.info("MPS backend not available")
        
        # Fallback to CPU
        logger.info("Falling back to CPU")
        return torch.device("cpu")
    
    def _select_best_cuda_gpu(self) -> int:
        """Select the best CUDA GPU based on memory and compute capability"""
        gpu_scores = []
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            # Calculate score based on memory and compute capability
            memory_score = props.total_memory / (1024**3)  # GB
            compute_score = props.major * 10 + props.minor  # Compute capability
            
            total_score = memory_score + compute_score * 0.1
            gpu_scores.append((i, total_score, props))
            
            logger.info(f"GPU {i}: {props.name}, Memory: {memory_score:.1f}GB, "
                       f"Compute: {props.major}.{props.minor}")
        
        # Select GPU with highest score
        best_gpu = max(gpu_scores, key=lambda x: x[1])[0]
        return best_gpu
    
    def _get_gpu_info(self) -> Dict[str, GPUInfo]:
        """Get comprehensive GPU information"""
        gpu_info = {}
        
        # CPU info
        cpu_memory = psutil.virtual_memory()
        gpu_info['cpu'] = GPUInfo(
            name="CPU",
            memory_total=cpu_memory.total // (1024 * 1024),  # MB
            memory_available=cpu_memory.available // (1024 * 1024),  # MB
            is_available=True
        )
        
        # CUDA GPU info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info[f'cuda:{i}'] = GPUInfo(
                    name=props.name,
                    memory_total=props.total_memory // (1024 * 1024),  # MB
                    memory_available=props.total_memory // (1024 * 1024),  # MB
                    compute_capability=f"{props.major}.{props.minor}",
                    is_available=True
                )
        
        # MPS GPU info (Apple Silicon)
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps'] = GPUInfo(
                    name="Apple Silicon GPU (MPS)",
                    memory_total=0,  # MPS doesn't expose memory info
                    memory_available=0,
                    is_available=True
                )
        except ImportError:
            pass
        
        return gpu_info
    
    def to_device(self, model_or_tensor) -> torch.Tensor:
        """Safely move model or tensor to optimal device"""
        try:
            return model_or_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to move to device {self.device}: {e}")
            # Fallback to CPU if device move fails
            if self.device.type != 'cpu':
                logger.info("Falling back to CPU")
                return model_or_tensor.to(torch.device('cpu'))
            return model_or_tensor
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for all devices"""
        memory_info = {}
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu'] = {
            'total_mb': cpu_memory.total // (1024 * 1024),
            'available_mb': cpu_memory.available // (1024 * 1024),
            'used_mb': cpu_memory.used // (1024 * 1024),
            'percent': cpu_memory.percent
        }
        
        # GPU memory
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() // (1024 * 1024)  # MB
                reserved = torch.cuda.memory_reserved() // (1024 * 1024)    # MB
                total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # MB
                
                memory_info['gpu'] = {
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'total_mb': total,
                    'free_mb': total - reserved,
                    'utilization_percent': (allocated / total) * 100 if total > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
        
        return memory_info
    
    def optimize_for_model(self, model_name: str) -> torch.device:
        """Optimize device selection for specific model requirements"""
        
        if "whisper" in model_name.lower():
            # Whisper works well on GPU but can run on CPU
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
            # Default to current device
            return self.device
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            'current_device': str(self.device),
            'device_type': self.device.type,
            'gpu_info': self.gpu_info,
            'memory_usage': self.get_memory_usage(),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self):
        self.memory_history = []
        self.peak_memory = 0
    
    def track_memory(self, stage: str):
        """Track memory usage at different stages"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.peak_memory = max(self.peak_memory, allocated)
            
            self.memory_history.append({
                'stage': stage,
                'allocated_gb': allocated,
                'timestamp': time.time()
            })
            
            logger.info(f"Memory [{stage}]: {allocated:.2f}GB allocated")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        if not self.memory_history:
            return {}
        
        return {
            'peak_memory_gb': self.peak_memory,
            'total_stages': len(self.memory_history),
            'memory_timeline': self.memory_history
        }

# Global device manager instance
device_manager = DeviceManager()

# Convenience functions
def get_optimal_device() -> torch.device:
    """Get the optimal device for ML operations"""
    return device_manager.device

def to_device(model_or_tensor) -> torch.Tensor:
    """Move model or tensor to optimal device"""
    return device_manager.to_device(model_or_tensor)

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    return device_manager.get_device_info()

def clear_gpu_cache():
    """Clear GPU memory cache"""
    device_manager.clear_gpu_cache()

# Example usage
if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Device Manager Test ===")
    print(f"Optimal device: {get_optimal_device()}")
    print(f"Device info: {get_device_info()}")
    
    # Test tensor movement
    test_tensor = torch.randn(1000, 1000)
    moved_tensor = to_device(test_tensor)
    print(f"Tensor device: {moved_tensor.device}")
    
    # Test memory tracking
    device_manager.memory_tracker.track_memory("test_stage")
    print(f"Memory report: {device_manager.memory_tracker.get_memory_report()}")