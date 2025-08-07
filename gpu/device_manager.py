import torch
import logging
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("DeviceManager")

@dataclass
class GPUInfo:
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    is_available: bool = True

class DeviceManager:
    def __init__(self):
        self.device = self.detect_optimal_device()
        self.gpu_info = self._get_gpu_info()
        self.memory_tracker = MemoryTracker()
        logger.info(f"DeviceManager initialized on device: {self.device}")
        logger.info(f"GPU Info: {self.gpu_info}")

    def detect_optimal_device(self) -> torch.device:
        """Detect and select the best available device for AI workloads"""
        # Priority: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected")
            return self._select_best_cuda_device()
        
        # Try Apple MPS GPU if on macOS
        try:
            import torch.backends.mps
            if torch.backends.mps.is_available():
                logger.info("Apple MPS GPU detected")
                return torch.device("mps")
        except ImportError:
            logger.info("MPS backend not available")
        
        # Fallback to CPU
        logger.info("Falling back to CPU")
        return torch.device("cpu")

    def _select_best_cuda_device(self) -> torch.device:
        """Select the best CUDA GPU based on memory and compute capability"""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return torch.device("cpu")
        
        best_device = 0
        best_score = 0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # Score based on memory and compute capability
            memory_gb = props.total_memory / (1024**3)
            compute_cap = float(props.major) + float(props.minor) / 10
            
            # Simple scoring: prioritize memory, then compute capability
            score = memory_gb * 0.7 + compute_cap * 0.3
            
            if score > best_score:
                best_score = score
                best_device = i
        
        logger.info(f"Selected CUDA device {best_device}: {torch.cuda.get_device_name(best_device)}")
        return torch.device(f"cuda:{best_device}")

    def _get_gpu_info(self) -> GPUInfo:
        """Get comprehensive GPU information"""
        if not torch.cuda.is_available():
            return GPUInfo(
                name="CPU",
                memory_total=psutil.virtual_memory().total // (1024**2),
                memory_available=psutil.virtual_memory().available // (1024**2),
                is_available=True
            )
        
        device = self.device
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            memory_allocated = torch.cuda.memory_allocated(device) // (1024**2)
            memory_reserved = torch.cuda.memory_reserved(device) // (1024**2)
            
            return GPUInfo(
                name=props.name,
                memory_total=props.total_memory // (1024**2),
                memory_available=props.total_memory // (1024**2) - memory_reserved,
                compute_capability=f"{props.major}.{props.minor}",
                is_available=True
            )
        elif device.type == "mps":
            return GPUInfo(
                name="Apple MPS",
                memory_total=0,  # MPS doesn't expose memory info easily
                memory_available=0,
                is_available=True
            )
        
        return GPUInfo(
            name="CPU",
            memory_total=psutil.virtual_memory().total // (1024**2),
            memory_available=psutil.virtual_memory().available // (1024**2),
            is_available=True
        )

    def to_device(self, model_or_tensor) -> torch.Tensor:
        """Safely move model or tensor to the selected device"""
        try:
            return model_or_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to move to device {self.device}: {e}")
            # Fallback to CPU if device move fails
            if self.device.type != "cpu":
                logger.warning("Falling back to CPU")
                return model_or_tensor.to("cpu")
            return model_or_tensor

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for CPU and GPU"""
        cpu_memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            device = self.device
            if device.type == "cuda":
                gpu_memory = {
                    "allocated_mb": torch.cuda.memory_allocated(device) // (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device) // (1024**2),
                    "total_mb": torch.cuda.get_device_properties(device).total_memory // (1024**2)
                }
        
        return {
            "cpu": {
                "total_mb": cpu_memory.total // (1024**2),
                "available_mb": cpu_memory.available // (1024**2),
                "percent": cpu_memory.percent
            },
            "gpu": gpu_memory
        }

    def optimize_for_model(self, model_name: str) -> torch.device:
        """Optimize device selection for specific model requirements"""
        # Model-specific optimizations
        if "whisper" in model_name.lower():
            # Whisper works well on CPU for smaller models
            if "base" in model_name.lower() or "small" in model_name.lower():
                logger.info("Using CPU for small Whisper model")
                return torch.device("cpu")
        
        return self.device

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return {
            "current_device": str(self.device),
            "gpu_info": self.gpu_info.__dict__,
            "memory_usage": self.get_memory_usage(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self):
        self.memory_history = []
        self.peak_memory = 0
    
    def record_memory(self):
        """Record current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() // (1024**2)
            self.memory_history.append(allocated)
            self.peak_memory = max(self.peak_memory, allocated)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.memory_history:
            return {"current": 0, "peak": 0, "average": 0}
        
        return {
            "current": self.memory_history[-1] if self.memory_history else 0,
            "peak": self.peak_memory,
            "average": sum(self.memory_history) / len(self.memory_history)
        }

# Singleton instance for global use
device_manager = DeviceManager()