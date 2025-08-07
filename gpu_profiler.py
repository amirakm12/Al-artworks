import torch
import time
import logging
import psutil
import asyncio
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

log = logging.getLogger("GPUProfiler")

@dataclass
class SystemStats:
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_memory_mb: float
    gpu_memory_percent: float
    gpu_name: str
    timestamp: float

class GPUProfiler:
    def __init__(self):
        self.device = self._get_best_device()
        self.monitoring = False
        self.stats_history = []
        self.max_history = 1000
        self.callbacks = []

    def _get_best_device(self):
        """Get the best available device for AI workloads"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def print_device_info(self):
        """Print comprehensive device information"""
        log.info(f"PyTorch version: {torch.__version__}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        log.info(f"Current device: {self.device}")
        
        if torch.cuda.is_available():
            log.info(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(f"Device {i}: {props.name}")
                log.info(f"  Compute Capability: {props.major}.{props.minor}")
                log.info(f"  Total Memory: {props.total_memory / (1024**3):.1f} GB")
                log.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
                log.info(f"  Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
        
        # CPU info
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total / (1024**3),
            "memory_available": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        log.info(f"CPU Cores: {cpu_info['cpu_count']}")
        log.info(f"CPU Usage: {cpu_info['cpu_percent']:.1f}%")
        log.info(f"RAM Total: {cpu_info['memory_total']:.1f} GB")
        log.info(f"RAM Available: {cpu_info['memory_available']:.1f} GB")
        log.info(f"RAM Usage: {cpu_info['memory_percent']:.1f}%")

    def get_system_stats(self) -> SystemStats:
        """Get current system statistics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # GPU stats
        gpu_memory_mb = 0
        gpu_memory_percent = 0
        gpu_name = "CPU"
        
        if torch.cuda.is_available():
            device = self.device
            if device.type == "cuda":
                gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024**2)
                total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                gpu_memory_percent = (gpu_memory_mb / total_gpu_memory) * 100
                gpu_name = torch.cuda.get_device_name(device)
        
        return SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_name=gpu_name,
            timestamp=time.time()
        )

    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function execution time and memory usage"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            log.error(f"Function {func.__name__} failed: {e}")
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        duration_ms = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        
        log.info(f"Function {func.__name__}:")
        log.info(f"  Duration: {duration_ms:.2f} ms")
        log.info(f"  Memory Delta: {memory_delta:.2f} MB")
        log.info(f"  Success: {success}")
        
        return {
            "function": func.__name__,
            "duration_ms": duration_ms,
            "memory_delta_mb": memory_delta,
            "success": success,
            "result": result
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available() and self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024**2)
        else:
            return psutil.virtual_memory().used / (1024**2)

    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        self.monitoring = True
        log.info(f"Starting system monitoring with {interval}s interval")
        
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                self.stats_history.append(stats)
                
                # Keep history size manageable
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(stats)
                        else:
                            callback(stats)
                    except Exception as e:
                        log.error(f"Callback error: {e}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                log.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        log.info("System monitoring stopped")

    def add_callback(self, callback: Callable[[SystemStats], None]):
        """Add a callback to receive system stats updates"""
        self.callbacks.append(callback)
        log.info(f"Added monitoring callback: {callback.__name__}")

    def remove_callback(self, callback: Callable[[SystemStats], None]):
        """Remove a monitoring callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            log.info(f"Removed monitoring callback: {callback.__name__}")

    def get_stats_history(self, limit: int = 100) -> list[SystemStats]:
        """Get recent system stats history"""
        return self.stats_history[-limit:]

    def get_average_stats(self, window: int = 10) -> Dict[str, float]:
        """Get average stats over a window"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-window:]
        
        return {
            "avg_cpu_percent": sum(s.cpu_percent for s in recent_stats) / len(recent_stats),
            "avg_memory_percent": sum(s.memory_percent for s in recent_stats) / len(recent_stats),
            "avg_gpu_memory_mb": sum(s.gpu_memory_mb for s in recent_stats) / len(recent_stats),
            "avg_gpu_memory_percent": sum(s.gpu_memory_percent for s in recent_stats) / len(recent_stats),
        }

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("GPU cache cleared")

    def benchmark_model_inference(self, model, input_tensor, num_runs: int = 10):
        """Benchmark model inference performance"""
        log.info(f"Benchmarking model inference on {self.device}")
        
        # Warmup
        with torch.no_grad():
            _ = model(input_tensor.to(self.device))
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_tensor.to(self.device))
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        log.info(f"Model inference benchmark results:")
        log.info(f"  Average time: {avg_time:.2f} ms")
        log.info(f"  Min time: {min_time:.2f} ms")
        log.info(f"  Max time: {max_time:.2f} ms")
        log.info(f"  Throughput: {1000/avg_time:.2f} inferences/sec")
        
        return {
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "throughput_fps": 1000/avg_time,
            "device": str(self.device)
        }

# Global profiler instance
profiler = GPUProfiler()

# Utility functions for easy access
def print_device_info():
    """Print device information"""
    profiler.print_device_info()

def get_system_stats():
    """Get current system statistics"""
    return profiler.get_system_stats()

def profile_function(func: Callable, *args, **kwargs):
    """Profile a function execution"""
    return profiler.profile_function(func, *args, **kwargs)

def start_monitoring(interval: float = 1.0):
    """Start system monitoring"""
    return asyncio.create_task(profiler.start_monitoring(interval))

def stop_monitoring():
    """Stop system monitoring"""
    profiler.stop_monitoring()

def add_monitoring_callback(callback: Callable[[SystemStats], None]):
    """Add a monitoring callback"""
    profiler.add_callback(callback)

def clear_gpu_cache():
    """Clear GPU cache"""
    profiler.clear_gpu_cache()

# Example monitoring callback
def example_stats_callback(stats: SystemStats):
    """Example callback for system stats"""
    log.info(f"System Stats - CPU: {stats.cpu_percent:.1f}%, "
             f"RAM: {stats.memory_percent:.1f}%, "
             f"GPU: {stats.gpu_memory_mb:.1f}MB")

# Example usage
async def example_monitoring():
    """Example of using the GPU profiler"""
    logging.basicConfig(level=logging.INFO)
    
    # Print device info
    print_device_info()
    
    # Add monitoring callback
    add_monitoring_callback(example_stats_callback)
    
    # Start monitoring
    monitoring_task = start_monitoring(interval=2.0)
    
    # Run for 10 seconds
    await asyncio.sleep(10)
    
    # Stop monitoring
    stop_monitoring()
    monitoring_task.cancel()
    
    # Get average stats
    avg_stats = profiler.get_average_stats()
    log.info(f"Average stats: {avg_stats}")

if __name__ == "__main__":
    asyncio.run(example_monitoring())