import torch
import time
import logging
import psutil
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger("GPUProfiler")

@dataclass
class GPUInfo:
    name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    is_available: bool = True

class GPUProfiler:
    def __init__(self):
        self.device = self._get_best_device()
        self.benchmark_results = {}
        
    def _get_best_device(self) -> torch.device:
        """Get the best available device for AI workloads"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_device_info(self) -> GPUInfo:
        """Get comprehensive device information"""
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

    def simple_gpu_benchmark(self, device: Optional[torch.device] = None, 
                           duration: float = 5.0, matrix_size: int = 2048) -> Tuple[int, float]:
        """
        Run a simple GPU benchmark with matrix multiplication
        
        Args:
            device: torch device to benchmark (defaults to best available)
            duration: benchmark duration in seconds
            matrix_size: size of matrices for multiplication
            
        Returns:
            Tuple of (iterations, elapsed_time)
        """
        device = device or self.device
        log.info(f"Running GPU benchmark on {device} for {duration}s")
        
        start_time = time.time()
        iterations = 0
        
        try:
            while (time.time() - start_time) < duration:
                # Create large tensors for matrix multiplication
                a = torch.randn(matrix_size, matrix_size, device=device)
                b = torch.randn(matrix_size, matrix_size, device=device)
                
                # Perform matrix multiplication
                c = torch.mm(a, b)
                
                # Synchronize if using CUDA
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                iterations += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
                
        except Exception as e:
            log.error(f"Benchmark error: {e}")
            return 0, 0
        
        elapsed = time.time() - start_time
        ops_per_second = iterations / elapsed if elapsed > 0 else 0
        
        log.info(f"Benchmark finished: {iterations} iterations in {elapsed:.2f} seconds")
        log.info(f"Operations per second: {ops_per_second:.2f}")
        
        # Store results
        self.benchmark_results[f"matrix_mult_{matrix_size}"] = {
            "iterations": iterations,
            "elapsed_time": elapsed,
            "ops_per_second": ops_per_second,
            "device": str(device)
        }
        
        return iterations, elapsed

    def profile_gpu_memory(self) -> Dict[str, Any]:
        """Profile current GPU memory usage"""
        if not torch.cuda.is_available():
            log.warning("CUDA device not available for memory profiling")
            return {
                "cuda_available": False,
                "memory_allocated_mb": 0,
                "memory_reserved_mb": 0,
                "memory_total_mb": 0
            }
        
        log.info("Profiling GPU memory usage")
        
        memory_info = {
            "cuda_available": True,
            "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2)
        }
        
        log.info(f"GPU Memory - Allocated: {memory_info['memory_allocated_mb']:.2f} MB")
        log.info(f"GPU Memory - Reserved: {memory_info['memory_reserved_mb']:.2f} MB")
        log.info(f"GPU Memory - Total: {memory_info['memory_total_mb']:.2f} MB")
        
        return memory_info

    def print_device_info(self):
        """Print comprehensive device information"""
        device_info = self.get_device_info()
        
        log.info("=== Device Information ===")
        log.info(f"Device: {self.device}")
        log.info(f"Device Name: {device_info.name}")
        log.info(f"Memory Total: {device_info.memory_total} MB")
        log.info(f"Memory Available: {device_info.memory_available} MB")
        
        if torch.cuda.is_available():
            log.info(f"CUDA Version: {torch.version.cuda}")
            log.info(f"GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(f"GPU {i}: {props.name}")
                log.info(f"  Compute Capability: {props.major}.{props.minor}")
                log.info(f"  Total Memory: {props.total_memory / (1024**3):.1f} GB")
        else:
            log.info("CUDA not available")
            
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Apple MPS (Metal) available")
        else:
            log.info("Apple MPS not available")

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("GPU cache cleared")
        else:
            log.info("No GPU cache to clear")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for CPU and GPU"""
        cpu_memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            device = self.device
            if device.type == "cuda":
                gpu_memory = {
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
                    "total_mb": torch.cuda.get_device_properties(device).total_memory / (1024**2)
                }
        
        return {
            "cpu": {
                "total_mb": cpu_memory.total / (1024**2),
                "available_mb": cpu_memory.available / (1024**2),
                "percent": cpu_memory.percent
            },
            "gpu": gpu_memory
        }

    def benchmark_model_inference(self, model, input_tensor, num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark model inference performance
        
        Args:
            model: PyTorch model to benchmark
            input_tensor: Input tensor for the model
            num_runs: Number of inference runs to average
            
        Returns:
            Dictionary with benchmark results
        """
        device = self.device
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        log.info(f"Benchmarking model inference on {device}")
        
        # Warmup run
        with torch.no_grad():
            _ = model(input_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results = {
            "device": str(device),
            "num_runs": num_runs,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "throughput_fps": 1.0 / avg_time if avg_time > 0 else 0
        }
        
        log.info(f"Model inference benchmark results:")
        log.info(f"  Average time: {results['avg_time_ms']:.2f} ms")
        log.info(f"  Throughput: {results['throughput_fps']:.2f} FPS")
        
        return results

    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get all benchmark results"""
        return self.benchmark_results

    def print_summary(self):
        """Print a comprehensive system summary"""
        log.info("=== GPU Profiler Summary ===")
        self.print_device_info()
        
        memory_usage = self.get_memory_usage()
        log.info(f"CPU Memory Usage: {memory_usage['cpu']['percent']:.1f}%")
        
        if memory_usage['gpu']:
            gpu = memory_usage['gpu']
            log.info(f"GPU Memory Usage: {gpu['allocated_mb']:.1f} MB allocated")
        
        if self.benchmark_results:
            log.info("=== Benchmark Results ===")
            for name, results in self.benchmark_results.items():
                log.info(f"{name}: {results['ops_per_second']:.2f} ops/sec")

# Utility functions for easy access
def simple_gpu_benchmark(device=None, duration=5.0):
    """Simple GPU benchmark function"""
    profiler = GPUProfiler()
    return profiler.simple_gpu_benchmark(device, duration)

def profile_gpu_memory():
    """Profile GPU memory usage"""
    profiler = GPUProfiler()
    return profiler.profile_gpu_memory()

def print_device_info():
    """Print device information"""
    profiler = GPUProfiler()
    profiler.print_device_info()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    profiler = GPUProfiler()
    
    # Print device information
    profiler.print_device_info()
    
    # Profile memory
    profiler.profile_gpu_memory()
    
    # Run benchmark
    profiler.simple_gpu_benchmark(duration=3.0)
    
    # Print summary
    profiler.print_summary()