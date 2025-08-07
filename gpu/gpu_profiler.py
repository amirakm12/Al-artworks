import torch
import time
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

log = logging.getLogger("GPUProfiler")

@dataclass
class BenchmarkResult:
    operation: str
    device: str
    duration: float
    iterations: int
    throughput: float
    memory_used: float

class GPUProfiler:
    def __init__(self):
        self.device = self._get_best_device()
        self.benchmark_results: List[BenchmarkResult] = []
        
    def _get_best_device(self) -> torch.device:
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

    def simple_gpu_benchmark(self, matrix_size: int = 1024, iterations: int = 100) -> BenchmarkResult:
        """Run a simple GPU benchmark with matrix multiplication"""
        if not torch.cuda.is_available():
            log.warning("CUDA not available, running CPU benchmark instead.")
            device = torch.device("cpu")
        else:
            device = self.device
        
        log.info(f"Running matrix multiplication benchmark on {device}")
        log.info(f"Matrix size: {matrix_size}x{matrix_size}, Iterations: {iterations}")
        
        # Create test tensors
        x = torch.randn((matrix_size, matrix_size), device=device)
        
        # Warmup
        for _ in range(10):
            _ = x @ x
        
        # Synchronize if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            y = x @ x
            if device.type == "cuda":
                torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        # Memory usage
        if device.type == "cuda":
            memory_used = torch.cuda.memory_allocated(device) / (1024**2)
        else:
            memory_used = 0
        
        result = BenchmarkResult(
            operation="matrix_multiplication",
            device=str(device),
            duration=duration,
            iterations=iterations,
            throughput=throughput,
            memory_used=memory_used
        )
        
        self.benchmark_results.append(result)
        
        log.info(f"Benchmark completed:")
        log.info(f"  Duration: {duration:.4f} seconds")
        log.info(f"  Throughput: {throughput:.2f} ops/sec")
        log.info(f"  Memory Used: {memory_used:.2f} MB")
        
        return result

    def model_inference_benchmark(self, model, input_tensor, num_runs: int = 10) -> BenchmarkResult:
        """Benchmark model inference performance"""
        device = self.device
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        log.info(f"Benchmarking model inference on {device}")
        
        # Warmup
        with torch.no_grad():
            _ = model(input_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_runs / duration
        
        # Memory usage
        if device.type == "cuda":
            memory_used = torch.cuda.memory_allocated(device) / (1024**2)
        else:
            memory_used = 0
        
        result = BenchmarkResult(
            operation="model_inference",
            device=str(device),
            duration=duration,
            iterations=num_runs,
            throughput=throughput,
            memory_used=memory_used
        )
        
        self.benchmark_results.append(result)
        
        log.info(f"Model inference benchmark completed:")
        log.info(f"  Duration: {duration:.4f} seconds")
        log.info(f"  Throughput: {throughput:.2f} inferences/sec")
        log.info(f"  Memory Used: {memory_used:.2f} MB")
        
        return result

    def memory_benchmark(self, tensor_sizes: List[int] = [1024, 2048, 4096]) -> List[BenchmarkResult]:
        """Benchmark memory allocation and deallocation"""
        if not torch.cuda.is_available():
            log.warning("CUDA not available, skipping memory benchmark.")
            return []
        
        results = []
        device = self.device
        
        for size in tensor_sizes:
            log.info(f"Testing memory allocation for {size}x{size} tensor")
            
            # Clear cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
            
            start_time = time.time()
            
            # Allocate tensor
            tensor = torch.randn((size, size), device=device)
            torch.cuda.synchronize()
            
            allocation_time = time.time() - start_time
            peak_memory = torch.cuda.memory_allocated(device)
            
            # Deallocate
            del tensor
            torch.cuda.empty_cache()
            
            deallocation_time = time.time() - start_time - allocation_time
            final_memory = torch.cuda.memory_allocated(device)
            
            result = BenchmarkResult(
                operation=f"memory_allocation_{size}x{size}",
                device=str(device),
                duration=allocation_time,
                iterations=1,
                throughput=1.0 / allocation_time if allocation_time > 0 else 0,
                memory_used=peak_memory / (1024**2)
            )
            
            results.append(result)
            self.benchmark_results.append(result)
            
            log.info(f"  Allocation time: {allocation_time:.4f}s")
            log.info(f"  Peak memory: {peak_memory / (1024**2):.2f} MB")
            log.info(f"  Memory overhead: {(peak_memory - initial_memory) / (1024**2):.2f} MB")
        
        return results

    def mixed_precision_benchmark(self, model, input_tensor, num_runs: int = 10) -> Dict[str, BenchmarkResult]:
        """Benchmark different precision modes"""
        device = self.device
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        results = {}
        
        # FP32 benchmark
        model_fp32 = model.float()
        log.info("Benchmarking FP32 precision")
        results["fp32"] = self.model_inference_benchmark(model_fp32, input_tensor, num_runs)
        
        # FP16 benchmark (if supported)
        if device.type == "cuda":
            try:
                model_fp16 = model.half()
                input_fp16 = input_tensor.half()
                log.info("Benchmarking FP16 precision")
                results["fp16"] = self.model_inference_benchmark(model_fp16, input_fp16, num_runs)
            except Exception as e:
                log.warning(f"FP16 benchmark failed: {e}")
        
        # BF16 benchmark (if supported)
        if device.type == "cuda" and hasattr(torch, "bfloat16"):
            try:
                model_bf16 = model.bfloat16()
                input_bf16 = input_tensor.bfloat16()
                log.info("Benchmarking BF16 precision")
                results["bf16"] = self.model_inference_benchmark(model_bf16, input_bf16, num_runs)
            except Exception as e:
                log.warning(f"BF16 benchmark failed: {e}")
        
        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {
            "cpu": {
                "memory_total": psutil.virtual_memory().total / (1024**3),
                "memory_available": psutil.virtual_memory().available / (1024**3),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent()
            }
        }
        
        if torch.cuda.is_available():
            device = self.device
            stats["gpu"] = {
                "device_name": torch.cuda.get_device_name(device),
                "memory_allocated": torch.cuda.memory_allocated(device) / (1024**2),
                "memory_reserved": torch.cuda.memory_reserved(device) / (1024**2),
                "memory_total": torch.cuda.get_device_properties(device).total_memory / (1024**2),
                "memory_percent": (torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory) * 100
            }
        
        return stats

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("GPU cache cleared")

    def print_benchmark_summary(self):
        """Print summary of all benchmark results"""
        if not self.benchmark_results:
            log.info("No benchmark results available")
            return
        
        log.info("=== Benchmark Summary ===")
        for result in self.benchmark_results:
            log.info(f"{result.operation} on {result.device}:")
            log.info(f"  Duration: {result.duration:.4f}s")
            log.info(f"  Throughput: {result.throughput:.2f} ops/sec")
            log.info(f"  Memory: {result.memory_used:.2f} MB")
            log.info("")

    def export_benchmark_results(self, filename: str = "benchmark_results.json"):
        """Export benchmark results to JSON file"""
        import json
        
        results_data = []
        for result in self.benchmark_results:
            results_data.append({
                "operation": result.operation,
                "device": result.device,
                "duration": result.duration,
                "iterations": result.iterations,
                "throughput": result.throughput,
                "memory_used": result.memory_used
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        log.info(f"Benchmark results exported to {filename}")

# Utility functions for easy access
def print_device_info():
    """Print device information"""
    profiler = GPUProfiler()
    profiler.print_device_info()

def simple_gpu_benchmark():
    """Run simple GPU benchmark"""
    profiler = GPUProfiler()
    return profiler.simple_gpu_benchmark()

def get_memory_stats():
    """Get current memory statistics"""
    profiler = GPUProfiler()
    return profiler.get_memory_stats()

def clear_gpu_cache():
    """Clear GPU cache"""
    profiler = GPUProfiler()
    profiler.clear_gpu_cache()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    profiler = GPUProfiler()
    
    # Print device information
    profiler.print_device_info()
    
    # Run benchmarks
    profiler.simple_gpu_benchmark()
    
    # Print memory stats
    memory_stats = profiler.get_memory_stats()
    log.info(f"Memory stats: {memory_stats}")
    
    # Print summary
    profiler.print_benchmark_summary()
    
    # Export results
    profiler.export_benchmark_results()