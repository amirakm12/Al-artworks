"""
Performance Tuner - GPU Acceleration and Optimization
Provides comprehensive performance tuning and GPU acceleration for ChatGPT+ Clone
"""

import os
import sys
import logging
import time
import json
import subprocess
import platform
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class PerformanceTuner:
    """Comprehensive performance tuning and GPU acceleration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path("performance_config.json")
        self.performance_stats = {}
        self.gpu_info = {}
        self.optimization_level = "balanced"  # balanced, performance, efficiency
        
        # Load existing config
        self.load_config()
        
        # Initialize performance monitoring
        self.start_monitoring()
    
    def detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities and hardware"""
        capabilities = {
            'cpu': self._detect_cpu(),
            'memory': self._detect_memory(),
            'gpu': self._detect_gpu(),
            'storage': self._detect_storage(),
            'network': self._detect_network(),
            'python': self._detect_python_environment()
        }
        
        self.logger.info("System capabilities detected")
        return capabilities
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information"""
        try:
            cpu_info = {
                'count': os.cpu_count(),
                'physical_count': psutil.cpu_count(logical=False),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'usage': psutil.cpu_percent(interval=1),
                'architecture': platform.machine(),
                'platform': platform.platform()
            }
            
            # Get CPU model on Windows
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            cpu_info['model'] = lines[1].strip()
                except Exception:
                    pass
            
            return cpu_info
        except Exception as e:
            self.logger.error(f"Error detecting CPU: {e}")
            return {}
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'free': memory.free
            }
        except Exception as e:
            self.logger.error(f"Error detecting memory: {e}")
            return {}
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information"""
        gpu_info = {}
        
        # Try to detect NVIDIA GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info['nvidia'] = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_info['nvidia'].append({
                                'name': parts[0],
                                'memory_total': int(parts[1]),
                                'memory_free': int(parts[2]),
                                'utilization': int(parts[3])
                            })
        except Exception:
            pass
        
        # Try to detect AMD GPUs
        try:
            result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info['amd'] = result.stdout
        except Exception:
            pass
        
        # Check PyTorch GPU availability
        if TORCH_AVAILABLE:
            gpu_info['pytorch_cuda'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                gpu_info['pytorch_gpu_count'] = torch.cuda.device_count()
                gpu_info['pytorch_gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        return gpu_info
    
    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage information"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        except Exception as e:
            self.logger.error(f"Error detecting storage: {e}")
            return {}
    
    def _detect_network(self) -> Dict[str, Any]:
        """Detect network information"""
        try:
            network = psutil.net_io_counters()
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except Exception as e:
            self.logger.error(f"Error detecting network: {e}")
            return {}
    
    def _detect_python_environment(self) -> Dict[str, Any]:
        """Detect Python environment"""
        return {
            'version': sys.version,
            'executable': sys.executable,
            'platform': sys.platform,
            'architecture': '64bit' if sys.maxsize > 2**32 else '32bit',
            'torch_available': TORCH_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE
        }
    
    def optimize_for_gpu(self) -> Dict[str, Any]:
        """Optimize settings for GPU acceleration"""
        optimizations = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Set PyTorch optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction for GPU
            gpu_memory_fraction = 0.8
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            optimizations['pytorch'] = {
                'cudnn_benchmark': True,
                'cudnn_deterministic': False,
                'gpu_memory_fraction': gpu_memory_fraction,
                'gpu_count': torch.cuda.device_count()
            }
            
            self.logger.info(f"GPU optimizations applied: {optimizations['pytorch']}")
        
        return optimizations
    
    def optimize_for_cpu(self) -> Dict[str, Any]:
        """Optimize settings for CPU-only operation"""
        optimizations = {}
        
        # Set number of threads for CPU operations
        cpu_count = os.cpu_count()
        if cpu_count:
            # For CPU-intensive tasks
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            
            optimizations['cpu'] = {
                'omp_num_threads': cpu_count,
                'mkl_num_threads': cpu_count,
                'cpu_count': cpu_count
            }
        
        # Disable GPU usage if needed
        if TORCH_AVAILABLE:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            optimizations['gpu_disabled'] = True
        
        self.logger.info(f"CPU optimizations applied: {optimizations}")
        return optimizations
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        optimizations = {}
        
        # Set garbage collection thresholds
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
        
        # Set memory limits for numpy if available
        if NUMPY_AVAILABLE:
            np.set_printoptions(precision=4, suppress=True)
        
        # Set PyTorch memory optimizations
        if TORCH_AVAILABLE:
            torch.backends.cudnn.benchmark = False  # Save memory
            torch.backends.cudnn.deterministic = True
        
        optimizations['memory'] = {
            'gc_threshold': (700, 10, 10),
            'cudnn_benchmark': False,
            'cudnn_deterministic': True
        }
        
        self.logger.info(f"Memory optimizations applied: {optimizations}")
        return optimizations
    
    def set_optimization_level(self, level: str):
        """Set optimization level"""
        if level not in ["balanced", "performance", "efficiency"]:
            level = "balanced"
        
        self.optimization_level = level
        
        if level == "performance":
            optimizations = self.optimize_for_gpu()
        elif level == "efficiency":
            optimizations = self.optimize_memory_usage()
        else:  # balanced
            optimizations = self.optimize_for_gpu()
            memory_opt = self.optimize_memory_usage()
            optimizations.update(memory_opt)
        
        self.save_config()
        return optimizations
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        benchmarks = {}
        
        # CPU benchmark
        benchmarks['cpu'] = self._benchmark_cpu()
        
        # Memory benchmark
        benchmarks['memory'] = self._benchmark_memory()
        
        # GPU benchmark (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            benchmarks['gpu'] = self._benchmark_gpu()
        
        # Storage benchmark
        benchmarks['storage'] = self._benchmark_storage()
        
        self.performance_stats = benchmarks
        self.save_config()
        
        return benchmarks
    
    def _benchmark_cpu(self) -> Dict[str, Any]:
        """Benchmark CPU performance"""
        start_time = time.time()
        
        # Simple CPU benchmark
        result = 0
        for i in range(1000000):
            result += i * i
        
        execution_time = time.time() - start_time
        
        return {
            'execution_time': execution_time,
            'operations_per_second': 1000000 / execution_time,
            'cpu_usage': psutil.cpu_percent(interval=1)
        }
    
    def _benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory performance"""
        start_time = time.time()
        
        # Memory allocation benchmark
        if NUMPY_AVAILABLE:
            # Allocate large array
            array = np.random.random((1000, 1000))
            memory_time = time.time() - start_time
            
            # Memory access benchmark
            access_start = time.time()
            _ = np.sum(array)
            access_time = time.time() - access_start
            
            return {
                'allocation_time': memory_time,
                'access_time': access_time,
                'total_time': time.time() - start_time,
                'memory_usage': psutil.virtual_memory().percent
            }
        else:
            # Fallback memory benchmark
            data = []
            for i in range(100000):
                data.append(i)
            
            return {
                'execution_time': time.time() - start_time,
                'memory_usage': psutil.virtual_memory().percent
            }
    
    def _benchmark_gpu(self) -> Dict[str, Any]:
        """Benchmark GPU performance"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        start_time = time.time()
        
        # GPU tensor operations benchmark
        device = torch.device('cuda')
        tensor = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        torch.cuda.synchronize()
        matmul_start = time.time()
        result = torch.mm(tensor, tensor)
        torch.cuda.synchronize()
        matmul_time = time.time() - matmul_start
        
        # Memory transfer benchmark
        transfer_start = time.time()
        cpu_tensor = tensor.cpu()
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start
        
        return {
            'matmul_time': matmul_time,
            'transfer_time': transfer_time,
            'total_time': time.time() - start_time,
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**2  # MB
        }
    
    def _benchmark_storage(self) -> Dict[str, Any]:
        """Benchmark storage performance"""
        start_time = time.time()
        
        # Write benchmark
        test_file = Path("storage_benchmark.tmp")
        data = b"x" * 1024 * 1024  # 1MB
        
        write_start = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(10):  # Write 10MB
                f.write(data)
        write_time = time.time() - write_start
        
        # Read benchmark
        read_start = time.time()
        with open(test_file, 'rb') as f:
            for _ in range(10):
                f.read(1024 * 1024)
        read_time = time.time() - read_start
        
        # Cleanup
        test_file.unlink()
        
        return {
            'write_time': write_time,
            'read_time': read_time,
            'write_speed': 10 / write_time,  # MB/s
            'read_speed': 10 / read_time,    # MB/s
            'total_time': time.time() - start_time
        }
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations based on system capabilities"""
        capabilities = self.detect_system_capabilities()
        recommendations = {
            'optimization_level': 'balanced',
            'gpu_usage': False,
            'memory_optimization': True,
            'storage_optimization': False,
            'network_optimization': False
        }
        
        # GPU recommendations
        if capabilities.get('gpu', {}).get('pytorch_cuda', False):
            recommendations['gpu_usage'] = True
            recommendations['optimization_level'] = 'performance'
        
        # Memory recommendations
        memory = capabilities.get('memory', {})
        if memory.get('percent', 0) > 80:
            recommendations['memory_optimization'] = True
            recommendations['optimization_level'] = 'efficiency'
        
        # Storage recommendations
        storage = capabilities.get('storage', {})
        if storage.get('percent', 0) > 90:
            recommendations['storage_optimization'] = True
        
        # Network recommendations
        network = capabilities.get('network', {})
        if network.get('bytes_recv', 0) > 1024**3:  # 1GB
            recommendations['network_optimization'] = True
        
        return recommendations
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        self.logger.info("Performance monitoring stopped")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'gpu_usage': self._get_gpu_usage() if TORCH_AVAILABLE else {}
        }
    
    def _get_gpu_usage(self) -> Dict[str, Any]:
        """Get GPU usage information"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        return {
            'memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'memory_reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device()
        }
    
    def load_config(self):
        """Load performance configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.optimization_level = config.get('optimization_level', 'balanced')
                    self.performance_stats = config.get('performance_stats', {})
                    self.gpu_info = config.get('gpu_info', {})
        except Exception as e:
            self.logger.error(f"Error loading performance config: {e}")
    
    def save_config(self):
        """Save performance configuration"""
        try:
            config = {
                'optimization_level': self.optimization_level,
                'performance_stats': self.performance_stats,
                'gpu_info': self.gpu_info,
                'timestamp': time.time()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving performance config: {e}")

# Convenience functions
def optimize_system(level: str = "balanced") -> Dict[str, Any]:
    """Optimize system for specified level"""
    tuner = PerformanceTuner()
    return tuner.set_optimization_level(level)

def benchmark_system() -> Dict[str, Any]:
    """Run comprehensive system benchmarks"""
    tuner = PerformanceTuner()
    return tuner.benchmark_performance()

def get_system_recommendations() -> Dict[str, Any]:
    """Get performance recommendations"""
    tuner = PerformanceTuner()
    return tuner.get_recommendations()

if __name__ == "__main__":
    # Test the performance tuner
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Performance Tuner...")
    
    tuner = PerformanceTuner()
    
    # Detect system capabilities
    capabilities = tuner.detect_system_capabilities()
    print(f"✅ System capabilities: {capabilities}")
    
    # Get recommendations
    recommendations = tuner.get_recommendations()
    print(f"✅ Recommendations: {recommendations}")
    
    # Run benchmarks
    benchmarks = tuner.benchmark_performance()
    print(f"✅ Benchmarks: {benchmarks}")
    
    # Optimize system
    optimizations = tuner.set_optimization_level("balanced")
    print(f"✅ Optimizations: {optimizations}")