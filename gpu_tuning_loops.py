import asyncio
import torch
import logging
import time
import psutil
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

log = logging.getLogger("GPUTuner")

@dataclass
class GPUStats:
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float]
    power_watts: Optional[float]
    timestamp: float

class GPUTuner:
    """Dynamic GPU load balancing and async profiling system"""
    
    def __init__(self, device: Optional[torch.device] = None, 
                 target_util: float = 0.75, check_interval: float = 5.0,
                 min_batch_size: int = 1, max_batch_size: int = 32):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.target_util = target_util
        self.check_interval = check_interval
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size = min_batch_size
        self.running = False
        self.stats_history = []
        self.max_history = 100
        self.callbacks = []
        
        # Tuning parameters
        self.utilization_threshold_high = target_util + 0.1
        self.utilization_threshold_low = target_util - 0.1
        self.memory_threshold_high = 0.85  # 85% memory usage
        self.memory_threshold_low = 0.50   # 50% memory usage

    async def _profile_utilization(self) -> GPUStats:
        """Get comprehensive GPU utilization statistics"""
        try:
            if self.device.type == "cuda":
                # Memory stats
                memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
                memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
                
                # Utilization (if available)
                try:
                    utilization = torch.cuda.utilization()
                except AttributeError:
                    # Fallback: estimate utilization based on memory usage
                    utilization = (memory_allocated / memory_total) * 100
                
                # Temperature and power (if available via nvidia-smi)
                temperature = None
                power = None
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw', 
                                          '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        temp_str, power_str = result.stdout.strip().split(', ')
                        temperature = float(temp_str)
                        power = float(power_str)
                except:
                    pass
                
                stats = GPUStats(
                    memory_allocated_mb=memory_allocated,
                    memory_reserved_mb=memory_reserved,
                    memory_total_mb=memory_total,
                    utilization_percent=utilization,
                    temperature_celsius=temperature,
                    power_watts=power,
                    timestamp=time.time()
                )
                
            else:
                # CPU fallback
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                stats = GPUStats(
                    memory_allocated_mb=memory.used / (1024**2),
                    memory_reserved_mb=memory.used / (1024**2),
                    memory_total_mb=memory.total / (1024**2),
                    utilization_percent=cpu_percent,
                    temperature_celsius=None,
                    power_watts=None,
                    timestamp=time.time()
                )
            
            return stats
            
        except Exception as e:
            log.warning(f"Failed GPU utilization check: {e}")
            return GPUStats(0, 0, 0, 0, None, None, time.time())

    async def _adjust_batch_size(self, stats: GPUStats):
        """Dynamically adjust batch size based on GPU utilization"""
        old_batch_size = self.batch_size
        
        # Check memory pressure
        memory_util = stats.memory_allocated_mb / stats.memory_total_mb
        utilization = stats.utilization_percent / 100.0
        
        # Decrease batch size if utilization is too high
        if (utilization > self.utilization_threshold_high or 
            memory_util > self.memory_threshold_high):
            self.batch_size = max(self.min_batch_size, self.batch_size - 1)
            log.info(f"Decreased batch size to {self.batch_size} (util: {utilization:.2f}, mem: {memory_util:.2f})")
        
        # Increase batch size if utilization is too low
        elif (utilization < self.utilization_threshold_low and 
              memory_util < self.memory_threshold_low):
            self.batch_size = min(self.max_batch_size, self.batch_size + 1)
            log.info(f"Increased batch size to {self.batch_size} (util: {utilization:.2f}, mem: {memory_util:.2f})")
        
        if old_batch_size != self.batch_size:
            log.info(f"Batch size adjusted: {old_batch_size} -> {self.batch_size}")

    async def _notify_callbacks(self, stats: GPUStats):
        """Notify all registered callbacks with GPU stats"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stats)
                else:
                    callback(stats)
            except Exception as e:
                log.error(f"Error in GPU stats callback: {e}")

    async def tuning_loop(self):
        """Main GPU tuning loop"""
        self.running = True
        log.info(f"Starting GPU tuning loop on device {self.device}")
        log.info(f"Target utilization: {self.target_util}, Check interval: {self.check_interval}s")
        
        while self.running:
            try:
                # Get current GPU stats
                stats = await self._profile_utilization()
                
                # Store in history
                self.stats_history.append(stats)
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)
                
                # Adjust batch size based on utilization
                await self._adjust_batch_size(stats)
                
                # Notify callbacks
                await self._notify_callbacks(stats)
                
                # Log stats periodically
                log.debug(f"GPU Stats - Util: {stats.utilization_percent:.1f}%, "
                         f"Mem: {stats.memory_allocated_mb:.1f}MB/{stats.memory_total_mb:.1f}MB, "
                         f"Batch: {self.batch_size}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                log.error(f"Error in GPU tuning loop: {e}")
                await asyncio.sleep(self.check_interval)

    def add_callback(self, callback: Callable[[GPUStats], None]):
        """Add a callback to receive GPU stats updates"""
        self.callbacks.append(callback)
        log.info(f"Added GPU stats callback: {callback.__name__}")

    def remove_callback(self, callback: Callable[[GPUStats], None]):
        """Remove a GPU stats callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            log.info(f"Removed GPU stats callback: {callback.__name__}")

    def get_current_batch_size(self) -> int:
        """Get current optimized batch size"""
        return self.batch_size

    def set_batch_size(self, batch_size: int):
        """Manually set batch size"""
        self.batch_size = max(self.min_batch_size, min(self.max_batch_size, batch_size))
        log.info(f"Manually set batch size to {self.batch_size}")

    def get_stats_history(self, limit: int = 50) -> list[GPUStats]:
        """Get recent GPU stats history"""
        return self.stats_history[-limit:]

    def get_average_stats(self, window: int = 10) -> Dict[str, float]:
        """Get average GPU stats over a window"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-window:]
        
        return {
            "avg_utilization": sum(s.utilization_percent for s in recent_stats) / len(recent_stats),
            "avg_memory_allocated": sum(s.memory_allocated_mb for s in recent_stats) / len(recent_stats),
            "avg_memory_utilization": sum(s.memory_allocated_mb / s.memory_total_mb for s in recent_stats) / len(recent_stats),
            "avg_temperature": sum(s.temperature_celsius for s in recent_stats if s.temperature_celsius) / max(len([s for s in recent_stats if s.temperature_celsius]), 1),
            "current_batch_size": self.batch_size
        }

    def stop(self):
        """Stop the GPU tuning loop"""
        self.running = False
        log.info("GPU tuning loop stopped")

    async def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            log.info("GPU memory cache cleared")

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "target_utilization": self.target_util,
            "current_batch_size": self.batch_size,
            "batch_size_limits": (self.min_batch_size, self.max_batch_size)
        }
        
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            info.update({
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3),
                "multi_processor_count": props.multi_processor_count
            })
        
        return info

# Example GPU stats callback
def example_gpu_callback(stats: GPUStats):
    """Example callback for GPU stats"""
    log.info(f"GPU Stats - Util: {stats.utilization_percent:.1f}%, "
             f"Mem: {stats.memory_allocated_mb:.1f}MB, "
             f"Temp: {stats.temperature_celsius}°C")

# Example usage
async def example_gpu_tuning():
    """Example of using the GPU tuner"""
    logging.basicConfig(level=logging.INFO)
    
    # Create GPU tuner
    tuner = GPUTuner(target_util=0.75, check_interval=2.0)
    
    # Add callback
    tuner.add_callback(example_gpu_callback)
    
    # Start tuning loop
    tuning_task = asyncio.create_task(tuner.tuning_loop())
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Get final stats
    avg_stats = tuner.get_average_stats()
    log.info(f"Average GPU stats: {avg_stats}")
    
    # Stop tuner
    tuner.stop()
    tuning_task.cancel()

if __name__ == "__main__":
    asyncio.run(example_gpu_tuning())