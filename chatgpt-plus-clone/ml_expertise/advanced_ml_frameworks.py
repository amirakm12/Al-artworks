"""
Advanced ML Frameworks Expertise
Deep dive into PyTorch, TensorFlow, and modern ML techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
import gc

logger = logging.getLogger(__name__)

class AdvancedMLFramework:
    """Comprehensive ML framework expertise and best practices"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.memory_tracker = MemoryTracker()
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"ML Framework initialized on device: {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Advanced device selection with memory optimization"""
        if torch.cuda.is_available():
            # Check GPU memory and select best GPU
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory.append((i, memory))
            
            # Select GPU with most memory
            best_gpu = max(gpu_memory, key=lambda x: x[1])[0]
            device = torch.device(f'cuda:{best_gpu}')
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info(f"Selected GPU {best_gpu} with {gpu_memory[best_gpu][1] // 1024**3}GB memory")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        return device

class MemoryTracker:
    """Advanced memory management for ML models"""
    
    def __init__(self):
        self.memory_history = []
        self.peak_memory = 0
    
    def track_memory(self, stage: str):
        """Track memory usage at different stages"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            self.peak_memory = max(self.peak_memory, allocated)
            
            self.memory_history.append({
                'stage': stage,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'timestamp': time.time()
            })
            
            logger.info(f"Memory [{stage}]: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def clear_cache(self):
        """Clear GPU cache and free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        if not self.memory_history:
            return {}
        
        return {
            'peak_memory_gb': self.peak_memory,
            'total_stages': len(self.memory_history),
            'memory_timeline': self.memory_history,
            'average_memory_gb': np.mean([m['allocated_gb'] for m in self.memory_history])
        }

class PerformanceMonitor:
    """Advanced performance monitoring for ML operations"""
    
    def __init__(self):
        self.operation_times = {}
        self.batch_times = []
    
    def time_operation(self, operation_name: str):
        """Decorator to time ML operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = []
                self.operation_times[operation_name].append(duration)
                
                logger.info(f"Operation '{operation_name}' took {duration:.4f}s")
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report"""
        report = {}
        for op_name, times in self.operation_times.items():
            report[op_name] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
        return report

class AdvancedModelOptimizer:
    """Advanced model optimization techniques"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.original_state = None
    
    def optimize_for_inference(self):
        """Optimize model for inference"""
        # Enable eval mode
        self.model.eval()
        
        # Use torch.jit for optimization
        if hasattr(self.model, 'forward'):
            try:
                self.model = torch.jit.optimize_for_inference(self.model)
                logger.info("Model optimized with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        # Use mixed precision if available
        if self.device.type == 'cuda':
            self.model = self.model.half()
            logger.info("Model converted to half precision")
    
    def quantize_model(self, quantization_type: str = 'dynamic'):
        """Quantize model for reduced memory usage"""
        if quantization_type == 'dynamic':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            logger.info("Model quantized with dynamic quantization")
        elif quantization_type == 'static':
            # Requires calibration data
            logger.info("Static quantization requires calibration data")
    
    def create_model_snapshot(self):
        """Create a snapshot of current model state"""
        self.original_state = self.model.state_dict().copy()
        logger.info("Model snapshot created")
    
    def restore_model_snapshot(self):
        """Restore model to snapshot state"""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            logger.info("Model restored from snapshot")

class AdvancedDataLoader:
    """Advanced data loading and preprocessing"""
    
    def __init__(self, dataset: Dataset, batch_size: int, device: torch.device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.dataloader = None
    
    def create_optimized_dataloader(self, num_workers: int = 4, pin_memory: bool = True):
        """Create optimized DataLoader"""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        logger.info(f"Optimized DataLoader created with {num_workers} workers")
    
    def prefetch_to_device(self, batch):
        """Prefetch batch to device"""
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
        elif torch.is_tensor(batch):
            return batch.to(self.device)
        return batch

class AdvancedTrainingLoop:
    """Advanced training loop with best practices"""
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 criterion: nn.Module, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def setup_scheduler(self, scheduler_type: str = 'cosine', **kwargs):
        """Setup learning rate scheduler"""
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=kwargs.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=kwargs.get('step_size', 30)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=kwargs.get('patience', 10)
            )
        
        logger.info(f"Scheduler '{scheduler_type}' configured")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Advanced training epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        # Record training history
        self.training_history.append({
            'epoch': self.epoch,
            'loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        self.epoch += 1
        return {'loss': avg_loss, 'lr': self.optimizer.param_groups[0]['lr']}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation with no gradient computation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {path}")

class ModelProfiler:
    """Advanced model profiling and analysis"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.profile_results = {}
    
    def profile_model(self, input_shape: Tuple[int, ...], num_runs: int = 100):
        """Profile model performance"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Profile inference
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()
                
                start_time = time.time()
                _ = self.model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    memory_usage.append(end_memory - start_memory)
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        self.profile_results = {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'throughput': 1.0 / np.mean(times)
        }
        
        logger.info(f"Model profiling completed: {self.profile_results}")
        return self.profile_results
    
    def analyze_model_complexity(self):
        """Analyze model complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate FLOPs (simplified)
        flops = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        
        complexity_report = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'estimated_flops': flops
        }
        
        logger.info(f"Model complexity: {complexity_report}")
        return complexity_report

# Example usage and advanced techniques
if __name__ == "__main__":
    # Initialize advanced ML framework
    ml_framework = AdvancedMLFramework()
    
    # Create a sample model
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = SampleModel()
    
    # Advanced optimization
    optimizer = AdvancedModelOptimizer(model, ml_framework.device)
    optimizer.optimize_for_inference()
    
    # Model profiling
    profiler = ModelProfiler(model, ml_framework.device)
    profile_results = profiler.profile_model((1, 784))
    complexity_report = profiler.analyze_model_complexity()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer_opt = optim.Adam(model.parameters(), lr=0.001)
    
    training_loop = AdvancedTrainingLoop(model, optimizer_opt, criterion, ml_framework.device)
    training_loop.setup_scheduler('cosine', epochs=100)
    
    print("Advanced ML Framework expertise demonstrated!")
    print(f"Profile results: {profile_results}")
    print(f"Complexity report: {complexity_report}")