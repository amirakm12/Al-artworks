"""
Model Optimizer for Athena 3D Avatar
Prunes and quantizes PyTorch models for 12GB RAM constraint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic, quantize_fx
from torch.nn.utils import prune
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum

class OptimizationLevel(Enum):
    ULTRA_LIGHT = 1    # <1GB, fastest inference
    LIGHT = 2          # <2GB, fast inference
    MEDIUM = 3         # <4GB, balanced
    HEAVY = 4          # <8GB, high quality

@dataclass
class ModelConfig:
    """Configuration for model optimization"""
    optimization_level: OptimizationLevel
    target_latency_ms: float = 250.0
    max_memory_gb: float = 12.0
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_fusion: bool = True
    enable_compilation: bool = True

class ModelOptimizer:
    """Advanced PyTorch model optimizer for memory and latency constraints"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(OptimizationLevel.MEDIUM)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.optimization_times: List[float] = []
        self.model_sizes: Dict[str, float] = {}
        self.latency_benchmarks: Dict[str, float] = {}
        
        # Optimization techniques
        self.quantization_enabled = self.config.enable_quantization
        self.pruning_enabled = self.config.enable_pruning
        self.fusion_enabled = self.config.enable_fusion
        self.compilation_enabled = self.config.enable_compilation
        
    def optimize_for_memory(self):
        """Apply memory optimization techniques"""
        try:
            self.logger.info("Starting memory optimization")
            
            # Set PyTorch memory efficient settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable memory efficient attention
            if hasattr(torch.backends, 'flash_attention'):
                torch.backends.flash_attention.enabled = True
            
            # Set memory fraction for GPU
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
                torch.cuda.empty_cache()
            
            self.logger.info("Memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    def optimize_model(self, model: nn.Module, model_name: str) -> nn.Module:
        """Optimize a PyTorch model for memory and latency constraints"""
        try:
            start_time = time.time()
            self.logger.info(f"Optimizing model: {model_name}")
            
            # Store original model size
            original_size = self.get_model_size_mb(model)
            self.model_sizes[f"{model_name}_original"] = original_size
            
            # Apply optimizations based on level
            if self.optimization_level == OptimizationLevel.ULTRA_LIGHT:
                model = self._apply_ultra_light_optimization(model)
            elif self.optimization_level == OptimizationLevel.LIGHT:
                model = self._apply_light_optimization(model)
            elif self.optimization_level == OptimizationLevel.MEDIUM:
                model = self._apply_medium_optimization(model)
            else:  # HEAVY
                model = self._apply_heavy_optimization(model)
            
            # Apply quantization if enabled
            if self.quantization_enabled:
                model = self._apply_quantization(model)
            
            # Apply pruning if enabled
            if self.pruning_enabled:
                model = self._apply_pruning(model)
            
            # Apply fusion if enabled
            if self.fusion_enabled:
                model = self._apply_fusion(model)
            
            # Apply compilation if enabled
            if self.compilation_enabled:
                model = self._apply_compilation(model)
            
            # Measure optimized model size
            optimized_size = self.get_model_size_mb(model)
            self.model_sizes[f"{model_name}_optimized"] = optimized_size
            
            # Benchmark latency
            latency = self._benchmark_latency(model)
            self.latency_benchmarks[model_name] = latency
            
            # Record optimization time
            optimization_time = time.time() - start_time
            self.optimization_times.append(optimization_time)
            
            self.logger.info(f"Model optimization completed: {original_size:.1f}MB -> {optimized_size:.1f}MB, "
                           f"Latency: {latency:.1f}ms, Time: {optimization_time:.1f}s")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model
    
    def _apply_ultra_light_optimization(self, model: nn.Module) -> nn.Module:
        """Apply ultra-light optimization for <1GB models"""
        try:
            # Convert to float16 for memory reduction
            model = model.half()
            
            # Apply aggressive pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.7)
                elif isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.6)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Ultra-light optimization failed: {e}")
            return model
    
    def _apply_light_optimization(self, model: nn.Module) -> nn.Module:
        """Apply light optimization for <2GB models"""
        try:
            # Convert to float16
            model = model.half()
            
            # Apply moderate pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.5)
                elif isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.4)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Light optimization failed: {e}")
            return model
    
    def _apply_medium_optimization(self, model: nn.Module) -> nn.Module:
        """Apply medium optimization for <4GB models"""
        try:
            # Apply light pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.3)
                elif isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Medium optimization failed: {e}")
            return model
    
    def _apply_heavy_optimization(self, model: nn.Module) -> nn.Module:
        """Apply heavy optimization for <8GB models"""
        try:
            # Apply minimal pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
                elif isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Heavy optimization failed: {e}")
            return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to reduce model size"""
        try:
            # Apply dynamic quantization
            quantized_model = quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d, nn.Conv3d}, 
                dtype=torch.qint8
            )
            
            self.logger.info("Quantization applied successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to reduce model complexity"""
        try:
            # Apply structured pruning to attention layers
            for name, module in model.named_modules():
                if 'attention' in name.lower() and isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                elif 'mlp' in name.lower() and isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
            
            self.logger.info("Pruning applied successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model
    
    def _apply_fusion(self, model: nn.Module) -> nn.Module:
        """Apply operator fusion for better performance"""
        try:
            # Fuse batch norm with conv layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Try to fuse with following batch norm
                    for child_name, child_module in module.named_children():
                        if isinstance(child_module, nn.BatchNorm2d):
                            # Fuse conv + bn
                            fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(
                                module, child_module
                            )
                            setattr(module, child_name, fused_conv)
            
            self.logger.info("Fusion applied successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            return model
    
    def _apply_compilation(self, model: nn.Module) -> nn.Module:
        """Apply TorchScript compilation for better performance"""
        try:
            # Convert to TorchScript
            if hasattr(torch.jit, 'script'):
                scripted_model = torch.jit.script(model)
                self.logger.info("TorchScript compilation applied successfully")
                return scripted_model
            else:
                return model
                
        except Exception as e:
            self.logger.error(f"Compilation failed: {e}")
            return model
    
    def _benchmark_latency(self, model: nn.Module) -> float:
        """Benchmark model inference latency"""
        try:
            model.eval()
            
            # Create dummy input
            if hasattr(model, 'input_shape'):
                dummy_input = torch.randn(1, *model.input_shape)
            else:
                # Default input shape
                dummy_input = torch.randn(1, 3, 224, 224)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    _ = model(dummy_input)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate average latency
            avg_latency = np.mean(times)
            
            return avg_latency
            
        except Exception as e:
            self.logger.error(f"Latency benchmarking failed: {e}")
            return 0.0
    
    def get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB"""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
            
        except Exception as e:
            self.logger.error(f"Failed to calculate model size: {e}")
            return 0.0
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        try:
            stats = {
                'optimization_level': self.config.optimization_level.name,
                'target_latency_ms': self.config.target_latency_ms,
                'max_memory_gb': self.config.max_memory_gb,
                'model_sizes': self.model_sizes,
                'latency_benchmarks': self.latency_benchmarks,
                'avg_optimization_time': np.mean(self.optimization_times) if self.optimization_times else 0.0,
                'total_optimizations': len(self.optimization_times)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        try:
            # Clear optimization data
            self.optimization_times.clear()
            self.model_sizes.clear()
            self.latency_benchmarks.clear()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Model optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Model optimizer cleanup failed: {e}")
    
    @property
    def optimization_level(self) -> OptimizationLevel:
        return self.config.optimization_level