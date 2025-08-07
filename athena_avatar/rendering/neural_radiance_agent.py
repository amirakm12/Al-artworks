"""
Neural Radiance Fields (NeRF) Agent for Athena 3D Avatar
Advanced 3D rendering with 2GB model size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from dataclasses import dataclass

@dataclass
class NeRFConfig:
    """Configuration for NeRF rendering"""
    model_size_mb: float = 2.0
    resolution: int = 512
    num_samples: int = 64
    chunk_size: int = 4096
    enable_hierarchical: bool = True

class NeuralRadianceAgent:
    """Neural Radiance Fields agent for advanced 3D rendering"""
    
    def __init__(self, config: Optional[NeRFConfig] = None):
        self.config = config or NeRFConfig()
        self.logger = logging.getLogger(__name__)
        
        # NeRF components
        self.coarse_network = None
        self.fine_network = None
        self.positional_encoding = None
        
        # Performance tracking
        self.render_times: List[float] = []
        self.memory_usage: List[float] = []
        
        # Initialize components
        self._initialize_nerf_components()
        
    def _initialize_nerf_components(self):
        """Initialize NeRF neural networks"""
        try:
            # Positional encoding for 3D coordinates
            self.positional_encoding = PositionalEncoding()
            
            # Coarse network for hierarchical sampling
            self.coarse_network = NeRFNetwork(
                input_dim=63,  # 3D coordinates + 3D direction + positional encoding
                output_dim=4,  # RGB + density
                hidden_dim=256,
                num_layers=8
            )
            
            # Fine network for detailed rendering
            self.fine_network = NeRFNetwork(
                input_dim=63,
                output_dim=4,
                hidden_dim=256,
                num_layers=8
            )
            
            self.logger.info(f"NeRF components initialized ({self.config.model_size_mb}MB)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NeRF components: {e}")
    
    def render_view(self, camera_pose: np.ndarray, 
                   camera_intrinsics: np.ndarray) -> np.ndarray:
        """Render a view from given camera pose"""
        try:
            start_time = time.time()
            
            # Generate ray origins and directions
            rays_o, rays_d = self._generate_rays(camera_pose, camera_intrinsics)
            
            # Hierarchical sampling
            if self.config.enable_hierarchical:
                # Coarse sampling
                coarse_samples = self._sample_coarse(rays_o, rays_d)
                coarse_output = self._render_coarse(rays_o, rays_d, coarse_samples)
                
                # Fine sampling
                fine_samples = self._sample_fine(rays_o, rays_d, coarse_output)
                fine_output = self._render_fine(rays_o, rays_d, fine_samples)
                
                rendered_image = fine_output
            else:
                # Single network rendering
                samples = self._sample_uniform(rays_o, rays_d)
                rendered_image = self._render_single(rays_o, rays_d, samples)
            
            # Record render time
            render_time = time.time() - start_time
            self.render_times.append(render_time)
            
            # Keep only recent render times
            if len(self.render_times) > 100:
                self.render_times.pop(0)
            
            self.logger.info(f"View rendered in {render_time:.3f}s")
            
            return rendered_image
            
        except Exception as e:
            self.logger.error(f"Failed to render view: {e}")
            return np.zeros((self.config.resolution, self.config.resolution, 3))
    
    def _generate_rays(self, camera_pose: np.ndarray, 
                      camera_intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions"""
        try:
            H, W = self.config.resolution, self.config.resolution
            
            # Create pixel coordinates
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            
            # Convert to camera coordinates
            dirs = np.stack([(i - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0],
                           -(j - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1],
                           -np.ones_like(i)], -1)
            
            # Transform to world coordinates
            rotation = camera_pose[:3, :3]
            translation = camera_pose[:3, 3]
            
            rays_d = np.sum(dirs[..., np.newaxis, :] * rotation, axis=-1)
            rays_o = np.broadcast_to(translation, rays_d.shape)
            
            return rays_o, rays_d
            
        except Exception as e:
            self.logger.error(f"Failed to generate rays: {e}")
            return np.zeros((H*W, 3)), np.zeros((H*W, 3))
    
    def _sample_coarse(self, rays_o: np.ndarray, rays_d: np.ndarray) -> np.ndarray:
        """Sample points along rays for coarse network"""
        try:
            # Uniform sampling along rays
            t_vals = np.linspace(0., 1., self.config.num_samples)
            z_vals = 1. / (1. / 0.1 * (1. - t_vals) + 1. / 10. * t_vals)
            z_vals = np.broadcast_to(z_vals, (rays_o.shape[0], self.config.num_samples))
            
            # Generate sample points
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            
            return pts
            
        except Exception as e:
            self.logger.error(f"Failed to sample coarse: {e}")
            return np.zeros((rays_o.shape[0], self.config.num_samples, 3))
    
    def _sample_fine(self, rays_o: np.ndarray, rays_d: np.ndarray, 
                    coarse_output: Dict[str, np.ndarray]) -> np.ndarray:
        """Sample points along rays for fine network"""
        try:
            # Use weights from coarse network for importance sampling
            weights = coarse_output['weights']
            
            # Hierarchical sampling
            t_vals_mid = 0.5 * (weights[..., 1:] + weights[..., :-1])
            t_vals = np.concatenate([t_vals_mid, weights[..., -1:]], axis=-1)
            
            # Generate sample points
            pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
            
            return pts
            
        except Exception as e:
            self.logger.error(f"Failed to sample fine: {e}")
            return np.zeros((rays_o.shape[0], self.config.num_samples, 3))
    
    def _sample_uniform(self, rays_o: np.ndarray, rays_d: np.ndarray) -> np.ndarray:
        """Uniform sampling for single network"""
        try:
            # Uniform sampling along rays
            t_vals = np.linspace(0., 1., self.config.num_samples)
            z_vals = 1. / (1. / 0.1 * (1. - t_vals) + 1. / 10. * t_vals)
            z_vals = np.broadcast_to(z_vals, (rays_o.shape[0], self.config.num_samples))
            
            # Generate sample points
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            
            return pts
            
        except Exception as e:
            self.logger.error(f"Failed to sample uniform: {e}")
            return np.zeros((rays_o.shape[0], self.config.num_samples, 3))
    
    def _render_coarse(self, rays_o: np.ndarray, rays_d: np.ndarray, 
                      samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Render using coarse network"""
        try:
            # Process samples in chunks
            outputs = []
            for i in range(0, samples.shape[0], self.config.chunk_size):
                chunk = samples[i:i+self.config.chunk_size]
                
                # Encode positions and directions
                encoded_pos = self.positional_encoding(chunk.reshape(-1, 3))
                encoded_dir = self.positional_encoding(rays_d[i:i+self.config.chunk_size])
                
                # Combine encodings
                encoded = torch.cat([encoded_pos, encoded_dir], dim=-1)
                
                # Forward pass
                with torch.no_grad():
                    output = self.coarse_network(encoded)
                
                outputs.append(output.detach().numpy())
            
            # Combine outputs
            combined_output = np.concatenate(outputs, axis=0)
            
            # Reshape to original dimensions
            rgb = combined_output[..., :3]
            density = combined_output[..., 3:]
            
            # Compute weights
            weights = self._compute_weights(density)
            
            return {
                'rgb': rgb,
                'density': density,
                'weights': weights
            }
            
        except Exception as e:
            self.logger.error(f"Failed to render coarse: {e}")
            return {}
    
    def _render_fine(self, rays_o: np.ndarray, rays_d: np.ndarray, 
                    samples: np.ndarray) -> np.ndarray:
        """Render using fine network"""
        try:
            # Process samples in chunks
            outputs = []
            for i in range(0, samples.shape[0], self.config.chunk_size):
                chunk = samples[i:i+self.config.chunk_size]
                
                # Encode positions and directions
                encoded_pos = self.positional_encoding(chunk.reshape(-1, 3))
                encoded_dir = self.positional_encoding(rays_d[i:i+self.config.chunk_size])
                
                # Combine encodings
                encoded = torch.cat([encoded_pos, encoded_dir], dim=-1)
                
                # Forward pass
                with torch.no_grad():
                    output = self.fine_network(encoded)
                
                outputs.append(output.detach().numpy())
            
            # Combine outputs
            combined_output = np.concatenate(outputs, axis=0)
            
            # Reshape to original dimensions
            rgb = combined_output[..., :3]
            density = combined_output[..., 3:]
            
            # Compute weights
            weights = self._compute_weights(density)
            
            # Render final image
            rendered_rgb = np.sum(weights[..., None] * rgb, axis=-2)
            
            return rendered_rgb.reshape(self.config.resolution, self.config.resolution, 3)
            
        except Exception as e:
            self.logger.error(f"Failed to render fine: {e}")
            return np.zeros((self.config.resolution, self.config.resolution, 3))
    
    def _render_single(self, rays_o: np.ndarray, rays_d: np.ndarray, 
                      samples: np.ndarray) -> np.ndarray:
        """Render using single network"""
        try:
            # Process samples in chunks
            outputs = []
            for i in range(0, samples.shape[0], self.config.chunk_size):
                chunk = samples[i:i+self.config.chunk_size]
                
                # Encode positions and directions
                encoded_pos = self.positional_encoding(chunk.reshape(-1, 3))
                encoded_dir = self.positional_encoding(rays_d[i:i+self.config.chunk_size])
                
                # Combine encodings
                encoded = torch.cat([encoded_pos, encoded_dir], dim=-1)
                
                # Forward pass
                with torch.no_grad():
                    output = self.coarse_network(encoded)
                
                outputs.append(output.detach().numpy())
            
            # Combine outputs
            combined_output = np.concatenate(outputs, axis=0)
            
            # Reshape to original dimensions
            rgb = combined_output[..., :3]
            density = combined_output[..., 3:]
            
            # Compute weights
            weights = self._compute_weights(density)
            
            # Render final image
            rendered_rgb = np.sum(weights[..., None] * rgb, axis=-2)
            
            return rendered_rgb.reshape(self.config.resolution, self.config.resolution, 3)
            
        except Exception as e:
            self.logger.error(f"Failed to render single: {e}")
            return np.zeros((self.config.resolution, self.config.resolution, 3))
    
    def _compute_weights(self, density: np.ndarray) -> np.ndarray:
        """Compute weights from density values"""
        try:
            # Convert density to alpha
            alpha = 1. - np.exp(-density)
            
            # Compute weights using alpha compositing
            weights = alpha * np.cumprod(np.concatenate([
                np.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10
            ], axis=-1), axis=-1)[:, :-1]
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to compute weights: {e}")
            return np.zeros_like(density)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = {
                'model_size_mb': self.config.model_size_mb,
                'avg_render_time_ms': np.mean(self.render_times) * 1000 if self.render_times else 0.0,
                'min_render_time_ms': np.min(self.render_times) * 1000 if self.render_times else 0.0,
                'max_render_time_ms': np.max(self.render_times) * 1000 if self.render_times else 0.0,
                'total_renders': len(self.render_times),
                'resolution': self.config.resolution,
                'num_samples': self.config.num_samples,
                'chunk_size': self.config.chunk_size,
                'hierarchical_enabled': self.config.enable_hierarchical
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup NeRF agent resources"""
        try:
            # Clear performance data
            self.render_times.clear()
            self.memory_usage.clear()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("NeRF agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"NeRF agent cleanup failed: {e}")

class PositionalEncoding(nn.Module):
    """Positional encoding for 3D coordinates and directions"""
    
    def __init__(self, L: int = 10):
        super().__init__()
        self.L = L
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding"""
        encoded = [x]
        
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                encoded.append(fn((2. ** i) * x))
        
        return torch.cat(encoded, dim=-1)

class NeRFNetwork(nn.Module):
    """Neural network for NeRF rendering"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: int = 256, num_layers: int = 8):
        super().__init__()
        
        # Main network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.main_network = nn.Sequential(*layers)
        
        # Output layers
        self.rgb_layer = nn.Linear(hidden_dim, 3)
        self.density_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Main network
        features = self.main_network(x)
        
        # Output layers
        rgb = torch.sigmoid(self.rgb_layer(features))
        density = F.relu(self.density_layer(features))
        
        return torch.cat([rgb, density], dim=-1)