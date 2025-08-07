"""
Rendering components for Athena 3D Avatar
3D rendering, neural radiance fields, and post-processing effects
"""

from .neural_radiance_agent import NeuralRadianceAgent, NeRFConfig, PositionalEncoding, NeRFNetwork
from .renderer_3d import Renderer3D, RenderQuality, RenderConfig
from .post_processing import PostProcessingSystem, PostProcessType, PostProcessConfig, BasePostProcessModel

__all__ = [
    # Neural radiance fields
    'NeuralRadianceAgent',
    'NeRFConfig',
    'PositionalEncoding',
    'NeRFNetwork',
    
    # 3D renderer
    'Renderer3D',
    'RenderQuality',
    'RenderConfig',
    
    # Post-processing system
    'PostProcessingSystem',
    'PostProcessType',
    'PostProcessConfig',
    'BasePostProcessModel'
]