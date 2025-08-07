"""
User interface components for Athena 3D Avatar
Main window, rendering widget, and performance monitoring
"""

from .main_window import AthenaMainWindow, AthenaRenderingWidget
from .performance_panel import PerformancePanel

__all__ = [
    # Main UI components
    'AthenaMainWindow',
    'AthenaRenderingWidget',
    
    # Performance monitoring
    'PerformancePanel'
]