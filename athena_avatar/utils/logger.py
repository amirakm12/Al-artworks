"""
Logging utility for Athena 3D Avatar
Cosmic-themed logging with performance tracking
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
import threading

class CosmicFormatter(logging.Formatter):
    """Cosmic-themed log formatter"""
    
    # Cosmic color codes
    COSMIC_COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
        
    def format(self, record):
        # Add cosmic prefix
        cosmic_prefix = "🌟" if record.levelno >= logging.ERROR else "✨"
        
        # Format the message
        formatted = super().format(record)
        
        if self.use_colors and record.levelname in self.COSMIC_COLORS:
            color = self.COSMIC_COLORS[record.levelname]
            reset = self.COSMIC_COLORS['RESET']
            formatted = f"{cosmic_prefix} {color}{formatted}{reset}"
        else:
            formatted = f"{cosmic_prefix} {formatted}"
        
        return formatted

def setup_logger(name: str = "athena_avatar", 
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                use_colors: bool = True) -> logging.Logger:
    """Setup cosmic-themed logger for Athena"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create cosmic formatter
    cosmic_formatter = CosmicFormatter(use_colors=use_colors)
    console_handler.setFormatter(cosmic_formatter)
    
    # Add console handler
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Use standard formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.performance_data = {}
        
    def log_performance(self, component: str, metric: str, value: float, unit: str = ""):
        """Log performance metric"""
        key = f"{component}_{metric}"
        self.performance_data[key] = {
            'value': value,
            'unit': unit,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"Performance - {component}: {metric} = {value:.3f} {unit}")
    
    def log_memory_usage(self, usage_gb: float):
        """Log memory usage"""
        self.log_performance("Memory", "Usage", usage_gb, "GB")
    
    def log_inference_time(self, time_ms: float):
        """Log inference time"""
        self.log_performance("Model", "Inference", time_ms, "ms")
    
    def log_fps(self, fps: float):
        """Log FPS"""
        self.log_performance("Rendering", "FPS", fps, "fps")
    
    def log_latency(self, latency_ms: float):
        """Log latency"""
        self.log_performance("System", "Latency", latency_ms, "ms")
    
    def get_performance_summary(self) -> dict:
        """Get performance summary"""
        return self.performance_data.copy()

class CosmicLogger:
    """Enhanced cosmic logger with performance tracking"""
    
    def __init__(self, name: str = "athena_cosmic"):
        self.logger = setup_logger(name)
        self.performance_logger = PerformanceLogger(self.logger)
        
    def log_cosmic_event(self, event: str, details: str = ""):
        """Log cosmic event"""
        self.logger.info(f"Cosmic Event: {event} - {details}")
    
    def log_divine_interaction(self, interaction: str):
        """Log divine interaction"""
        self.logger.info(f"Divine Interaction: {interaction}")
    
    def log_mystical_occurrence(self, occurrence: str):
        """Log mystical occurrence"""
        self.logger.info(f"Mystical Occurrence: {occurrence}")
    
    def log_celestial_phenomenon(self, phenomenon: str):
        """Log celestial phenomenon"""
        self.logger.info(f"Celestial Phenomenon: {phenomenon}")
    
    def log_transcendence(self, level: str):
        """Log transcendence event"""
        self.logger.info(f"Transcendence Level: {level}")
    
    def log_performance(self, component: str, metric: str, value: float, unit: str = ""):
        """Log performance metric"""
        self.performance_logger.log_performance(component, metric, value, unit)
    
    def get_performance_summary(self) -> dict:
        """Get performance summary"""
        return self.performance_logger.get_performance_summary()
    
    def __getattr__(self, name):
        """Delegate to underlying logger"""
        return getattr(self.logger, name)