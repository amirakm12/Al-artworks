"""
Logging utilities for Al-artworks.
Provides centralized logging with cosmic-themed formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class CosmicFormatter(logging.Formatter):
    """Custom formatter with cosmic-themed colors and symbols."""
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[35m',      # Magenta
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    
    SYMBOLS = {
        'DEBUG': '🔍',
        'INFO': '✨',
        'WARNING': '⚡',
        'ERROR': '🔥',
        'CRITICAL': '💥',
    }
    
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True, use_symbols: bool = True):
        """Initialize the cosmic formatter."""
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_symbols = use_symbols
        
        format_str = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with cosmic styling."""
        # Add symbol to level name
        if self.use_symbols:
            symbol = self.SYMBOLS.get(record.levelname, '')
            record.levelname = f"{symbol} {record.levelname}"
            
        # Apply color
        if self.use_colors:
            color = self.COLORS.get(record.levelname.split()[-1], '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            record.name = f"\033[34m{record.name}{self.RESET}"  # Blue for name
            
        return super().format(record)


class LogManager:
    """Centralized log management for the application."""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            self.log_file = self.log_dir / f"al-artworks_{datetime.now():%Y%m%d_%H%M%S}.log"
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            
            # Console handler with cosmic formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(CosmicFormatter(use_colors=True, use_symbols=True))
            logger.addHandler(console_handler)
            
            # File handler with standard formatting
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(CosmicFormatter(use_colors=False, use_symbols=False))
            logger.addHandler(file_handler)
            
            # Prevent propagation to root logger
            logger.propagate = False
            
            self._loggers[name] = logger
            
        return self._loggers[name]
        
    def set_log_level(self, level: str, logger_name: Optional[str] = None):
        """Set log level for a specific logger or all loggers."""
        level_obj = getattr(logging, level.upper(), logging.INFO)
        
        if logger_name:
            if logger_name in self._loggers:
                self._loggers[logger_name].setLevel(level_obj)
        else:
            # Set for all loggers
            for logger in self._loggers.values():
                logger.setLevel(level_obj)
                
    def get_log_file_path(self) -> Path:
        """Get the current log file path."""
        return self.log_file
        
    def rotate_logs(self, max_files: int = 10):
        """Rotate log files, keeping only the most recent ones."""
        log_files = sorted(self.log_dir.glob("al-artworks_*.log"))
        
        if len(log_files) > max_files:
            for old_file in log_files[:-max_files]:
                old_file.unlink()


def setup_logger(name: str = "al-artworks") -> logging.Logger:
    """Setup and return a logger instance."""
    manager = LogManager()
    logger = manager.get_logger(name)
    
    # Log startup message
    logger.info("=" * 60)
    logger.info("Al-artworks - The Birth of Celestial Art")
    logger.info("Logger initialized")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    manager = LogManager()
    return manager.get_logger(name)


# Convenience functions for module-level logging
def log_debug(message: str, logger_name: str = "al-artworks"):
    """Log a debug message."""
    get_logger(logger_name).debug(message)
    
    
def log_info(message: str, logger_name: str = "al-artworks"):
    """Log an info message."""
    get_logger(logger_name).info(message)
    
    
def log_warning(message: str, logger_name: str = "al-artworks"):
    """Log a warning message."""
    get_logger(logger_name).warning(message)
    
    
def log_error(message: str, logger_name: str = "al-artworks", exc_info: bool = False):
    """Log an error message."""
    get_logger(logger_name).error(message, exc_info=exc_info)
    
    
def log_critical(message: str, logger_name: str = "al-artworks", exc_info: bool = True):
    """Log a critical message."""
    get_logger(logger_name).critical(message, exc_info=exc_info)