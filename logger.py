import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime

LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = logging.INFO
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

def setup_logger(name: str = "ChatGPTPlus", level: int = DEFAULT_LOG_LEVEL,
                log_to_file: bool = True, log_to_console: bool = True,
                max_size: int = MAX_LOG_SIZE, backup_count: int = BACKUP_COUNT) -> logging.Logger:
    """
    Setup a comprehensive logger with file and console output
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    """
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    if log_to_file:
        # Main log file
        main_log_file = os.path.join(LOG_DIR, f"{name.lower()}.log")
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (only errors and critical)
        error_log_file = os.path.join(LOG_DIR, f"{name.lower()}_error.log")
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Debug log file (only debug messages)
        debug_log_file = os.path.join(LOG_DIR, f"{name.lower()}_debug.log")
        debug_handler = RotatingFileHandler(
            debug_log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        logger.addHandler(debug_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_specialized_logger(name: str, log_type: str = "general") -> logging.Logger:
    """
    Setup specialized loggers for different components
    
    Args:
        name: Logger name
        log_type: Type of logger (voice, ai, plugin, system, etc.)
    """
    
    # Create specialized log directory
    specialized_log_dir = os.path.join(LOG_DIR, log_type)
    os.makedirs(specialized_log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"{name}.{log_type}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Specialized formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for specialized logs
    log_file = os.path.join(specialized_log_dir, f"{name.lower()}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    return logger

def setup_voice_logger(name: str = "VoiceAgent") -> logging.Logger:
    """Setup logger specifically for voice processing"""
    return setup_specialized_logger(name, "voice")

def setup_ai_logger(name: str = "AIAgent") -> logging.Logger:
    """Setup logger specifically for AI processing"""
    return setup_specialized_logger(name, "ai")

def setup_plugin_logger(name: str = "PluginManager") -> logging.Logger:
    """Setup logger specifically for plugin management"""
    return setup_specialized_logger(name, "plugin")

def setup_system_logger(name: str = "SystemMonitor") -> logging.Logger:
    """Setup logger specifically for system monitoring"""
    return setup_specialized_logger(name, "system")

def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("Performance")
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper

def log_async_performance(func):
    """Decorator to log async function performance"""
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger("Performance")
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Async function {func.__name__} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Async function {func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper

class LogManager:
    """Centralized log manager for the application"""
    
    def __init__(self):
        self.loggers = {}
        self.log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
    
    def get_logger(self, name: str, log_type: str = "general") -> logging.Logger:
        """Get or create a logger"""
        key = f"{name}.{log_type}"
        
        if key not in self.loggers:
            if log_type == "voice":
                self.loggers[key] = setup_voice_logger(name)
            elif log_type == "ai":
                self.loggers[key] = setup_ai_logger(name)
            elif log_type == "plugin":
                self.loggers[key] = setup_plugin_logger(name)
            elif log_type == "system":
                self.loggers[key] = setup_system_logger(name)
            else:
                self.loggers[key] = setup_logger(name)
        
        return self.loggers[key]
    
    def set_log_level(self, logger_name: str, level: str):
        """Set log level for a specific logger"""
        if level.upper() in self.log_levels:
            logger = logging.getLogger(logger_name)
            logger.setLevel(self.log_levels[level.upper()])
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "total_loggers": len(self.loggers),
            "log_files": [],
            "log_directory": LOG_DIR
        }
        
        # Get list of log files
        if os.path.exists(LOG_DIR):
            for root, dirs, files in os.walk(LOG_DIR):
                for file in files:
                    if file.endswith('.log'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        stats["log_files"].append({
                            "name": file,
                            "path": file_path,
                            "size_bytes": file_size,
                            "size_mb": file_size / (1024 * 1024)
                        })
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        cleaned_files = 0
        total_size_freed = 0
        
        if os.path.exists(LOG_DIR):
            for root, dirs, files in os.walk(LOG_DIR):
                for file in files:
                    if file.endswith('.log'):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        
                        if file_time < cutoff_time:
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                cleaned_files += 1
                                total_size_freed += file_size
                                print(f"Cleaned up old log file: {file_path}")
                            except Exception as e:
                                print(f"Failed to clean up {file_path}: {e}")
        
        print(f"Log cleanup completed: {cleaned_files} files removed, "
              f"{total_size_freed / (1024 * 1024):.2f} MB freed")

# Global log manager instance
log_manager = LogManager()

# Example usage
def example_logging():
    """Example of using the logging system"""
    
    # Setup main logger
    main_logger = setup_logger("ChatGPTPlus")
    main_logger.info("Application started")
    
    # Setup specialized loggers
    voice_logger = setup_voice_logger("VoiceAgent")
    voice_logger.info("Voice agent initialized")
    
    ai_logger = setup_ai_logger("AIAgent")
    ai_logger.info("AI agent initialized")
    
    plugin_logger = setup_plugin_logger("PluginManager")
    plugin_logger.info("Plugin manager initialized")
    
    # Test performance logging
    @log_performance
    def example_function():
        import time
        time.sleep(0.1)
        return "Function completed"
    
    example_function()
    
    # Get log stats
    stats = log_manager.get_log_stats()
    print(f"Log stats: {stats}")

if __name__ == "__main__":
    example_logging()