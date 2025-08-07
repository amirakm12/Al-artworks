"""
Centralized Error Handling & Logging System
Provides comprehensive error handling, logging, and crash reporting
"""

import sys
import os
import traceback
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json

class ErrorHandler:
    """Centralized error handling and logging system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Error tracking
        self.error_count = 0
        self.critical_errors = []
        self.error_callbacks = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Install global exception handlers
        self.install_exception_handlers()
        
        self.logger.info("Error handler initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Main application log
        main_handler = logging.FileHandler(self.log_dir / "app.log")
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.INFO)
        
        # Error log
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Debug log
        debug_handler = logging.FileHandler(self.log_dir / "debug.log")
        debug_handler.setFormatter(detailed_formatter)
        debug_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(main_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(debug_handler)
        root_logger.addHandler(console_handler)
        
        # Create component loggers
        self.logger = logging.getLogger("ErrorHandler")
        self.plugin_logger = logging.getLogger("PluginSystem")
        self.voice_logger = logging.getLogger("VoiceSystem")
        self.ui_logger = logging.getLogger("UISystem")
        self.llm_logger = logging.getLogger("LLMSystem")
    
    def install_exception_handlers(self):
        """Install global exception handlers"""
        # Set up sys.excepthook for unhandled exceptions
        sys.excepthook = self.handle_uncaught_exception
        
        # Set up threading exception handler
        threading.excepthook = self.handle_thread_exception
    
    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        with self.lock:
            self.error_count += 1
            
            # Log the exception
            self.logger.error(
                f"Uncaught exception: {exc_type.__name__}: {exc_value}",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
            # Store critical error
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback),
                "thread": threading.current_thread().name
            }
            
            self.critical_errors.append(error_info)
            
            # Call error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(error_info)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")
            
            # Save error report
            self.save_error_report(error_info)
    
    def handle_thread_exception(self, args):
        """Handle thread exceptions"""
        with self.lock:
            self.error_count += 1
            
            self.logger.error(
                f"Thread exception in {args.thread}: {args.exc_type.__name__}: {args.exc_value}",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
            )
            
            # Store thread error
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "ThreadException",
                "thread": args.thread.name,
                "message": f"{args.exc_type.__name__}: {args.exc_value}",
                "traceback": traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            }
            
            self.critical_errors.append(error_info)
    
    def handle_plugin_error(self, plugin_name: str, error: Exception, context: str = ""):
        """Handle plugin-specific errors"""
        with self.lock:
            self.error_count += 1
            
            self.plugin_logger.error(
                f"Plugin error in {plugin_name}: {error}",
                exc_info=True,
                extra={"plugin": plugin_name, "context": context}
            )
            
            # Store plugin error
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "PluginError",
                "plugin": plugin_name,
                "message": str(error),
                "context": context,
                "traceback": traceback.format_exception(type(error), error, error.__traceback__)
            }
            
            self.critical_errors.append(error_info)
    
    def handle_voice_error(self, error: Exception, context: str = ""):
        """Handle voice system errors"""
        with self.lock:
            self.error_count += 1
            
            self.voice_logger.error(
                f"Voice system error: {error}",
                exc_info=True,
                extra={"context": context}
            )
    
    def handle_ui_error(self, error: Exception, context: str = ""):
        """Handle UI system errors"""
        with self.lock:
            self.error_count += 1
            
            self.ui_logger.error(
                f"UI error: {error}",
                exc_info=True,
                extra={"context": context}
            )
    
    def handle_llm_error(self, error: Exception, context: str = ""):
        """Handle LLM system errors"""
        with self.lock:
            self.error_count += 1
            
            self.llm_logger.error(
                f"LLM error: {error}",
                exc_info=True,
                extra={"context": context}
            )
    
    def add_error_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback to be called when errors occur"""
        self.error_callbacks.append(callback)
    
    def save_error_report(self, error_info: Dict[str, Any]):
        """Save detailed error report"""
        try:
            report_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Add system information
            error_info["system_info"] = {
                "platform": sys.platform,
                "python_version": sys.version,
                "memory_usage": self.get_memory_usage(),
                "thread_count": threading.active_count()
            }
            
            with open(report_file, 'w') as f:
                json.dump(error_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors"""
        with self.lock:
            return {
                "total_errors": self.error_count,
                "critical_errors": len(self.critical_errors),
                "recent_errors": self.critical_errors[-10:] if self.critical_errors else [],
                "log_files": [
                    str(f) for f in self.log_dir.glob("*.log")
                ]
            }
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up old log files"""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.logger.info(f"Cleaned up old log file: {log_file}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up logs: {e}")
    
    def log_system_info(self):
        """Log system information for debugging"""
        try:
            import psutil
            
            self.logger.info("=== System Information ===")
            self.logger.info(f"Platform: {sys.platform}")
            self.logger.info(f"Python Version: {sys.version}")
            self.logger.info(f"CPU Count: {psutil.cpu_count()}")
            self.logger.info(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
            self.logger.info(f"Disk: {psutil.disk_usage('/').free / 1024 / 1024 / 1024:.1f} GB free")
            
        except ImportError:
            self.logger.warning("psutil not available - limited system info")
    
    def create_crash_report(self) -> str:
        """Create a comprehensive crash report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "error_summary": self.get_error_summary(),
                "system_info": {
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "memory_usage": self.get_memory_usage()
                },
                "recent_errors": self.critical_errors[-5:] if self.critical_errors else []
            }
            
            report_file = self.log_dir / f"crash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create crash report: {e}")
            return ""

# Global error handler instance
error_handler = ErrorHandler()

def handle_error(error: Exception, context: str = "", error_type: str = "General"):
    """Global error handling function"""
    if error_type == "Plugin":
        error_handler.handle_plugin_error(context, error)
    elif error_type == "Voice":
        error_handler.handle_voice_error(error, context)
    elif error_type == "UI":
        error_handler.handle_ui_error(error, context)
    elif error_type == "LLM":
        error_handler.handle_llm_error(error, context)
    else:
        error_handler.logger.error(f"General error: {error}", exc_info=True)

if __name__ == "__main__":
    # Test the error handler
    print("🧪 Testing Error Handler...")
    
    # Test basic logging
    error_handler.logger.info("Test info message")
    error_handler.logger.warning("Test warning message")
    error_handler.logger.error("Test error message")
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        handle_error(e, "Test context", "General")
    
    # Test system info
    error_handler.log_system_info()
    
    # Test error summary
    summary = error_handler.get_error_summary()
    print(f"Error summary: {summary}")
    
    print("✅ Error handler test completed")