"""
Global Error Handler - Comprehensive Error Handling and UI Feedback
Provides centralized error handling, logging, and user-friendly error dialogs
"""

import sys
import os
import logging
import traceback
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path

try:
    from PyQt6.QtWidgets import (QMessageBox, QApplication, QDialog, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QTextEdit, QPushButton, 
                                 QProgressBar, QCheckBox, QGroupBox)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt6.QtGui import QFont, QIcon, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling and reporting system"""
    
    def __init__(self):
        self.error_history = []
        self.max_history = 100
        self.error_callbacks = {}
        self.recovery_actions = {}
        self.critical_errors = []
        
        # Error categories
        self.error_categories = {
            'ui': 'User Interface Errors',
            'llm': 'Language Model Errors', 
            'voice': 'Voice Recognition Errors',
            'plugin': 'Plugin System Errors',
            'network': 'Network/Connection Errors',
            'file': 'File System Errors',
            'security': 'Security Violations',
            'performance': 'Performance Issues',
            'unknown': 'Unknown Errors'
        }
        
        # Initialize global exception handlers
        self._setup_global_handlers()
        
        logger.info("Error handler initialized")
    
    def _setup_global_handlers(self):
        """Setup global exception handlers"""
        # Store original handlers
        self.original_excepthook = sys.excepthook
        self.original_thread_excepthook = threading.excepthook
        
        # Set custom handlers
        sys.excepthook = self._handle_uncaught_exception
        threading.excepthook = self._handle_thread_exception
    
    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions in main thread"""
        error_info = {
            'timestamp': datetime.now(),
            'type': exc_type.__name__,
            'message': str(exc_value),
            'traceback': ''.join(traceback.format_tb(exc_traceback)),
            'thread': 'main',
            'category': self._categorize_error(exc_type, exc_value)
        }
        
        self._log_error(error_info)
        self._show_error_dialog(error_info)
        
        # Call original handler
        self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _handle_thread_exception(self, args):
        """Handle exceptions in background threads"""
        error_info = {
            'timestamp': datetime.now(),
            'type': type(args.exc_value).__name__,
            'message': str(args.exc_value),
            'traceback': ''.join(traceback.format_tb(args.exc_traceback)),
            'thread': args.thread.name if hasattr(args.thread, 'name') else 'unknown',
            'category': self._categorize_error(type(args.exc_value), args.exc_value)
        }
        
        self._log_error(error_info)
        
        # Show error dialog in main thread if PyQt is available
        if PYQT_AVAILABLE and QApplication.instance():
            QApplication.instance().postEvent(
                QApplication.instance().activeWindow(),
                self._create_error_event(error_info)
            )
    
    def _categorize_error(self, exc_type, exc_value) -> str:
        """Categorize error based on type and message"""
        error_str = str(exc_value).lower()
        
        if any(keyword in error_str for keyword in ['ui', 'qt', 'widget', 'dialog']):
            return 'ui'
        elif any(keyword in error_str for keyword in ['llm', 'model', 'generation', 'inference']):
            return 'llm'
        elif any(keyword in error_str for keyword in ['voice', 'audio', 'speech', 'whisper']):
            return 'voice'
        elif any(keyword in error_str for keyword in ['plugin', 'module', 'import']):
            return 'plugin'
        elif any(keyword in error_str for keyword in ['network', 'connection', 'http', 'url']):
            return 'network'
        elif any(keyword in error_str for keyword in ['file', 'path', 'directory', 'permission']):
            return 'file'
        elif any(keyword in error_str for keyword in ['security', 'violation', 'sandbox']):
            return 'security'
        elif any(keyword in error_str for keyword in ['performance', 'timeout', 'memory']):
            return 'performance'
        else:
            return 'unknown'
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error to file and console"""
        self.error_history.append(error_info)
        
        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log to file
        logger.error(f"Error [{error_info['category']}]: {error_info['type']}: {error_info['message']}")
        logger.error(f"Thread: {error_info['thread']}")
        logger.error(f"Traceback: {error_info['traceback']}")
        
        # Check if this is a critical error
        if self._is_critical_error(error_info):
            self.critical_errors.append(error_info)
            logger.critical(f"CRITICAL ERROR DETECTED: {error_info['message']}")
    
    def _is_critical_error(self, error_info: Dict[str, Any]) -> bool:
        """Determine if error is critical"""
        critical_keywords = [
            'out of memory', 'segmentation fault', 'access violation',
            'corruption', 'fatal', 'critical', 'system crash'
        ]
        
        error_str = error_info['message'].lower()
        return any(keyword in error_str for keyword in critical_keywords)
    
    def _show_error_dialog(self, error_info: Dict[str, Any]):
        """Show user-friendly error dialog"""
        if not PYQT_AVAILABLE:
            print(f"Error: {error_info['message']}")
            return
        
        try:
            dialog = ErrorDialog(error_info)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to show error dialog: {e}")
            print(f"Error: {error_info['message']}")
    
    def register_error_callback(self, category: str, callback: Callable):
        """Register callback for specific error category"""
        self.error_callbacks[category] = callback
        logger.info(f"Registered error callback for category: {category}")
    
    def register_recovery_action(self, category: str, action: Callable):
        """Register recovery action for error category"""
        self.recovery_actions[category] = action
        logger.info(f"Registered recovery action for category: {category}")
    
    def handle_error(self, error: Exception, category: str = 'unknown', 
                    show_dialog: bool = True, attempt_recovery: bool = True):
        """Handle error with optional recovery"""
        error_info = {
            'timestamp': datetime.now(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'thread': threading.current_thread().name,
            'category': category
        }
        
        # Log error
        self._log_error(error_info)
        
        # Call category-specific callback
        if category in self.error_callbacks:
            try:
                self.error_callbacks[category](error_info)
            except Exception as e:
                logger.error(f"Error in callback for {category}: {e}")
        
        # Attempt recovery
        if attempt_recovery and category in self.recovery_actions:
            try:
                self.recovery_actions[category](error_info)
                logger.info(f"Recovery action executed for {category}")
            except Exception as e:
                logger.error(f"Recovery action failed for {category}: {e}")
        
        # Show dialog
        if show_dialog:
            self._show_error_dialog(error_info)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and statistics"""
        if not self.error_history:
            return {'total_errors': 0, 'categories': {}}
        
        category_counts = {}
        for error in self.error_history:
            category = error['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'critical_errors': len(self.critical_errors),
            'categories': category_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'error_rate': len(self.error_history) / max(1, (datetime.now() - self.error_history[0]['timestamp']).total_seconds() / 3600)
        }
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.critical_errors.clear()
        logger.info("Error history cleared")
    
    def save_error_report(self, filename: str = None):
        """Save error report to file"""
        if not filename:
            filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_error_summary(),
            'error_history': self.error_history,
            'critical_errors': self.critical_errors
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Error report saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")

class ErrorDialog(QDialog):
    """User-friendly error dialog"""
    
    def __init__(self, error_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.error_info = error_info
        self.setup_ui()
    
    def setup_ui(self):
        """Setup error dialog UI"""
        self.setWindowTitle("Error - ChatGPT+ Clone")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # Error icon and title
        title_layout = QHBoxLayout()
        title_label = QLabel(f"⚠️  {self.error_categories.get(self.error_info['category'], 'Unknown Error')}")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        layout.addLayout(title_layout)
        
        # Error message
        message_label = QLabel(self.error_info['message'])
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #d32f2f; padding: 10px;")
        layout.addWidget(message_label)
        
        # Error details (collapsible)
        details_group = QGroupBox("Error Details")
        details_layout = QVBoxLayout()
        
        # Thread info
        thread_label = QLabel(f"Thread: {self.error_info['thread']}")
        details_layout.addWidget(thread_label)
        
        # Timestamp
        time_label = QLabel(f"Time: {self.error_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        details_layout.addWidget(time_label)
        
        # Traceback (scrollable)
        traceback_text = QTextEdit()
        traceback_text.setPlainText(self.error_info['traceback'])
        traceback_text.setMaximumHeight(150)
        traceback_text.setReadOnly(True)
        details_layout.addWidget(traceback_text)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Copy button
        copy_btn = QPushButton("Copy Error")
        copy_btn.clicked.connect(self.copy_error)
        button_layout.addWidget(copy_btn)
        
        # Report button
        report_btn = QPushButton("Report Issue")
        report_btn.clicked.connect(self.report_issue)
        button_layout.addWidget(report_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def copy_error(self):
        """Copy error details to clipboard"""
        try:
            clipboard = QApplication.clipboard()
            error_text = f"""
Error: {self.error_info['message']}
Type: {self.error_info['type']}
Thread: {self.error_info['thread']}
Time: {self.error_info['timestamp']}
Category: {self.error_info['category']}

Traceback:
{self.error_info['traceback']}
"""
            clipboard.setText(error_text)
        except Exception as e:
            logger.error(f"Failed to copy error: {e}")
    
    def report_issue(self):
        """Open issue reporting dialog"""
        try:
            # This could open a web browser or email client
            # For now, just show a message
            QMessageBox.information(self, "Report Issue", 
                                  "Please copy the error details and report the issue to the development team.")
        except Exception as e:
            logger.error(f"Failed to report issue: {e}")

class ErrorMonitor(QThread):
    """Background thread for monitoring errors and system health"""
    
    error_detected = pyqtSignal(dict)
    system_health_update = pyqtSignal(dict)
    
    def __init__(self, error_handler: ErrorHandler):
        super().__init__()
        self.error_handler = error_handler
        self.monitoring = True
    
    def run(self):
        """Monitor system health and errors"""
        while self.monitoring:
            try:
                # Check error rate
                summary = self.error_handler.get_error_summary()
                
                # Emit health update
                health_info = {
                    'error_rate': summary.get('error_rate', 0),
                    'total_errors': summary.get('total_errors', 0),
                    'critical_errors': summary.get('critical_errors', 0),
                    'timestamp': datetime.now()
                }
                
                self.system_health_update.emit(health_info)
                
                # Check for critical errors
                if summary.get('critical_errors', 0) > 0:
                    self.error_detected.emit({
                        'type': 'critical',
                        'message': f"Critical errors detected: {summary['critical_errors']}",
                        'timestamp': datetime.now()
                    })
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in error monitor: {e}")
                time.sleep(60)  # Wait longer on error
    
    def stop_monitoring(self):
        """Stop error monitoring"""
        self.monitoring = False

# Global error handler instance
_global_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    error_handler = get_error_handler()
    error_handler._handle_uncaught_exception(exc_type, exc_value, exc_traceback)

def handle_error(error: Exception, category: str = 'unknown', 
                show_dialog: bool = True, attempt_recovery: bool = True):
    """Handle error with global error handler"""
    error_handler = get_error_handler()
    error_handler.handle_error(error, category, show_dialog, attempt_recovery)

# Convenience functions
def log_system_info():
    """Log system information for debugging"""
    import platform
    import psutil
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    logger.info(f"System info: {system_info}")
    return system_info

if __name__ == "__main__":
    # Test the error handler
    print("🧪 Testing Error Handler...")
    
    error_handler = get_error_handler()
    
    # Test error handling
    try:
        raise ValueError("Test error for error handler")
    except Exception as e:
        handle_error(e, category='test', show_dialog=False)
    
    # Test error summary
    summary = error_handler.get_error_summary()
    print(f"✅ Error summary: {summary}")
    
    # Test system info
    system_info = log_system_info()
    print(f"✅ System info logged")