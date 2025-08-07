#!/usr/bin/env python3
"""
ChatGPT+ Clone - Main Application Entry Point
Comprehensive AI assistant with plugins, voice, AR overlay, and live toggles
"""

import sys
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                              QLabel, QWidget, QPushButton, QMenuBar, QAction,
                              QStatusBar, QMessageBox, QSplitter, QTextEdit)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

# Import our modules
from plugin_loader import load_plugins, SandboxPlugin, start_plugin_watcher
from voice_hotkey import VoiceHotkeyListener
from overlay_ar import AROverlay
from error_handler import error_handler, handle_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

class MainApp(QMainWindow):
    """Main application window with comprehensive feature management"""
    
    # Signals for UI updates
    status_updated = pyqtSignal(str)
    plugin_status_changed = pyqtSignal(str, bool)
    voice_status_changed = pyqtSignal(bool)
    ar_overlay_status_changed = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT+ Clone - AI Assistant")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.plugins = []
        self.plugin_watcher = None
        self.voice_listener = None
        self.ar_overlay = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Initialize systems based on config
        self.init_voice_hotkey()
        self.init_ar_overlay()
        self.init_plugins()
        self.init_plugin_watcher()
        
        # Connect signals
        self.connect_signals()
        
        # Log system info
        error_handler.log_system_info()
        
        logger.info("ChatGPT+ Clone initialized successfully")
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Status display
        self.status_label = QLabel("System initializing...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                background-color: #1E1E1E;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.voice_btn = QPushButton("🎤 Voice Hotkey")
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        self.voice_btn.clicked.connect(self.toggle_voice_hotkey)
        button_layout.addWidget(self.voice_btn)
        
        self.ar_btn = QPushButton("🧠 AR Overlay")
        self.ar_btn.setStyleSheet("""
            QPushButton {
                background-color: #68217A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4A1B5A;
            }
        """)
        self.ar_btn.clicked.connect(self.toggle_ar_overlay)
        button_layout.addWidget(self.ar_btn)
        
        self.plugin_btn = QPushButton("🔌 Plugins")
        self.plugin_btn.setStyleSheet("""
            QPushButton {
                background-color: #107C10;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0B5A0B;
            }
        """)
        self.plugin_btn.clicked.connect(self.toggle_plugins)
        button_layout.addWidget(self.plugin_btn)
        
        main_layout.addLayout(button_layout)
        
        # Chat area (placeholder)
        self.chat_area = QTextEdit()
        self.chat_area.setPlaceholderText("Chat interface will be implemented here...")
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', monospace;
            }
        """)
        main_layout.addWidget(self.chat_area)
        
        # Plugin status area
        self.plugin_status = QLabel("Plugin Status: Loading...")
        self.plugin_status.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-weight: bold;
                padding: 5px;
                background-color: #1E1E1E;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.plugin_status)
    
    def setup_menu(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        plugin_test_action = QAction('Plugin Test', self)
        plugin_test_action.triggered.connect(self.open_plugin_test)
        tools_menu.addAction(plugin_test_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready")
        
        # Add memory usage indicator
        self.memory_label = QLabel("Memory: --")
        self.statusBar().addPermanentWidget(self.memory_label)
        
        # Update memory usage periodically
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds
    
    def connect_signals(self):
        """Connect all signal handlers"""
        self.status_updated.connect(self.status_label.setText)
        self.plugin_status_changed.connect(self.update_plugin_status)
        self.voice_status_changed.connect(self.update_voice_status)
        self.ar_overlay_status_changed.connect(self.update_ar_status)
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if Path(CONFIG_FILE).exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info("Configuration loaded successfully")
                return config
            else:
                logger.warning("Config file not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "enable_voice_hotkey": True,
            "enable_ar_overlay": True,
            "enable_plugins": True,
            "plugins_config": {},
            "voice_settings": {
                "hotkey": "ctrl+shift+v",
                "sample_rate": 16000,
                "recording_duration": 5
            },
            "ar_settings": {
                "opacity": 0.8,
                "position": "top-right"
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            handle_error(e, "Config save", "UI")
    
    # Voice hotkey management
    def init_voice_hotkey(self):
        """Initialize voice hotkey system"""
        if self.config.get("enable_voice_hotkey", True):
            self.start_voice_listener()
        else:
            self.update_status("Voice hotkey disabled")
    
    def start_voice_listener(self):
        """Start voice hotkey listener"""
        try:
            if not self.voice_listener:
                self.voice_listener = VoiceHotkeyListener()
                self.voice_listener.start()
            
            self.config["enable_voice_hotkey"] = True
            self.save_config()
            self.update_status("Voice hotkey enabled")
            self.voice_status_changed.emit(True)
            logger.info("Voice hotkey started")
            
        except Exception as e:
            logger.error(f"Failed to start voice hotkey: {e}")
            handle_error(e, "Voice hotkey start", "Voice")
    
    def stop_voice_listener(self):
        """Stop voice hotkey listener"""
        try:
            if self.voice_listener:
                self.voice_listener.stop()
                self.voice_listener = None
            
            self.config["enable_voice_hotkey"] = False
            self.save_config()
            self.update_status("Voice hotkey disabled")
            self.voice_status_changed.emit(False)
            logger.info("Voice hotkey stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop voice hotkey: {e}")
            handle_error(e, "Voice hotkey stop", "Voice")
    
    def toggle_voice_hotkey(self):
        """Toggle voice hotkey on/off"""
        if self.voice_listener:
            self.stop_voice_listener()
        else:
            self.start_voice_listener()
    
    # AR overlay management
    def init_ar_overlay(self):
        """Initialize AR overlay"""
        if self.config.get("enable_ar_overlay", True):
            self.enable_ar_overlay()
        else:
            self.update_status("AR overlay disabled")
    
    def enable_ar_overlay(self):
        """Enable AR overlay"""
        try:
            if not self.ar_overlay:
                self.ar_overlay = AROverlay()
                self.ar_overlay.start()
            
            self.config["enable_ar_overlay"] = True
            self.save_config()
            self.update_status("AR overlay enabled")
            self.ar_overlay_status_changed.emit(True)
            logger.info("AR overlay started")
            
        except Exception as e:
            logger.error(f"Failed to start AR overlay: {e}")
            handle_error(e, "AR overlay start", "UI")
    
    def disable_ar_overlay(self):
        """Disable AR overlay"""
        try:
            if self.ar_overlay:
                self.ar_overlay.stop()
                self.ar_overlay = None
            
            self.config["enable_ar_overlay"] = False
            self.save_config()
            self.update_status("AR overlay disabled")
            self.ar_overlay_status_changed.emit(False)
            logger.info("AR overlay stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop AR overlay: {e}")
            handle_error(e, "AR overlay stop", "UI")
    
    def toggle_ar_overlay(self):
        """Toggle AR overlay on/off"""
        if self.ar_overlay:
            self.disable_ar_overlay()
        else:
            self.enable_ar_overlay()
    
    # Plugin management
    def init_plugins(self):
        """Initialize plugin system"""
        if self.config.get("enable_plugins", True):
            self.enable_plugins()
        else:
            self.update_status("Plugins disabled")
    
    def enable_plugins(self):
        """Enable all plugins"""
        try:
            if not self.plugins:
                self.plugins = load_plugins()
            
            for plugin in self.plugins:
                plugin.start()
            
            self.config["enable_plugins"] = True
            self.save_config()
            self.update_status(f"Plugins enabled ({len(self.plugins)} loaded)")
            self.plugin_status_changed.emit("enabled", True)
            logger.info(f"Plugins enabled: {len(self.plugins)} loaded")
            
        except Exception as e:
            logger.error(f"Failed to enable plugins: {e}")
            handle_error(e, "Plugin enable", "Plugin")
    
    def disable_plugins(self):
        """Disable all plugins"""
        try:
            for plugin in self.plugins:
                plugin.stop()
            
            self.plugins = []
            self.config["enable_plugins"] = False
            self.save_config()
            self.update_status("Plugins disabled")
            self.plugin_status_changed.emit("disabled", False)
            logger.info("Plugins disabled")
            
        except Exception as e:
            logger.error(f"Failed to disable plugins: {e}")
            handle_error(e, "Plugin disable", "Plugin")
    
    def toggle_plugins(self):
        """Toggle plugins on/off"""
        if self.plugins:
            self.disable_plugins()
        else:
            self.enable_plugins()
    
    # Plugin watcher
    def init_plugin_watcher(self):
        """Initialize plugin file watcher"""
        try:
            self.plugin_watcher = start_plugin_watcher(self)
            logger.info("Plugin watcher started")
        except Exception as e:
            logger.error(f"Failed to start plugin watcher: {e}")
            handle_error(e, "Plugin watcher", "Plugin")
    
    # UI update methods
    def update_status(self, message):
        """Update status display"""
        self.status_updated.emit(message)
        logger.info(f"Status: {message}")
    
    def update_plugin_status(self, status, enabled):
        """Update plugin status display"""
        self.plugin_status.setText(f"Plugin Status: {status}")
        self.plugin_btn.setText(f"🔌 Plugins ({len(self.plugins) if enabled else 0})")
    
    def update_voice_status(self, enabled):
        """Update voice status display"""
        self.voice_btn.setText("🎤 Voice Hotkey (ON)" if enabled else "🎤 Voice Hotkey (OFF)")
    
    def update_ar_status(self, enabled):
        """Update AR overlay status display"""
        self.ar_btn.setText("🧠 AR Overlay (ON)" if enabled else "🧠 AR Overlay (OFF)")
    
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        except ImportError:
            self.memory_label.setText("Memory: --")
    
    # Menu actions
    def open_settings(self):
        """Open settings dialog"""
        try:
            from ui.settings_dialog import SettingsDialog
            dialog = SettingsDialog(self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open settings: {e}")
            handle_error(e, "Settings dialog", "UI")
    
    def open_plugin_test(self):
        """Open plugin test dialog"""
        try:
            from ui.plugin_test_dialog import PluginTestDialog
            dialog = PluginTestDialog(self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open plugin test: {e}")
            handle_error(e, "Plugin test dialog", "UI")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About ChatGPT+ Clone",
                         "ChatGPT+ Clone v1.0\n\n"
                         "A comprehensive AI assistant with:\n"
                         "• Plugin system with sandboxing\n"
                         "• Voice hotkey support\n"
                         "• AR overlay interface\n"
                         "• Live configuration toggles\n\n"
                         "Built with PyQt6 and Python")
    
    # Plugin config helpers
    def save_plugin_config(self, plugin_name, config_dict):
        """Save plugin-specific configuration"""
        all_configs = self.config.get("plugins_config", {})
        all_configs[plugin_name] = config_dict
        self.config["plugins_config"] = all_configs
        self.save_config()
        logger.info(f"Plugin config saved for {plugin_name}")
    
    def load_plugin_config(self, plugin_name):
        """Load plugin-specific configuration"""
        return self.config.get("plugins_config", {}).get(plugin_name, {})
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            logger.info("Application shutting down...")
            
            # Stop all systems
            if self.voice_listener:
                self.stop_voice_listener()
            
            if self.ar_overlay:
                self.disable_ar_overlay()
            
            if self.plugins:
                self.disable_plugins()
            
            if self.plugin_watcher:
                self.plugin_watcher.stop()
            
            # Save final config
            self.save_config()
            
            # Create final log entry
            logger.info("Application shutdown complete")
            
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            handle_error(e, "Application shutdown", "General")
            event.accept()

def main():
    """Main application entry point"""
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("ChatGPT+ Clone")
        app.setApplicationVersion("1.0.0")
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create and show main window
        window = MainApp()
        window.show()
        
        logger.info("ChatGPT+ Clone started successfully")
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        handle_error(e, "Application startup", "General")
        sys.exit(1)

if __name__ == "__main__":
    main()