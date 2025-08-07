"""
Settings Dialog - Application Configuration Management
Provides a comprehensive settings interface with live toggles and persistence
"""

import logging
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                              QCheckBox, QPushButton, QLabel, QSpinBox,
                              QComboBox, QLineEdit, QGroupBox, QFormLayout,
                              QSlider, QTextEdit, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from config_manager import ConfigManager

class SettingsDialog(QDialog):
    """Comprehensive settings dialog with live toggles"""
    
    # Signals for live updates
    settings_changed = pyqtSignal(dict)
    voice_hotkey_toggled = pyqtSignal(bool)
    toggle_changed = pyqtSignal(str, bool)
    
    def __init__(self, main_app=None, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Setup UI
        self.setup_ui()
        self.load_settings()
        self.connect_signals()
        
        self.logger.info("Settings dialog initialized")
    
    def setup_ui(self):
        """Setup the settings dialog UI"""
        self.setWindowTitle("ChatGPT+ Clone Settings")
        self.setGeometry(200, 200, 800, 600)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_general_tab()
        self.create_voice_tab()
        self.create_plugins_tab()
        self.create_ar_overlay_tab()
        self.create_advanced_tab()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_settings)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #545B62;
            }
        """)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
        """)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
    
    def create_general_tab(self):
        """Create general settings tab"""
        general_widget = QWidget()
        layout = QVBoxLayout(general_widget)
        
        # Application settings group
        app_group = QGroupBox("Application Settings")
        app_layout = QFormLayout(app_group)
        
        # General toggles
        self.auto_save_chat = QCheckBox("Auto-save chat sessions")
        app_layout.addRow("Chat Auto-save:", self.auto_save_chat)
        
        self.startup_minimize = QCheckBox("Start minimized")
        app_layout.addRow("Startup Behavior:", self.startup_minimize)
        
        self.check_updates = QCheckBox("Check for updates")
        app_layout.addRow("Updates:", self.check_updates)
        
        self.debug_mode = QCheckBox("Enable debug mode")
        app_layout.addRow("Debug Mode:", self.debug_mode)
        
        self.show_technical_info = QCheckBox("Show technical information")
        app_layout.addRow("Technical Info:", self.show_technical_info)
        
        layout.addWidget(app_group)
        
        # Memory settings group
        memory_group = QGroupBox("Memory Settings")
        memory_layout = QFormLayout(memory_group)
        
        self.enable_memory_system = QCheckBox("Enable memory system")
        memory_layout.addRow("Memory System:", self.enable_memory_system)
        
        self.memory_limit = QSpinBox()
        self.memory_limit.setRange(100, 10000)
        self.memory_limit.setSuffix(" MB")
        memory_layout.addRow("Memory Limit:", self.memory_limit)
        
        layout.addWidget(memory_group)
        
        # File settings group
        file_group = QGroupBox("File Settings")
        file_layout = QFormLayout(file_group)
        
        self.enable_file_upload = QCheckBox("Enable file upload")
        file_layout.addRow("File Upload:", self.enable_file_upload)
        
        self.max_file_size = QSpinBox()
        self.max_file_size.setRange(1, 1000)
        self.max_file_size.setSuffix(" MB")
        file_layout.addRow("Max File Size:", self.max_file_size)
        
        layout.addWidget(file_group)
        
        layout.addStretch()
        self.tab_widget.addTab(general_widget, "General")
    
    def create_voice_tab(self):
        """Create voice settings tab"""
        voice_widget = QWidget()
        layout = QVBoxLayout(voice_widget)
        
        # Voice hotkey settings
        hotkey_group = QGroupBox("Voice Hotkey Settings")
        hotkey_layout = QFormLayout(hotkey_group)
        
        self.voice_hotkey_enabled = QCheckBox("Enable voice hotkey")
        hotkey_layout.addRow("Voice Hotkey:", self.voice_hotkey_enabled)
        
        self.voice_hotkey_combo = QComboBox()
        self.voice_hotkey_combo.addItems([
            "Ctrl+Shift+V", "Ctrl+Alt+V", "F12", "Ctrl+Shift+Space"
        ])
        hotkey_layout.addRow("Hotkey:", self.voice_hotkey_combo)
        
        layout.addWidget(hotkey_group)
        
        # Voice recording settings
        recording_group = QGroupBox("Recording Settings")
        recording_layout = QFormLayout(recording_group)
        
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["8000", "16000", "22050", "44100"])
        recording_layout.addRow("Sample Rate:", self.sample_rate)
        
        self.recording_duration = QSpinBox()
        self.recording_duration.setRange(1, 30)
        self.recording_duration.setSuffix(" seconds")
        recording_layout.addRow("Recording Duration:", self.recording_duration)
        
        self.silence_threshold = QSpinBox()
        self.silence_threshold.setRange(1, 50)
        recording_layout.addRow("Silence Threshold:", self.silence_threshold)
        
        layout.addWidget(recording_group)
        
        # TTS settings
        tts_group = QGroupBox("Text-to-Speech Settings")
        tts_layout = QFormLayout(tts_group)
        
        self.tts_enabled = QCheckBox("Enable TTS")
        tts_layout.addRow("TTS Enabled:", self.tts_enabled)
        
        self.tts_voice = QComboBox()
        self.tts_voice.addItems(["Default", "Male", "Female", "Robot"])
        tts_layout.addRow("TTS Voice:", self.tts_voice)
        
        self.tts_speed = QSlider(Qt.Orientation.Horizontal)
        self.tts_speed.setRange(50, 200)
        self.tts_speed.setValue(100)
        tts_layout.addRow("TTS Speed:", self.tts_speed)
        
        layout.addWidget(tts_group)
        
        layout.addStretch()
        self.tab_widget.addTab(voice_widget, "Voice")
    
    def create_plugins_tab(self):
        """Create plugins settings tab"""
        plugins_widget = QWidget()
        layout = QVBoxLayout(plugins_widget)
        
        # Plugin system settings
        system_group = QGroupBox("Plugin System")
        system_layout = QFormLayout(system_group)
        
        self.enable_plugins = QCheckBox("Enable plugin system")
        system_layout.addRow("Plugin System:", self.enable_plugins)
        
        self.plugin_hot_reload = QCheckBox("Enable hot-reload")
        system_layout.addRow("Hot Reload:", self.plugin_hot_reload)
        
        self.plugin_sandbox = QCheckBox("Enable sandboxing")
        system_layout.addRow("Sandboxing:", self.plugin_sandbox)
        
        layout.addWidget(system_group)
        
        # Plugin debug settings
        debug_group = QGroupBox("Plugin Debug")
        debug_layout = QFormLayout(debug_group)
        
        self.plugin_debug_logging = QCheckBox("Enable debug logging")
        debug_layout.addRow("Debug Logging:", self.plugin_debug_logging)
        
        self.plugin_error_notifications = QCheckBox("Show error notifications")
        debug_layout.addRow("Error Notifications:", self.plugin_error_notifications)
        
        layout.addWidget(debug_group)
        
        # Plugin configuration area
        config_group = QGroupBox("Plugin Configuration")
        config_layout = QVBoxLayout(config_group)
        
        self.plugin_config_text = QTextEdit()
        self.plugin_config_text.setPlaceholderText("Plugin configuration will appear here...")
        self.plugin_config_text.setMaximumHeight(200)
        config_layout.addWidget(self.plugin_config_text)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        self.tab_widget.addTab(plugins_widget, "Plugins")
    
    def create_ar_overlay_tab(self):
        """Create AR overlay settings tab"""
        ar_widget = QWidget()
        layout = QVBoxLayout(ar_widget)
        
        # AR overlay settings
        ar_group = QGroupBox("AR Overlay Settings")
        ar_layout = QFormLayout(ar_group)
        
        self.enable_ar_overlay = QCheckBox("Enable AR overlay")
        ar_layout.addRow("AR Overlay:", self.enable_ar_overlay)
        
        self.ar_opacity = QSlider(Qt.Orientation.Horizontal)
        self.ar_opacity.setRange(10, 100)
        self.ar_opacity.setValue(80)
        ar_layout.addRow("Opacity:", self.ar_opacity)
        
        self.ar_position = QComboBox()
        self.ar_position.addItems(["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Center"])
        ar_layout.addRow("Position:", self.ar_position)
        
        layout.addWidget(ar_group)
        
        # Visual effects settings
        effects_group = QGroupBox("Visual Effects")
        effects_layout = QFormLayout(effects_group)
        
        self.ar_particles = QCheckBox("Enable particle effects")
        effects_layout.addRow("Particles:", self.ar_particles)
        
        self.ar_neural_network = QCheckBox("Show neural network")
        effects_layout.addRow("Neural Network:", self.ar_neural_network)
        
        self.ar_data_flow = QCheckBox("Show data flow")
        effects_layout.addRow("Data Flow:", self.ar_data_flow)
        
        self.ar_holographic_text = QCheckBox("Show holographic text")
        effects_layout.addRow("Holographic Text:", self.ar_holographic_text)
        
        layout.addWidget(effects_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.ar_fps = QComboBox()
        self.ar_fps.addItems(["30 FPS", "60 FPS", "120 FPS"])
        perf_layout.addRow("Target FPS:", self.ar_fps)
        
        self.ar_quality = QComboBox()
        self.ar_quality.addItems(["Low", "Medium", "High", "Ultra"])
        perf_layout.addRow("Quality:", self.ar_quality)
        
        layout.addWidget(perf_group)
        
        layout.addStretch()
        self.tab_widget.addTab(ar_widget, "AR Overlay")
    
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        advanced_widget = QWidget()
        layout = QVBoxLayout(advanced_widget)
        
        # Logging settings
        logging_group = QGroupBox("Logging Settings")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        logging_layout.addRow("Log Level:", self.log_level)
        
        self.log_to_file = QCheckBox("Log to file")
        logging_layout.addRow("File Logging:", self.log_to_file)
        
        self.log_to_console = QCheckBox("Log to console")
        logging_layout.addRow("Console Logging:", self.log_to_console)
        
        layout.addWidget(logging_group)
        
        # Error handling settings
        error_group = QGroupBox("Error Handling")
        error_layout = QFormLayout(error_group)
        
        self.show_error_dialogs = QCheckBox("Show error dialogs")
        error_layout.addRow("Error Dialogs:", self.show_error_dialogs)
        
        self.auto_restart_on_crash = QCheckBox("Auto-restart on crash")
        error_layout.addRow("Auto Restart:", self.auto_restart_on_crash)
        
        self.save_crash_reports = QCheckBox("Save crash reports")
        error_layout.addRow("Crash Reports:", self.save_crash_reports)
        
        layout.addWidget(error_group)
        
        # Development settings
        dev_group = QGroupBox("Development")
        dev_layout = QFormLayout(dev_group)
        
        self.dev_mode = QCheckBox("Enable developer mode")
        dev_layout.addRow("Developer Mode:", self.dev_mode)
        
        self.show_debug_info = QCheckBox("Show debug information")
        dev_layout.addRow("Debug Info:", self.show_debug_info)
        
        layout.addWidget(dev_group)
        
        layout.addStretch()
        self.tab_widget.addTab(advanced_widget, "Advanced")
    
    def connect_signals(self):
        """Connect all signal handlers"""
        # Connect toggle signals
        self.voice_hotkey_enabled.stateChanged.connect(self.on_voice_hotkey_toggled)
        self.enable_plugins.stateChanged.connect(self.on_plugins_toggled)
        self.enable_ar_overlay.stateChanged.connect(self.on_ar_overlay_toggled)
        
        # Connect other settings changes
        self.auto_save_chat.stateChanged.connect(self.on_setting_changed)
        self.debug_mode.stateChanged.connect(self.on_setting_changed)
        self.enable_memory_system.stateChanged.connect(self.on_setting_changed)
        self.enable_file_upload.stateChanged.connect(self.on_setting_changed)
    
    def load_settings(self):
        """Load current settings from config"""
        try:
            # Load app settings
            app_settings = self.config.get_app_settings()
            self.auto_save_chat.setChecked(app_settings.get("auto_save_chat", True))
            self.startup_minimize.setChecked(app_settings.get("startup_minimize", False))
            self.check_updates.setChecked(app_settings.get("check_updates", True))
            self.debug_mode.setChecked(app_settings.get("debug_mode", False))
            self.show_technical_info.setChecked(app_settings.get("show_technical_info", False))
            self.enable_memory_system.setChecked(app_settings.get("enable_memory_system", True))
            self.enable_file_upload.setChecked(app_settings.get("enable_file_upload", True))
            
            # Load voice settings
            voice_settings = self.config.get_voice_settings()
            self.voice_hotkey_enabled.setChecked(voice_settings.get("enabled", True))
            self.sample_rate.setCurrentText(str(voice_settings.get("sample_rate", 16000)))
            self.recording_duration.setValue(voice_settings.get("recording_duration", 5))
            self.silence_threshold.setValue(voice_settings.get("silence_threshold", 10))
            self.tts_enabled.setChecked(voice_settings.get("tts_enabled", True))
            
            # Load UI settings
            ui_settings = self.config.get_ui_settings()
            self.enable_ar_overlay.setChecked(ui_settings.get("enable_ar_overlay", False))
            self.ar_opacity.setValue(int(ui_settings.get("ar_opacity", 80)))
            self.ar_position.setCurrentText(ui_settings.get("ar_position", "Top-Right"))
            
            # Load plugin settings
            self.enable_plugins.setChecked(app_settings.get("enable_plugins", True))
            self.plugin_hot_reload.setChecked(app_settings.get("plugin_hot_reload", True))
            self.plugin_sandbox.setChecked(app_settings.get("plugin_sandbox", True))
            self.plugin_debug_logging.setChecked(app_settings.get("plugin_debug_logging", False))
            
            self.logger.info("Settings loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save current settings to config"""
        try:
            # Collect app settings
            app_settings = {
                "auto_save_chat": self.auto_save_chat.isChecked(),
                "startup_minimize": self.startup_minimize.isChecked(),
                "check_updates": self.check_updates.isChecked(),
                "debug_mode": self.debug_mode.isChecked(),
                "show_technical_info": self.show_technical_info.isChecked(),
                "enable_memory_system": self.enable_memory_system.isChecked(),
                "enable_file_upload": self.enable_file_upload.isChecked(),
                "enable_plugins": self.enable_plugins.isChecked(),
                "plugin_hot_reload": self.plugin_hot_reload.isChecked(),
                "plugin_sandbox": self.plugin_sandbox.isChecked(),
                "plugin_debug_logging": self.plugin_debug_logging.isChecked(),
            }
            
            # Collect voice settings
            voice_settings = {
                "enabled": self.voice_hotkey_enabled.isChecked(),
                "hotkey": self.voice_hotkey_combo.currentText(),
                "sample_rate": int(self.sample_rate.currentText()),
                "recording_duration": self.recording_duration.value(),
                "silence_threshold": self.silence_threshold.value(),
                "tts_enabled": self.tts_enabled.isChecked(),
                "tts_voice": self.tts_voice.currentText(),
                "tts_speed": self.tts_speed.value(),
            }
            
            # Collect UI settings
            ui_settings = {
                "enable_ar_overlay": self.enable_ar_overlay.isChecked(),
                "ar_opacity": self.ar_opacity.value(),
                "ar_position": self.ar_position.currentText(),
                "ar_particles": self.ar_particles.isChecked(),
                "ar_neural_network": self.ar_neural_network.isChecked(),
                "ar_data_flow": self.ar_data_flow.isChecked(),
                "ar_holographic_text": self.ar_holographic_text.isChecked(),
            }
            
            # Save to config
            self.config.set_app_settings(app_settings)
            self.config.set_voice_settings(voice_settings)
            self.config.set_ui_settings(ui_settings)
            
            # Emit settings changed signal
            all_settings = {**app_settings, **voice_settings, **ui_settings}
            self.settings_changed.emit(all_settings)
            
            self.logger.info("Settings saved successfully")
            QMessageBox.information(self, "Success", "Settings saved successfully!")
            
            self.accept()
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        try:
            reply = QMessageBox.question(
                self, "Reset Settings",
                "Are you sure you want to reset all settings to defaults?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Reset config to defaults
                self.config.reset_to_defaults()
                
                # Reload settings
                self.load_settings()
                
                self.logger.info("Settings reset to defaults")
                QMessageBox.information(self, "Success", "Settings reset to defaults!")
                
        except Exception as e:
            self.logger.error(f"Error resetting settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset settings: {e}")
    
    def on_voice_hotkey_toggled(self, state):
        """Handle voice hotkey toggle"""
        enabled = state == Qt.CheckState.Checked
        self.voice_hotkey_toggled.emit(enabled)
        self.toggle_changed.emit("voice_hotkey_enabled", enabled)
        self.logger.info(f"Voice hotkey toggled: {enabled}")
    
    def on_plugins_toggled(self, state):
        """Handle plugins toggle"""
        enabled = state == Qt.CheckState.Checked
        self.toggle_changed.emit("enable_plugins", enabled)
        self.logger.info(f"Plugins toggled: {enabled}")
    
    def on_ar_overlay_toggled(self, state):
        """Handle AR overlay toggle"""
        enabled = state == Qt.CheckState.Checked
        self.toggle_changed.emit("enable_ar_overlay", enabled)
        self.logger.info(f"AR overlay toggled: {enabled}")
    
    def on_setting_changed(self):
        """Handle any setting change"""
        self.logger.debug("Setting changed")
        # This could trigger auto-save or other actions