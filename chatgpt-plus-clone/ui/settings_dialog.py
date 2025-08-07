"""
Settings Dialog - Application Configuration Interface
Provides settings for voice, plugins, and application preferences
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                              QCheckBox, QPushButton, QLabel, QLineEdit, 
                              QSpinBox, QComboBox, QGroupBox, QTextEdit,
                              QSlider, QFormLayout, QScrollArea, QWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Dict, Any
import json
from pathlib import Path
from config_manager import ConfigManager

class SettingsDialog(QDialog):
    """Main settings dialog for ChatGPT+ Clone"""
    
    # Signals
    settings_changed = pyqtSignal(dict)
    voice_hotkey_toggled = pyqtSignal(bool)  # Signal for live voice hotkey toggle
    
    def __init__(self, current_settings: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self.config = ConfigManager()
        self.current_settings = current_settings or self.load_default_settings()
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Settings - ChatGPT+ Clone")
        self.setGeometry(200, 200, 600, 500)
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Voice settings tab
        self.voice_tab = self.create_voice_tab()
        self.tab_widget.addTab(self.voice_tab, "🎤 Voice")
        
        # Plugin settings tab
        self.plugin_tab = self.create_plugin_tab()
        self.tab_widget.addTab(self.plugin_tab, "🔌 Plugins")
        
        # Application settings tab
        self.app_tab = self.create_app_tab()
        self.tab_widget.addTab(self.app_tab, "⚙️ Application")
        
        # Advanced settings tab
        self.advanced_tab = self.create_advanced_tab()
        self.tab_widget.addTab(self.advanced_tab, "🔧 Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
        """)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def create_voice_tab(self) -> QWidget:
        """Create voice settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Voice activation
        voice_group = QGroupBox("Voice Activation")
        voice_layout = QFormLayout(voice_group)
        
        self.voice_enabled = QCheckBox("Enable voice hotkey (Ctrl+Shift+V)")
        self.voice_enabled.setChecked(True)
        self.voice_enabled.stateChanged.connect(self.on_voice_hotkey_toggled)
        voice_layout.addRow("Voice Hotkey:", self.voice_enabled)
        
        self.voice_hotkey = QLineEdit("ctrl+shift+v")
        voice_layout.addRow("Hotkey:", self.voice_hotkey)
        
        # Recording settings
        recording_group = QGroupBox("Recording Settings")
        recording_layout = QFormLayout(recording_group)
        
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["8000", "16000", "22050", "44100"])
        self.sample_rate.setCurrentText("16000")
        recording_layout.addRow("Sample Rate:", self.sample_rate)
        
        self.recording_duration = QSpinBox()
        self.recording_duration.setRange(1, 60)
        self.recording_duration.setValue(5)
        recording_layout.addRow("Recording Duration (s):", self.recording_duration)
        
        self.silence_threshold = QSlider(Qt.Orientation.Horizontal)
        self.silence_threshold.setRange(1, 100)
        self.silence_threshold.setValue(10)
        recording_layout.addRow("Silence Threshold:", self.silence_threshold)
        
        # TTS settings
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QFormLayout(tts_group)
        
        self.tts_enabled = QCheckBox("Enable TTS responses")
        self.tts_enabled.setChecked(True)
        tts_layout.addRow("TTS Enabled:", self.tts_enabled)
        
        self.tts_engine = QComboBox()
        self.tts_engine.addItems(["TTS", "Bark", "Fallback"])
        self.tts_engine.setCurrentText("TTS")
        tts_layout.addRow("TTS Engine:", self.tts_engine)
        
        self.tts_voice = QComboBox()
        self.tts_voice.addItems(["Default", "Male", "Female", "Neutral"])
        self.tts_voice.setCurrentText("Default")
        tts_layout.addRow("Voice:", self.tts_voice)
        
        # Add groups to layout
        layout.addWidget(voice_group)
        layout.addWidget(recording_group)
        layout.addWidget(tts_group)
        layout.addStretch()
        
        return widget
    
    def create_plugin_tab(self) -> QWidget:
        """Create plugin settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Plugin management
        plugin_group = QGroupBox("Plugin Management")
        plugin_layout = QVBoxLayout(plugin_group)
        
        self.plugin_hot_reload = QCheckBox("Enable hot-reload (auto-reload plugins on changes)")
        self.plugin_hot_reload.setChecked(True)
        plugin_layout.addWidget(self.plugin_hot_reload)
        
        self.plugin_debug = QCheckBox("Enable plugin debug logging")
        self.plugin_debug.setChecked(False)
        plugin_layout.addWidget(self.plugin_debug)
        
        self.plugin_sandbox = QCheckBox("Enable plugin sandboxing (security)")
        self.plugin_sandbox.setChecked(True)
        plugin_layout.addWidget(self.plugin_sandbox)
        
        # Plugin directory
        dir_group = QGroupBox("Plugin Directory")
        dir_layout = QFormLayout(dir_group)
        
        self.plugin_directory = QLineEdit("plugins")
        dir_layout.addRow("Directory:", self.plugin_directory)
        
        # Plugin status
        status_group = QGroupBox("Plugin Status")
        status_layout = QVBoxLayout(status_group)
        
        self.plugin_status = QTextEdit()
        self.plugin_status.setMaximumHeight(100)
        self.plugin_status.setReadOnly(True)
        self.plugin_status.setPlaceholderText("Plugin status will appear here...")
        status_layout.addWidget(self.plugin_status)
        
        # Add groups to layout
        layout.addWidget(plugin_group)
        layout.addWidget(dir_group)
        layout.addWidget(status_group)
        layout.addStretch()
        
        return widget
    
    def create_app_tab(self) -> QWidget:
        """Create application settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # General settings
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout(general_group)
        
        self.auto_save = QCheckBox("Auto-save conversations")
        self.auto_save.setChecked(True)
        general_layout.addRow("Auto Save:", self.auto_save)
        
        self.startup_minimize = QCheckBox("Start minimized to system tray")
        self.startup_minimize.setChecked(False)
        general_layout.addRow("Start Minimized:", self.startup_minimize)
        
        self.check_updates = QCheckBox("Check for updates on startup")
        self.check_updates.setChecked(True)
        general_layout.addRow("Check Updates:", self.check_updates)
        
        # UI settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout(ui_group)
        
        self.theme = QComboBox()
        self.theme.addItems(["Light", "Dark", "System"])
        self.theme.setCurrentText("System")
        ui_layout.addRow("Theme:", self.theme)
        
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setValue(12)
        ui_layout.addRow("Font Size:", self.font_size)
        
        self.window_opacity = QSlider(Qt.Orientation.Horizontal)
        self.window_opacity.setRange(50, 100)
        self.window_opacity.setValue(100)
        ui_layout.addRow("Window Opacity (%):", self.window_opacity)
        
        # Memory settings
        memory_group = QGroupBox("Memory & Storage")
        memory_layout = QFormLayout(memory_group)
        
        self.max_conversations = QSpinBox()
        self.max_conversations.setRange(10, 1000)
        self.max_conversations.setValue(100)
        memory_layout.addRow("Max Conversations:", self.max_conversations)
        
        self.max_memory_size = QComboBox()
        self.max_memory_size.addItems(["256MB", "512MB", "1GB", "2GB", "4GB"])
        self.max_memory_size.setCurrentText("1GB")
        memory_layout.addRow("Max Memory:", self.max_memory_size)
        
        # Add groups to layout
        layout.addWidget(general_group)
        layout.addWidget(ui_group)
        layout.addWidget(memory_group)
        layout.addStretch()
        
        return widget
    
    def create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # AI Model settings
        model_group = QGroupBox("AI Model Settings")
        model_layout = QFormLayout(model_group)
        
        self.default_model = QComboBox()
        self.default_model.addItems([
            "dolphin-mixtral:8x22b",
            "llama2:13b", 
            "mistral:7b",
            "codellama:13b",
            "neural-chat:7b"
        ])
        self.default_model.setCurrentText("dolphin-mixtral:8x22b")
        model_layout.addRow("Default Model:", self.default_model)
        
        self.model_temperature = QSlider(Qt.Orientation.Horizontal)
        self.model_temperature.setRange(0, 100)
        self.model_temperature.setValue(70)
        model_layout.addRow("Temperature:", self.model_temperature)
        
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(100, 4000)
        self.max_tokens.setValue(2048)
        model_layout.addRow("Max Tokens:", self.max_tokens)
        
        # Network settings
        network_group = QGroupBox("Network Settings")
        network_layout = QFormLayout(network_group)
        
        self.ollama_url = QLineEdit("http://localhost:11434")
        network_layout.addRow("Ollama URL:", self.ollama_url)
        
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 300)
        self.timeout.setValue(30)
        network_layout.addRow("Timeout (s):", self.timeout)
        
        self.retry_attempts = QSpinBox()
        self.retry_attempts.setRange(1, 10)
        self.retry_attempts.setValue(3)
        network_layout.addRow("Retry Attempts:", self.retry_attempts)
        
        # Debug settings
        debug_group = QGroupBox("Debug Settings")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_mode = QCheckBox("Enable debug mode")
        debug_layout.addWidget(self.debug_mode)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        self.log_level.setCurrentText("INFO")
        debug_layout.addWidget(QLabel("Log Level:"))
        debug_layout.addWidget(self.log_level)
        
        self.show_technical_info = QCheckBox("Show technical information in UI")
        debug_layout.addWidget(self.show_technical_info)
        
        # Add groups to layout
        layout.addWidget(model_group)
        layout.addWidget(network_group)
        layout.addWidget(debug_group)
        layout.addStretch()
        
        return widget
    
    def load_settings(self):
        """Load current settings into UI"""
        # Load from config manager
        app_settings = self.config.get_app_settings()
        voice_settings = self.config.get_voice_settings()
        ui_settings = self.config.get_ui_settings()
        llm_settings = self.config.get_llm_settings()
        
        # Voice settings
        self.voice_enabled.setChecked(voice_settings.get('enabled', True))
        self.voice_hotkey.setText(voice_settings.get('hotkey', 'ctrl+shift+v'))
        self.sample_rate.setCurrentText(str(voice_settings.get('sample_rate', 16000)))
        self.recording_duration.setValue(voice_settings.get('recording_duration', 5))
        self.silence_threshold.setValue(voice_settings.get('silence_threshold', 10))
        self.tts_enabled.setChecked(voice_settings.get('tts_enabled', True))
        self.tts_engine.setCurrentText(voice_settings.get('tts_engine', 'TTS'))
        self.tts_voice.setCurrentText(voice_settings.get('tts_voice', 'Default'))
        
        # Plugin settings
        self.plugin_hot_reload.setChecked(app_settings.get('plugin_hot_reload', True))
        self.plugin_debug.setChecked(app_settings.get('plugin_debug', False))
        self.plugin_sandbox.setChecked(app_settings.get('plugin_sandbox', True))
        self.plugin_directory.setText(app_settings.get('plugin_directory', 'plugins'))
        
        # App settings
        self.auto_save.setChecked(app_settings.get('auto_save', True))
        self.startup_minimize.setChecked(app_settings.get('startup_minimize', False))
        self.check_updates.setChecked(app_settings.get('check_updates', True))
        self.theme.setCurrentText(ui_settings.get('theme', 'System'))
        self.font_size.setValue(ui_settings.get('font_size', 12))
        self.window_opacity.setValue(ui_settings.get('window_opacity', 100))
        self.max_conversations.setValue(app_settings.get('max_conversations', 100))
        self.max_memory_size.setCurrentText(app_settings.get('max_memory_size', '1GB'))
        
        # Advanced settings
        self.default_model.setCurrentText(llm_settings.get('default_model', 'dolphin-mixtral:8x22b'))
        self.model_temperature.setValue(llm_settings.get('temperature', 70))
        self.max_tokens.setValue(llm_settings.get('max_tokens', 2048))
        self.ollama_url.setText(llm_settings.get('ollama_url', 'http://localhost:11434'))
        self.timeout.setValue(llm_settings.get('timeout', 30))
        self.retry_attempts.setValue(llm_settings.get('retry_attempts', 3))
        self.debug_mode.setChecked(app_settings.get('debug_mode', False))
        self.log_level.setCurrentText(app_settings.get('log_level', 'INFO'))
        self.show_technical_info.setChecked(ui_settings.get('show_technical_info', False))
    
    def save_settings(self):
        """Save current UI settings"""
        settings = {
            # Voice settings
            'voice_enabled': self.voice_enabled.isChecked(),
            'voice_hotkey': self.voice_hotkey.text(),
            'sample_rate': int(self.sample_rate.currentText()),
            'recording_duration': self.recording_duration.value(),
            'silence_threshold': self.silence_threshold.value(),
            'tts_enabled': self.tts_enabled.isChecked(),
            'tts_engine': self.tts_engine.currentText(),
            'tts_voice': self.tts_voice.currentText(),
            
            # Plugin settings
            'plugin_hot_reload': self.plugin_hot_reload.isChecked(),
            'plugin_debug': self.plugin_debug.isChecked(),
            'plugin_sandbox': self.plugin_sandbox.isChecked(),
            'plugin_directory': self.plugin_directory.text(),
            
            # App settings
            'auto_save': self.auto_save.isChecked(),
            'startup_minimize': self.startup_minimize.isChecked(),
            'check_updates': self.check_updates.isChecked(),
            'theme': self.theme.currentText(),
            'font_size': self.font_size.value(),
            'window_opacity': self.window_opacity.value(),
            'max_conversations': self.max_conversations.value(),
            'max_memory_size': self.max_memory_size.currentText(),
            
            # Advanced settings
            'default_model': self.default_model.currentText(),
            'model_temperature': self.model_temperature.value(),
            'max_tokens': self.max_tokens.value(),
            'ollama_url': self.ollama_url.text(),
            'timeout': self.timeout.value(),
            'retry_attempts': self.retry_attempts.value(),
            'debug_mode': self.debug_mode.isChecked(),
            'log_level': self.log_level.currentText(),
            'show_technical_info': self.show_technical_info.isChecked(),
        }
        
        # Save to file
        self.save_settings_to_file(settings)
        
        # Emit signal
        self.settings_changed.emit(settings)
        
        # Close dialog
        self.accept()
    
    def reset_settings(self):
        """Reset settings to defaults"""
        self.current_settings = self.load_default_settings()
        self.load_settings()
    
    def load_default_settings(self) -> Dict[str, Any]:
        """Load default settings"""
        return {
            'voice_enabled': True,
            'voice_hotkey': 'ctrl+shift+v',
            'sample_rate': 16000,
            'recording_duration': 5,
            'silence_threshold': 10,
            'tts_enabled': True,
            'tts_engine': 'TTS',
            'tts_voice': 'Default',
            'plugin_hot_reload': True,
            'plugin_debug': False,
            'plugin_sandbox': True,
            'plugin_directory': 'plugins',
            'auto_save': True,
            'startup_minimize': False,
            'check_updates': True,
            'theme': 'System',
            'font_size': 12,
            'window_opacity': 100,
            'max_conversations': 100,
            'max_memory_size': '1GB',
            'default_model': 'dolphin-mixtral:8x22b',
            'model_temperature': 70,
            'max_tokens': 2048,
            'ollama_url': 'http://localhost:11434',
            'timeout': 30,
            'retry_attempts': 3,
            'debug_mode': False,
            'log_level': 'INFO',
            'show_technical_info': False,
        }
    
    def save_settings_to_file(self, settings: Dict[str, Any]):
        """Save settings to JSON file"""
        try:
            config_path = Path("config.json")
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def on_voice_hotkey_toggled(self, state):
        """Handle voice hotkey toggle and emit signal"""
        enabled = state == Qt.CheckState.Checked
        self.voice_hotkey_toggled.emit(enabled)
        print(f"[Settings] Voice hotkey toggled: {enabled}")
    
    def update_plugin_status(self, status_text: str):
        """Update plugin status display"""
        self.plugin_status.setText(status_text)