"""
Settings Dialog for ChatGPT+ Clone
Comprehensive configuration management with GPU optimization options
"""

import json
import logging
from PyQt6.QtWidgets import (
    QDialog, QCheckBox, QVBoxLayout, QPushButton, QHBoxLayout,
    QTabWidget, QWidget, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QSlider, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from device_utils import get_device_info, is_gpu_available

logger = logging.getLogger("ChatGPTPlus.SettingsDialog")

class SettingsDialog(QDialog):
    """Comprehensive settings dialog with GPU optimization"""
    
    config_saved = pyqtSignal(dict)  # Signal when config is saved
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ChatGPT+ Settings")
        self.setMinimumSize(600, 500)
        
        # Load current config
        self.config = self.load_config()
        
        # Create UI
        self.setup_ui()
        self.load_current_settings()
        
        logger.info("Settings dialog initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs
        self.tab_widget.addTab(self.create_general_tab(), "General")
        self.tab_widget.addTab(self.create_voice_tab(), "Voice")
        self.tab_widget.addTab(self.create_ai_tab(), "AI Models")
        self.tab_widget.addTab(self.create_performance_tab(), "Performance")
        self.tab_widget.addTab(self.create_plugins_tab(), "Plugins")
        self.tab_widget.addTab(self.create_advanced_tab(), "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_config)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_general_tab(self):
        """Create general settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # UI Settings Group
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        ui_layout.addRow("Theme:", self.theme_combo)
        
        self.show_performance_panel = QCheckBox("Show Performance Panel")
        ui_layout.addRow(self.show_performance_panel)
        
        self.auto_save = QCheckBox("Auto-save conversations")
        ui_layout.addRow(self.auto_save)
        
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)
        
        # Window Settings Group
        window_group = QGroupBox("Window Settings")
        window_layout = QFormLayout()
        
        self.startup_minimize = QCheckBox("Start minimized")
        window_layout.addRow(self.startup_minimize)
        
        self.check_updates = QCheckBox("Check for updates")
        window_layout.addRow(self.check_updates)
        
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_voice_tab(self):
        """Create voice settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Voice Input Group
        voice_group = QGroupBox("Voice Input")
        voice_layout = QFormLayout()
        
        self.voice_hotkey_enabled = QCheckBox("Enable Voice Hotkey")
        voice_layout.addRow(self.voice_hotkey_enabled)
        
        self.hotkey_combo = QComboBox()
        self.hotkey_combo.addItems(["Ctrl+Shift+V", "Ctrl+Shift+Space", "F12"])
        voice_layout.addRow("Hotkey:", self.hotkey_combo)
        
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(16000)
        voice_layout.addRow("Sample Rate (Hz):", self.sample_rate_spin)
        
        self.silence_threshold_spin = QDoubleSpinBox()
        self.silence_threshold_spin.setRange(0.001, 0.1)
        self.silence_threshold_spin.setSingleStep(0.001)
        self.silence_threshold_spin.setValue(0.01)
        voice_layout.addRow("Silence Threshold:", self.silence_threshold_spin)
        
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        # TTS Settings Group
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QFormLayout()
        
        self.tts_enabled = QCheckBox("Enable TTS")
        tts_layout.addRow(self.tts_enabled)
        
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.addItems(["Default", "Male", "Female", "Fast"])
        tts_layout.addRow("Voice:", self.tts_voice_combo)
        
        self.tts_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_speed_slider.setRange(50, 200)
        self.tts_speed_slider.setValue(100)
        tts_layout.addRow("Speed:", self.tts_speed_slider)
        
        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_ai_tab(self):
        """Create AI models settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # LLM Settings Group
        llm_group = QGroupBox("Language Model")
        llm_layout = QFormLayout()
        
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.addItems(["gpt2", "dolphin-mixtral:8x22b", "llama2:7b"])
        llm_layout.addRow("Model:", self.llm_model_combo)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)
        llm_layout.addRow("Temperature:", self.temperature_spin)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 4096)
        self.max_tokens_spin.setValue(2048)
        llm_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        
        # Whisper Settings Group
        whisper_group = QGroupBox("Speech Recognition")
        whisper_layout = QFormLayout()
        
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        whisper_layout.addRow("Whisper Model:", self.whisper_model_combo)
        
        whisper_group.setLayout(whisper_layout)
        layout.addWidget(whisper_group)
        
        # Image Generation Group
        image_group = QGroupBox("Image Generation")
        image_layout = QFormLayout()
        
        self.image_model_combo = QComboBox()
        self.image_model_combo.addItems([
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "CompVis/stable-diffusion-v1-4"
        ])
        image_layout.addRow("Model:", self.image_model_combo)
        
        self.image_steps_spin = QSpinBox()
        self.image_steps_spin.setRange(10, 100)
        self.image_steps_spin.setValue(50)
        image_layout.addRow("Inference Steps:", self.image_steps_spin)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_performance_tab(self):
        """Create performance settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # GPU Settings Group
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()
        
        self.gpu_acceleration = QCheckBox("Enable GPU Acceleration")
        gpu_layout.addRow(self.gpu_acceleration)
        
        # Show GPU info
        device_info = get_device_info()
        gpu_info_label = QLabel(f"Detected Device: {device_info.get('current_device', 'Unknown')}")
        gpu_layout.addRow(gpu_info_label)
        
        if is_gpu_available():
            gpu_status_label = QLabel("✅ GPU Available")
            gpu_status_label.setStyleSheet("color: green;")
        else:
            gpu_status_label = QLabel("❌ No GPU Available")
            gpu_status_label.setStyleSheet("color: red;")
        gpu_layout.addRow(gpu_status_label)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # Memory Settings Group
        memory_group = QGroupBox("Memory Management")
        memory_layout = QFormLayout()
        
        self.auto_cleanup = QCheckBox("Auto-cleanup memory")
        memory_layout.addRow(self.auto_cleanup)
        
        self.max_memory_spin = QSpinBox()
        self.max_memory_spin.setRange(1, 32)
        self.max_memory_spin.setValue(8)
        memory_layout.addRow("Max Memory (GB):", self.max_memory_spin)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        # Performance Monitoring Group
        monitor_group = QGroupBox("Performance Monitoring")
        monitor_layout = QFormLayout()
        
        self.monitoring_enabled = QCheckBox("Enable Performance Monitoring")
        monitor_layout.addRow(self.monitoring_enabled)
        
        self.update_interval_spin = QDoubleSpinBox()
        self.update_interval_spin.setRange(0.5, 10.0)
        self.update_interval_spin.setSingleStep(0.5)
        self.update_interval_spin.setValue(1.0)
        monitor_layout.addRow("Update Interval (s):", self.update_interval_spin)
        
        monitor_group.setLayout(monitor_layout)
        layout.addWidget(monitor_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_plugins_tab(self):
        """Create plugins settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Plugin System Group
        plugin_group = QGroupBox("Plugin System")
        plugin_layout = QFormLayout()
        
        self.plugins_enabled = QCheckBox("Enable Plugins")
        plugin_layout.addRow(self.plugins_enabled)
        
        self.plugin_test_mode = QCheckBox("Plugin Test Mode")
        plugin_layout.addRow(self.plugin_test_mode)
        
        self.plugin_hot_reload = QCheckBox("Hot Reload Plugins")
        plugin_layout.addRow(self.plugin_hot_reload)
        
        self.plugin_sandbox = QCheckBox("Sandbox Plugins")
        plugin_layout.addRow(self.plugin_sandbox)
        
        plugin_group.setLayout(plugin_layout)
        layout.addWidget(plugin_group)
        
        # Plugin Log Group
        log_group = QGroupBox("Plugin Logging")
        log_layout = QFormLayout()
        
        self.plugin_debug_logging = QCheckBox("Debug Logging")
        log_layout.addRow(self.plugin_debug_logging)
        
        self.plugin_log_display = QTextEdit()
        self.plugin_log_display.setMaximumHeight(150)
        self.plugin_log_display.setReadOnly(True)
        log_layout.addRow("Plugin Log:", self.plugin_log_display)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # AR Overlay Group
        ar_group = QGroupBox("AR Overlay")
        ar_layout = QFormLayout()
        
        self.ar_overlay_enabled = QCheckBox("Enable AR Overlay")
        ar_layout.addRow(self.ar_overlay_enabled)
        
        self.ar_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ar_opacity_slider.setRange(10, 100)
        self.ar_opacity_slider.setValue(80)
        ar_layout.addRow("Opacity:", self.ar_opacity_slider)
        
        ar_group.setLayout(ar_layout)
        layout.addWidget(ar_group)
        
        # Security Group
        security_group = QGroupBox("Security")
        security_layout = QFormLayout()
        
        self.sandbox_enabled = QCheckBox("Enable Code Sandbox")
        security_layout.addRow(self.sandbox_enabled)
        
        self.code_timeout_spin = QSpinBox()
        self.code_timeout_spin.setRange(5, 60)
        self.code_timeout_spin.setValue(10)
        security_layout.addRow("Code Timeout (s):", self.code_timeout_spin)
        
        security_group.setLayout(security_layout)
        layout.addWidget(security_group)
        
        # Debug Group
        debug_group = QGroupBox("Debug")
        debug_layout = QFormLayout()
        
        self.debug_mode = QCheckBox("Debug Mode")
        debug_layout.addRow(self.debug_mode)
        
        self.show_technical_info = QCheckBox("Show Technical Info")
        debug_layout.addRow(self.show_technical_info)
        
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open("config.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            "voice_hotkey_enabled": True,
            "ar_overlay_enabled": True,
            "plugins_enabled": True,
            "plugin_test_mode": False,
            "gpu_acceleration": True,
            "voice_settings": {
                "hotkey": "ctrl+shift+v",
                "sample_rate": 16000,
                "silence_threshold": 0.01,
                "silence_duration": 2.0
            },
            "ai_models": {
                "whisper_model": "base",
                "llm_model": "gpt2",
                "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
                "image_model": "runwayml/stable-diffusion-v1-5"
            },
            "performance": {
                "monitoring_enabled": True,
                "update_interval": 1.0,
                "alert_thresholds": {
                    "cpu_percent": 80.0,
                    "memory_percent": 85.0,
                    "gpu_memory_percent": 90.0
                }
            },
            "security": {
                "sandbox_enabled": True,
                "code_execution_timeout": 10,
                "max_memory_usage": 512
            },
            "ui": {
                "theme": "dark",
                "window_size": [1200, 800],
                "show_performance_panel": True
            }
        }
    
    def load_current_settings(self):
        """Load current settings into UI controls"""
        try:
            # General settings
            self.theme_combo.setCurrentText(self.config.get("ui", {}).get("theme", "dark").title())
            self.show_performance_panel.setChecked(self.config.get("ui", {}).get("show_performance_panel", True))
            self.auto_save.setChecked(self.config.get("app_settings", {}).get("auto_save_chat", True))
            self.startup_minimize.setChecked(self.config.get("app_settings", {}).get("startup_minimize", False))
            self.check_updates.setChecked(self.config.get("app_settings", {}).get("check_updates", True))
            
            # Voice settings
            self.voice_hotkey_enabled.setChecked(self.config.get("voice_hotkey_enabled", True))
            hotkey = self.config.get("voice_settings", {}).get("hotkey", "ctrl+shift+v")
            self.hotkey_combo.setCurrentText(hotkey.replace("+", "+").title())
            self.sample_rate_spin.setValue(self.config.get("voice_settings", {}).get("sample_rate", 16000))
            self.silence_threshold_spin.setValue(self.config.get("voice_settings", {}).get("silence_threshold", 0.01))
            self.tts_enabled.setChecked(self.config.get("voice_settings", {}).get("tts_enabled", True))
            self.tts_voice_combo.setCurrentText(self.config.get("voice_settings", {}).get("tts_voice", "Default"))
            self.tts_speed_slider.setValue(self.config.get("voice_settings", {}).get("tts_speed", 100))
            
            # AI settings
            self.llm_model_combo.setCurrentText(self.config.get("ai_models", {}).get("llm_model", "gpt2"))
            self.temperature_spin.setValue(self.config.get("llm_settings", {}).get("temperature", 0.7))
            self.max_tokens_spin.setValue(self.config.get("llm_settings", {}).get("max_tokens", 2048))
            self.whisper_model_combo.setCurrentText(self.config.get("ai_models", {}).get("whisper_model", "base"))
            self.image_model_combo.setCurrentText(self.config.get("ai_models", {}).get("image_model", "runwayml/stable-diffusion-v1-5"))
            self.image_steps_spin.setValue(self.config.get("image_settings", {}).get("inference_steps", 50))
            
            # Performance settings
            self.gpu_acceleration.setChecked(self.config.get("gpu_acceleration", True))
            self.auto_cleanup.setChecked(self.config.get("memory_settings", {}).get("auto_cleanup", True))
            self.max_memory_spin.setValue(self.config.get("memory_settings", {}).get("max_memory_size", "8GB").replace("GB", ""))
            self.monitoring_enabled.setChecked(self.config.get("performance", {}).get("monitoring_enabled", True))
            self.update_interval_spin.setValue(self.config.get("performance", {}).get("update_interval", 1.0))
            
            # Plugin settings
            self.plugins_enabled.setChecked(self.config.get("plugins_enabled", True))
            self.plugin_test_mode.setChecked(self.config.get("plugin_test_mode", False))
            self.plugin_hot_reload.setChecked(self.config.get("app_settings", {}).get("plugin_hot_reload", True))
            self.plugin_sandbox.setChecked(self.config.get("app_settings", {}).get("plugin_sandbox", True))
            self.plugin_debug_logging.setChecked(self.config.get("development", {}).get("plugin_debug_logging", False))
            
            # Advanced settings
            self.ar_overlay_enabled.setChecked(self.config.get("ar_overlay_enabled", True))
            self.ar_opacity_slider.setValue(self.config.get("ui_settings", {}).get("ar_opacity", 80))
            self.sandbox_enabled.setChecked(self.config.get("security", {}).get("sandbox_enabled", True))
            self.code_timeout_spin.setValue(self.config.get("security", {}).get("code_execution_timeout", 10))
            self.debug_mode.setChecked(self.config.get("development", {}).get("dev_mode", False))
            self.show_technical_info.setChecked(self.config.get("app_settings", {}).get("show_technical_info", False))
            
        except Exception as e:
            logger.error(f"Failed to load current settings: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            # Build new config from UI controls
            new_config = self.build_config_from_ui()
            
            # Save to file
            with open("config.json", 'w') as f:
                json.dump(new_config, f, indent=4)
            
            logger.info("Configuration saved successfully")
            
            # Emit signal
            self.config_saved.emit(new_config)
            
            self.accept()
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def apply_settings(self):
        """Apply settings without closing dialog"""
        try:
            new_config = self.build_config_from_ui()
            
            # Save to file
            with open("config.json", 'w') as f:
                json.dump(new_config, f, indent=4)
            
            logger.info("Settings applied")
            
            # Emit signal
            self.config_saved.emit(new_config)
            
        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")
    
    def build_config_from_ui(self):
        """Build configuration dictionary from UI controls"""
        return {
            "voice_hotkey_enabled": self.voice_hotkey_enabled.isChecked(),
            "ar_overlay_enabled": self.ar_overlay_enabled.isChecked(),
            "plugins_enabled": self.plugins_enabled.isChecked(),
            "plugin_test_mode": self.plugin_test_mode.isChecked(),
            "gpu_acceleration": self.gpu_acceleration.isChecked(),
            "voice_settings": {
                "hotkey": self.hotkey_combo.currentText().lower().replace("+", "+"),
                "sample_rate": self.sample_rate_spin.value(),
                "silence_threshold": self.silence_threshold_spin.value(),
                "silence_duration": 2.0,
                "tts_enabled": self.tts_enabled.isChecked(),
                "tts_voice": self.tts_voice_combo.currentText(),
                "tts_speed": self.tts_speed_slider.value()
            },
            "ai_models": {
                "whisper_model": self.whisper_model_combo.currentText(),
                "llm_model": self.llm_model_combo.currentText(),
                "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
                "image_model": self.image_model_combo.currentText()
            },
            "llm_settings": {
                "temperature": self.temperature_spin.value(),
                "max_tokens": self.max_tokens_spin.value()
            },
            "performance": {
                "monitoring_enabled": self.monitoring_enabled.isChecked(),
                "update_interval": self.update_interval_spin.value(),
                "alert_thresholds": {
                    "cpu_percent": 80.0,
                    "memory_percent": 85.0,
                    "gpu_memory_percent": 90.0
                }
            },
            "security": {
                "sandbox_enabled": self.sandbox_enabled.isChecked(),
                "code_execution_timeout": self.code_timeout_spin.value(),
                "max_memory_usage": 512
            },
            "ui": {
                "theme": self.theme_combo.currentText().lower(),
                "window_size": [1200, 800],
                "show_performance_panel": self.show_performance_panel.isChecked()
            },
            "app_settings": {
                "auto_save_chat": self.auto_save.isChecked(),
                "startup_minimize": self.startup_minimize.isChecked(),
                "check_updates": self.check_updates.isChecked(),
                "plugin_hot_reload": self.plugin_hot_reload.isChecked(),
                "plugin_sandbox": self.plugin_sandbox.isChecked(),
                "show_technical_info": self.show_technical_info.isChecked()
            },
            "development": {
                "dev_mode": self.debug_mode.isChecked(),
                "plugin_debug_logging": self.plugin_debug_logging.isChecked()
            },
            "ui_settings": {
                "ar_opacity": self.ar_opacity_slider.value()
            }
        }