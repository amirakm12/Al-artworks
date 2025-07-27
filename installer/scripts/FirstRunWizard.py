#!/usr/bin/env python3
"""
AI-ARTWORKS First Run Wizard
Post-installation configuration wizard with Qt6 interface
"""

import sys
import os
import json
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import platform
import winreg
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QComboBox,
    QProgressBar, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy,
    QMessageBox, QFileDialog, QTabWidget, QWidget, QSlider,
    QSpinBox, QListWidget, QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt, QThread, QTimer, pyqtSignal, QSettings
from PySide6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

@dataclass
class GPUInfo:
    name: str
    vendor: str
    memory_mb: int
    supports_cuda: bool
    supports_directml: bool

class GPUDetectionThread(QThread):
    """Background thread for GPU detection and system scanning"""
    
    gpu_detected = pyqtSignal(list)  # List of GPUInfo
    progress_updated = pyqtSignal(int, str)
    
    def run(self):
        """Detect available GPUs and their capabilities"""
        self.progress_updated.emit(10, "Scanning system hardware...")
        
        gpus = []
        
        try:
            # Windows GPU detection via WMI
            if platform.system() == "Windows":
                gpus.extend(self._detect_windows_gpus())
            
            self.progress_updated.emit(50, "Checking CUDA availability...")
            self._check_cuda_support(gpus)
            
            self.progress_updated.emit(80, "Verifying DirectML support...")
            self._check_directml_support(gpus)
            
            self.progress_updated.emit(100, "GPU detection completed")
            
        except Exception as e:
            self.progress_updated.emit(100, f"GPU detection failed: {str(e)}")
        
        self.gpu_detected.emit(gpus)
    
    def _detect_windows_gpus(self) -> list:
        """Detect GPUs on Windows using WMI"""
        gpus = []
        
        try:
            import wmi
            c = wmi.WMI()
            
            for gpu in c.Win32_VideoController():
                if gpu.Name and "Basic" not in gpu.Name:
                    memory_mb = 0
                    if gpu.AdapterRAM:
                        memory_mb = int(gpu.AdapterRAM) // (1024 * 1024)
                    
                    vendor = "Unknown"
                    supports_cuda = False
                    
                    if any(x in gpu.Name.upper() for x in ["NVIDIA", "GEFORCE", "QUADRO", "TESLA"]):
                        vendor = "NVIDIA"
                        supports_cuda = True
                    elif any(x in gpu.Name.upper() for x in ["AMD", "RADEON"]):
                        vendor = "AMD"
                    elif "INTEL" in gpu.Name.upper():
                        vendor = "Intel"
                    
                    gpu_info = GPUInfo(
                        name=gpu.Name,
                        vendor=vendor,
                        memory_mb=memory_mb,
                        supports_cuda=supports_cuda,
                        supports_directml=True  # Assume modern DirectX support
                    )
                    gpus.append(gpu_info)
                    
        except ImportError:
            # Fallback without WMI
            self.progress_updated.emit(30, "WMI not available, using registry...")
            gpus.extend(self._detect_gpus_registry())
        
        return gpus
    
    def _detect_gpus_registry(self) -> list:
        """Fallback GPU detection using Windows registry"""
        gpus = []
        
        try:
            # Check display adapters in registry
            key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        if subkey_name.isdigit():
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    driver_desc = winreg.QueryValueEx(subkey, "DriverDesc")[0]
                                    if "Basic" not in driver_desc:
                                        vendor = "Unknown"
                                        supports_cuda = False
                                        
                                        if any(x in driver_desc.upper() for x in ["NVIDIA", "GEFORCE"]):
                                            vendor = "NVIDIA"
                                            supports_cuda = True
                                        elif "AMD" in driver_desc.upper() or "RADEON" in driver_desc.upper():
                                            vendor = "AMD"
                                        elif "INTEL" in driver_desc.upper():
                                            vendor = "Intel"
                                        
                                        gpu_info = GPUInfo(
                                            name=driver_desc,
                                            vendor=vendor,
                                            memory_mb=0,  # Unknown from registry
                                            supports_cuda=supports_cuda,
                                            supports_directml=True
                                        )
                                        gpus.append(gpu_info)
                                except FileNotFoundError:
                                    pass
                        i += 1
                    except OSError:
                        break
        except Exception:
            pass
        
        return gpus
    
    def _check_cuda_support(self, gpus: list):
        """Check CUDA installation and compatibility"""
        cuda_available = False
        
        # Check CUDA environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            cuda_available = True
        
        # Check for CUDA runtime
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cuda_available = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Update GPU info with actual CUDA availability
        for gpu in gpus:
            if gpu.vendor == "NVIDIA":
                gpu.supports_cuda = gpu.supports_cuda and cuda_available
    
    def _check_directml_support(self, gpus: list):
        """Check DirectML support (Windows 10 1903+)"""
        try:
            import winver
            version = winver.get_winver()
            directml_supported = (version.major >= 10 and 
                                version.build >= 18362)  # Windows 10 1903
            
            for gpu in gpus:
                gpu.supports_directml = gpu.supports_directml and directml_supported
        except ImportError:
            # Assume DirectML is supported on modern systems
            pass

class WelcomePage(QWizardPage):
    """Welcome page with system information"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to AI-ARTWORKS")
        self.setSubTitle("Let's configure your AI creative studio for optimal performance")
        
        layout = QVBoxLayout()
        
        # Welcome message
        welcome_label = QLabel(
            "Welcome to AI-ARTWORKS Ultimate Creative Studio!\n\n"
            "This wizard will help you configure the application for your system, "
            "set up API keys, choose AI models, and optimize performance settings.\n\n"
            "The setup process will take just a few minutes."
        )
        welcome_label.setWordWrap(True)
        welcome_label.setStyleSheet("QLabel { font-size: 12px; margin: 10px; }")
        
        # System info group
        system_group = QGroupBox("System Information")
        system_layout = QGridLayout()
        
        system_layout.addWidget(QLabel("Operating System:"), 0, 0)
        system_layout.addWidget(QLabel(f"{platform.system()} {platform.release()}"), 0, 1)
        
        system_layout.addWidget(QLabel("Architecture:"), 1, 0)
        system_layout.addWidget(QLabel(platform.machine()), 1, 1)
        
        system_layout.addWidget(QLabel("Python Version:"), 2, 0)
        system_layout.addWidget(QLabel(f"{sys.version.split()[0]}"), 2, 1)
        
        system_group.setLayout(system_layout)
        
        layout.addWidget(welcome_label)
        layout.addWidget(system_group)
        layout.addStretch()
        
        self.setLayout(layout)

class GPUConfigPage(QWizardPage):
    """GPU detection and configuration page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("GPU Configuration")
        self.setSubTitle("Detecting and configuring graphics acceleration")
        
        self.gpus = []
        self.detection_thread = None
        
        layout = QVBoxLayout()
        
        # Detection progress
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Starting GPU detection...")
        
        # GPU list
        self.gpu_list = QListWidget()
        self.gpu_list.setMinimumHeight(150)
        
        # Configuration options
        config_group = QGroupBox("Acceleration Settings")
        config_layout = QVBoxLayout()
        
        self.enable_cuda = QCheckBox("Enable CUDA acceleration (NVIDIA)")
        self.enable_directml = QCheckBox("Enable DirectML acceleration (AMD/Intel)")
        self.enable_cpu_fallback = QCheckBox("Use CPU fallback if GPU unavailable")
        self.enable_cpu_fallback.setChecked(True)
        
        config_layout.addWidget(self.enable_cuda)
        config_layout.addWidget(self.enable_directml)
        config_layout.addWidget(self.enable_cpu_fallback)
        config_group.setLayout(config_layout)
        
        # Rescan button
        self.rescan_button = QPushButton("Rescan Hardware")
        self.rescan_button.clicked.connect(self.start_detection)
        
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Detected GPUs:"))
        layout.addWidget(self.gpu_list)
        layout.addWidget(config_group)
        layout.addWidget(self.rescan_button)
        
        self.setLayout(layout)
    
    def initializePage(self):
        """Start GPU detection when page is shown"""
        self.start_detection()
    
    def start_detection(self):
        """Start GPU detection in background thread"""
        if self.detection_thread and self.detection_thread.isRunning():
            return
        
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting GPU detection...")
        self.gpu_list.clear()
        
        self.detection_thread = GPUDetectionThread()
        self.detection_thread.gpu_detected.connect(self.on_gpus_detected)
        self.detection_thread.progress_updated.connect(self.on_progress_updated)
        self.detection_thread.start()
    
    def on_progress_updated(self, value: int, message: str):
        """Update progress bar and message"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_gpus_detected(self, gpus: list):
        """Handle detected GPUs"""
        self.gpus = gpus
        self.gpu_list.clear()
        
        if not gpus:
            item = QListWidgetItem("No dedicated GPUs detected - CPU processing will be used")
            item.setIcon(QIcon.fromTheme("computer"))
            self.gpu_list.addItem(item)
            self.enable_cuda.setEnabled(False)
            self.enable_directml.setEnabled(False)
        else:
            cuda_available = any(gpu.supports_cuda for gpu in gpus)
            directml_available = any(gpu.supports_directml for gpu in gpus)
            
            self.enable_cuda.setEnabled(cuda_available)
            self.enable_cuda.setChecked(cuda_available)
            self.enable_directml.setEnabled(directml_available)
            self.enable_directml.setChecked(directml_available and not cuda_available)
            
            for gpu in gpus:
                memory_text = f" ({gpu.memory_mb}MB)" if gpu.memory_mb > 0 else ""
                item_text = f"{gpu.vendor} {gpu.name}{memory_text}"
                
                features = []
                if gpu.supports_cuda:
                    features.append("CUDA")
                if gpu.supports_directml:
                    features.append("DirectML")
                
                if features:
                    item_text += f" - {', '.join(features)}"
                
                item = QListWidgetItem(item_text)
                
                # Set icon based on vendor
                if gpu.vendor == "NVIDIA":
                    item.setIcon(QIcon.fromTheme("nvidia"))
                elif gpu.vendor == "AMD":
                    item.setIcon(QIcon.fromTheme("amd"))
                elif gpu.vendor == "Intel":
                    item.setIcon(QIcon.fromTheme("intel"))
                else:
                    item.setIcon(QIcon.fromTheme("gpu"))
                
                self.gpu_list.addItem(item)
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration settings"""
        return {
            'enable_cuda': self.enable_cuda.isChecked(),
            'enable_directml': self.enable_directml.isChecked(),
            'enable_cpu_fallback': self.enable_cpu_fallback.isChecked(),
            'detected_gpus': [
                {
                    'name': gpu.name,
                    'vendor': gpu.vendor,
                    'memory_mb': gpu.memory_mb,
                    'supports_cuda': gpu.supports_cuda,
                    'supports_directml': gpu.supports_directml
                }
                for gpu in self.gpus
            ]
        }

class ModelSelectionPage(QWizardPage):
    """AI model selection and download page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("AI Model Selection")
        self.setSubTitle("Choose which AI models to download and configure")
        
        layout = QVBoxLayout()
        
        # Model categories
        tabs = QTabWidget()
        
        # Image Generation Models
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        
        self.stable_diffusion = QCheckBox("Stable Diffusion 1.5 (4GB) - General purpose image generation")
        self.stable_diffusion.setChecked(True)
        
        self.stable_diffusion_xl = QCheckBox("Stable Diffusion XL (7GB) - High resolution images")
        
        self.controlnet = QCheckBox("ControlNet (2GB) - Guided image generation")
        
        image_layout.addWidget(self.stable_diffusion)
        image_layout.addWidget(self.stable_diffusion_xl)
        image_layout.addWidget(self.controlnet)
        image_layout.addStretch()
        image_tab.setLayout(image_layout)
        
        # Voice/Audio Models
        audio_tab = QWidget()
        audio_layout = QVBoxLayout()
        
        self.whisper_base = QCheckBox("Whisper Base (290MB) - Speech recognition")
        self.whisper_base.setChecked(True)
        
        self.whisper_large = QCheckBox("Whisper Large (3GB) - High accuracy speech recognition")
        
        self.bark_voice = QCheckBox("Bark (2GB) - Text-to-speech synthesis")
        
        audio_layout.addWidget(self.whisper_base)
        audio_layout.addWidget(self.whisper_large)
        audio_layout.addWidget(self.bark_voice)
        audio_layout.addStretch()
        audio_tab.setLayout(audio_layout)
        
        # Language Models
        text_tab = QWidget()
        text_layout = QVBoxLayout()
        
        self.llama_7b = QCheckBox("LLaMA 7B (13GB) - Large language model")
        
        self.code_llama = QCheckBox("Code Llama (13GB) - Code generation model")
        
        text_layout.addWidget(self.llama_7b)
        text_layout.addWidget(self.code_llama)
        text_layout.addStretch()
        text_tab.setLayout(text_layout)
        
        tabs.addTab(image_tab, "Image Generation")
        tabs.addTab(audio_tab, "Voice & Audio")
        tabs.addTab(text_tab, "Language Models")
        
        # Storage info
        storage_group = QGroupBox("Storage Information")
        storage_layout = QVBoxLayout()
        
        self.storage_label = QLabel("Selected models will require approximately 0 GB of storage")
        self.update_storage_info()
        
        # Connect checkboxes to update storage info
        for checkbox in [self.stable_diffusion, self.stable_diffusion_xl, self.controlnet,
                        self.whisper_base, self.whisper_large, self.bark_voice,
                        self.llama_7b, self.code_llama]:
            checkbox.toggled.connect(self.update_storage_info)
        
        storage_layout.addWidget(self.storage_label)
        storage_group.setLayout(storage_layout)
        
        # Download later option
        self.download_later = QCheckBox("Skip model download - I'll download them later")
        
        layout.addWidget(tabs)
        layout.addWidget(storage_group)
        layout.addWidget(self.download_later)
        
        self.setLayout(layout)
    
    def update_storage_info(self):
        """Update storage requirements based on selected models"""
        total_gb = 0
        
        model_sizes = {
            self.stable_diffusion: 4,
            self.stable_diffusion_xl: 7,
            self.controlnet: 2,
            self.whisper_base: 0.3,
            self.whisper_large: 3,
            self.bark_voice: 2,
            self.llama_7b: 13,
            self.code_llama: 13
        }
        
        for checkbox, size in model_sizes.items():
            if checkbox.isChecked():
                total_gb += size
        
        self.storage_label.setText(f"Selected models will require approximately {total_gb:.1f} GB of storage")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model selection configuration"""
        return {
            'download_models': not self.download_later.isChecked(),
            'models': {
                'stable_diffusion_15': self.stable_diffusion.isChecked(),
                'stable_diffusion_xl': self.stable_diffusion_xl.isChecked(),
                'controlnet': self.controlnet.isChecked(),
                'whisper_base': self.whisper_base.isChecked(),
                'whisper_large': self.whisper_large.isChecked(),
                'bark_voice': self.bark_voice.isChecked(),
                'llama_7b': self.llama_7b.isChecked(),
                'code_llama': self.code_llama.isChecked()
            }
        }

class APIConfigPage(QWizardPage):
    """API keys and service configuration page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("API Configuration")
        self.setSubTitle("Configure API keys for external services (optional)")
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(
            "API keys are optional but enable additional features like cloud model access "
            "and enhanced capabilities. You can configure these later in the application settings."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; margin-bottom: 10px; }")
        
        # API configuration
        api_group = QGroupBox("API Keys")
        api_layout = QGridLayout()
        
        # OpenAI API
        api_layout.addWidget(QLabel("OpenAI API Key:"), 0, 0)
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.Password)
        self.openai_key.setPlaceholderText("sk-...")
        api_layout.addWidget(self.openai_key, 0, 1)
        
        # Hugging Face API
        api_layout.addWidget(QLabel("Hugging Face Token:"), 1, 0)
        self.hf_token = QLineEdit()
        self.hf_token.setEchoMode(QLineEdit.Password)
        self.hf_token.setPlaceholderText("hf_...")
        api_layout.addWidget(self.hf_token, 1, 1)
        
        # Stability AI API
        api_layout.addWidget(QLabel("Stability AI Key:"), 2, 0)
        self.stability_key = QLineEdit()
        self.stability_key.setEchoMode(QLineEdit.Password)
        self.stability_key.setPlaceholderText("sk-...")
        api_layout.addWidget(self.stability_key, 2, 1)
        
        api_group.setLayout(api_layout)
        
        # Voice settings
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QGridLayout()
        
        voice_layout.addWidget(QLabel("Default Voice:"), 0, 0)
        self.voice_selection = QComboBox()
        self.voice_selection.addItems(["Female (Aria)", "Male (Davis)", "Neutral (Sam)"])
        voice_layout.addWidget(self.voice_selection, 0, 1)
        
        voice_layout.addWidget(QLabel("Speech Rate:"), 1, 0)
        self.speech_rate = QSlider(Qt.Horizontal)
        self.speech_rate.setRange(50, 200)
        self.speech_rate.setValue(100)
        self.rate_label = QLabel("100%")
        self.speech_rate.valueChanged.connect(lambda v: self.rate_label.setText(f"{v}%"))
        voice_layout.addWidget(self.speech_rate, 1, 1)
        voice_layout.addWidget(self.rate_label, 1, 2)
        
        voice_group.setLayout(voice_layout)
        
        layout.addWidget(info_label)
        layout.addWidget(api_group)
        layout.addWidget(voice_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            'api_keys': {
                'openai': self.openai_key.text().strip(),
                'huggingface': self.hf_token.text().strip(),
                'stability_ai': self.stability_key.text().strip()
            },
            'voice_settings': {
                'default_voice': self.voice_selection.currentText(),
                'speech_rate': self.speech_rate.value()
            }
        }

class CompletionPage(QWizardPage):
    """Final configuration page"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Setup Complete")
        self.setSubTitle("AI-ARTWORKS is ready to use!")
        
        layout = QVBoxLayout()
        
        # Completion message
        completion_label = QLabel(
            "Congratulations! AI-ARTWORKS has been successfully configured for your system.\n\n"
            "Your settings have been saved and the application is ready to use. "
            "You can modify these settings at any time through the application preferences."
        )
        completion_label.setWordWrap(True)
        completion_label.setStyleSheet("QLabel { font-size: 12px; margin: 10px; }")
        
        # Quick start options
        start_group = QGroupBox("Quick Start")
        start_layout = QVBoxLayout()
        
        self.launch_app = QCheckBox("Launch AI-ARTWORKS now")
        self.launch_app.setChecked(True)
        
        self.show_tutorial = QCheckBox("Show tutorial on first launch")
        self.show_tutorial.setChecked(True)
        
        self.create_sample = QCheckBox("Create a sample project")
        
        start_layout.addWidget(self.launch_app)
        start_layout.addWidget(self.show_tutorial)
        start_layout.addWidget(self.create_sample)
        start_group.setLayout(start_layout)
        
        # Resources
        resources_group = QGroupBox("Resources")
        resources_layout = QVBoxLayout()
        
        resources_text = QLabel(
            "• Documentation: Help → User Guide\n"
            "• Video Tutorials: Help → Video Tutorials\n"
            "• Community: Help → Community Forum\n"
            "• Support: Help → Contact Support"
        )
        resources_layout.addWidget(resources_text)
        resources_group.setLayout(resources_layout)
        
        layout.addWidget(completion_label)
        layout.addWidget(start_group)
        layout.addWidget(resources_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def get_completion_config(self) -> Dict[str, Any]:
        """Get completion configuration"""
        return {
            'launch_app': self.launch_app.isChecked(),
            'show_tutorial': self.show_tutorial.isChecked(),
            'create_sample': self.create_sample.isChecked()
        }

class FirstRunWizard(QWizard):
    """Main first-run configuration wizard"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AI-ARTWORKS First Run Setup")
        self.setFixedSize(700, 500)
        self.setWizardStyle(QWizard.ModernStyle)
        
        # Set up pages
        self.welcome_page = WelcomePage()
        self.gpu_page = GPUConfigPage()
        self.model_page = ModelSelectionPage()
        self.api_page = APIConfigPage()
        self.completion_page = CompletionPage()
        
        self.addPage(self.welcome_page)
        self.addPage(self.gpu_page)
        self.addPage(self.model_page)
        self.addPage(self.api_page)
        self.addPage(self.completion_page)
        
        # Configure wizard
        self.setOption(QWizard.HaveHelpButton, True)
        self.setOption(QWizard.HelpButtonOnRight, False)
        
        # Connect signals
        self.helpRequested.connect(self.show_help)
        self.finished.connect(self.on_finished)
        
        # Apply styling
        self.apply_theme()
    
    def apply_theme(self):
        """Apply modern theme to the wizard"""
        self.setStyleSheet("""
            QWizard {
                background-color: #f5f5f5;
            }
            QWizard QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
        """)
    
    def show_help(self):
        """Show context-sensitive help"""
        current_id = self.currentId()
        help_text = {
            0: "Welcome to AI-ARTWORKS! This wizard will guide you through the initial setup process.",
            1: "GPU configuration optimizes performance for AI processing. CUDA provides the best performance for NVIDIA GPUs.",
            2: "Select AI models based on your intended use. You can always download additional models later.",
            3: "API keys enable cloud features and additional model access. These are optional for basic functionality.",
            4: "Setup is complete! You can modify these settings later in the application preferences."
        }
        
        QMessageBox.information(self, "Help", help_text.get(current_id, "No help available for this page."))
    
    def on_finished(self, result):
        """Handle wizard completion"""
        if result == QWizard.Accepted:
            self.save_configuration()
            
            completion_config = self.completion_page.get_completion_config()
            if completion_config.get('launch_app', False):
                self.launch_application()
    
    def save_configuration(self):
        """Save all configuration to files"""
        try:
            # Get configuration from all pages
            config = {
                'gpu': self.gpu_page.get_gpu_config(),
                'models': self.model_page.get_model_config(),
                'api': self.api_page.get_api_config(),
                'completion': self.completion_page.get_completion_config(),
                'setup_completed': True,
                'setup_date': str(Path.ctime(Path.now()))
            }
            
            # Save to application config directory
            config_dir = Path.home() / ".ai-artworks"
            config_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            config_file = config_dir / "first_run_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save to INI format for compatibility
            ini_config = configparser.ConfigParser()
            
            # GPU settings
            ini_config['GPU'] = {
                'enable_cuda': str(config['gpu']['enable_cuda']),
                'enable_directml': str(config['gpu']['enable_directml']),
                'enable_cpu_fallback': str(config['gpu']['enable_cpu_fallback'])
            }
            
            # Model settings
            ini_config['Models'] = {
                'download_models': str(config['models']['download_models'])
            }
            for model, enabled in config['models']['models'].items():
                ini_config['Models'][model] = str(enabled)
            
            # API settings
            ini_config['API'] = config['api']['api_keys']
            
            # Voice settings
            ini_config['Voice'] = config['api']['voice_settings']
            
            ini_file = config_dir / "config.ini"
            with open(ini_file, 'w') as f:
                ini_config.write(f)
            
            print(f"Configuration saved to {config_dir}")
            
        except Exception as e:
            QMessageBox.warning(self, "Configuration Error", 
                              f"Failed to save configuration: {str(e)}")
    
    def launch_application(self):
        """Launch the main AI-ARTWORKS application"""
        try:
            # Find the main application executable
            app_paths = [
                Path(sys.executable).parent / "AI-ARTWORKS.exe",
                Path(__file__).parent.parent / "AI-ARTWORKS" / "main.py",
                Path("C:/Program Files/AI-ARTWORKS/bin/AI-ARTWORKS.exe")
            ]
            
            for app_path in app_paths:
                if app_path.exists():
                    if app_path.suffix == '.py':
                        subprocess.Popen([sys.executable, str(app_path)])
                    else:
                        subprocess.Popen([str(app_path)])
                    break
            else:
                QMessageBox.information(self, "Launch Application", 
                                      "Please launch AI-ARTWORKS from the Start Menu or desktop shortcut.")
                
        except Exception as e:
            QMessageBox.warning(self, "Launch Error", 
                              f"Failed to launch application: {str(e)}")

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI-ARTWORKS First Run Setup")
    app.setOrganizationName("AI-ARTWORKS")
    
    # Set application icon
    app.setWindowIcon(QIcon(":/icons/app_icon.png"))
    
    # Create and show wizard
    wizard = FirstRunWizard()
    wizard.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())