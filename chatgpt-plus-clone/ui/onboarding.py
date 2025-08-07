"""
Onboarding Dialog - First-time User Experience
Provides guided setup and tips for new users
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                 QPushButton, QTextEdit, QCheckBox, QGroupBox,
                                 QProgressBar, QTabWidget, QWidget, QScrollArea)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QFont, QPixmap, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

class OnboardingDialog(QDialog):
    """Comprehensive onboarding dialog for new users"""
    
    setup_completed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_file = "config.json"
        self.setup_steps = []
        self.current_step = 0
        
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        """Setup onboarding dialog UI"""
        self.setWindowTitle("Welcome to ChatGPT+ Clone")
        self.setModal(True)
        self.resize(700, 500)
        
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("🚀 Welcome to ChatGPT+ Clone")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        layout.addLayout(header_layout)
        
        # Subtitle
        subtitle_label = QLabel("Your personal AI assistant with voice, plugins, and more!")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setStyleSheet("color: #666; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        
        # Welcome tab
        self.tab_widget.addTab(self.create_welcome_tab(), "Welcome")
        
        # Features tab
        self.tab_widget.addTab(self.create_features_tab(), "Features")
        
        # Setup tab
        self.tab_widget.addTab(self.create_setup_tab(), "Setup")
        
        # Tips tab
        self.tab_widget.addTab(self.create_tips_tab(), "Tips")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(25)  # Start at 25% (welcome tab)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.previous_step)
        self.back_btn.setEnabled(False)
        button_layout.addWidget(self.back_btn)
        
        button_layout.addStretch()
        
        self.skip_btn = QPushButton("Skip Setup")
        self.skip_btn.clicked.connect(self.skip_setup)
        button_layout.addWidget(self.skip_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)
        self.next_btn.setDefault(True)
        button_layout.addWidget(self.next_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect tab changes to progress updates
        self.tab_widget.currentChanged.connect(self.update_progress)
    
    def create_welcome_tab(self) -> QWidget:
        """Create welcome tab content"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Welcome message
        welcome_text = """
        <h2>Welcome to ChatGPT+ Clone! 🤖</h2>
        
        <p>You're about to experience a powerful AI assistant that runs locally on your computer. 
        This application combines the best features of ChatGPT with additional capabilities:</p>
        
        <ul>
        <li><strong>🎤 Voice Interaction:</strong> Talk to your AI with natural speech</li>
        <li><strong>🧩 Plugin System:</strong> Extend functionality with custom plugins</li>
        <li><strong>🎨 AR Overlay:</strong> Futuristic visual interface</li>
        <li><strong>💻 Code Execution:</strong> Run Python code directly</li>
        <li><strong>🌐 Web Integration:</strong> Browse and search the web</li>
        <li><strong>🎨 Image Generation:</strong> Create images with AI</li>
        </ul>
        
        <p>This setup will help you configure the application to your preferences and get you started quickly.</p>
        """
        
        welcome_label = QLabel(welcome_text)
        welcome_label.setWordWrap(True)
        welcome_label.setOpenExternalLinks(True)
        layout.addWidget(welcome_label)
        
        # Quick start checkbox
        self.quick_start_cb = QCheckBox("Enable quick start mode (recommended for first-time users)")
        self.quick_start_cb.setChecked(True)
        layout.addWidget(self.quick_start_cb)
        
        widget.setLayout(layout)
        return widget
    
    def create_features_tab(self) -> QWidget:
        """Create features tab content"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Features overview
        features_text = """
        <h2>Key Features Overview</h2>
        
        <h3>🎤 Voice Assistant</h3>
        <p>Press <strong>Ctrl+Shift+V</strong> to activate voice input. The AI will listen to your command, 
        process it, and respond both in text and speech.</p>
        
        <h3>🧩 Plugin System</h3>
        <p>Extend the application with custom plugins. Plugins can add new commands, 
        integrate with external services, or provide specialized functionality.</p>
        
        <h3>🎨 AR Overlay</h3>
        <p>Experience a futuristic interface with holographic-style overlays that 
        provide visual feedback and enhance the user experience.</p>
        
        <h3>💻 Code Interpreter</h3>
        <p>Execute Python code directly within the application. Perfect for data analysis, 
        automation, or testing code snippets.</p>
        
        <h3>🌐 Web Browser</h3>
        <p>Search the web, browse websites, and gather information in real-time 
        through the integrated web browser functionality.</p>
        
        <h3>🎨 Image Generation</h3>
        <p>Create images using AI models. Generate artwork, diagrams, or visual content 
        based on your descriptions.</p>
        """
        
        features_label = QLabel(features_text)
        features_label.setWordWrap(True)
        layout.addWidget(features_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_setup_tab(self) -> QWidget:
        """Create setup tab content"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Setup options
        setup_group = QGroupBox("Initial Configuration")
        setup_layout = QVBoxLayout()
        
        # Voice hotkey
        self.voice_hotkey_cb = QCheckBox("Enable voice hotkey (Ctrl+Shift+V)")
        self.voice_hotkey_cb.setChecked(True)
        setup_layout.addWidget(self.voice_hotkey_cb)
        
        # AR overlay
        self.ar_overlay_cb = QCheckBox("Enable AR overlay (futuristic interface)")
        self.ar_overlay_cb.setChecked(True)
        setup_layout.addWidget(self.ar_overlay_cb)
        
        # Plugins
        self.plugins_cb = QCheckBox("Enable plugin system")
        self.plugins_cb.setChecked(True)
        setup_layout.addWidget(self.plugins_cb)
        
        # Plugin test mode
        self.plugin_test_cb = QCheckBox("Enable plugin test mode (for developers)")
        self.plugin_test_cb.setChecked(False)
        setup_layout.addWidget(self.plugin_test_cb)
        
        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QVBoxLayout()
        
        self.gpu_acceleration_cb = QCheckBox("Enable GPU acceleration (if available)")
        self.gpu_acceleration_cb.setChecked(True)
        perf_layout.addWidget(self.gpu_acceleration_cb)
        
        self.auto_optimize_cb = QCheckBox("Auto-optimize for your system")
        self.auto_optimize_cb.setChecked(True)
        perf_layout.addWidget(self.auto_optimize_cb)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # System check
        system_group = QGroupBox("System Check")
        system_layout = QVBoxLayout()
        
        self.system_status_label = QLabel("Checking system compatibility...")
        system_layout.addWidget(self.system_status_label)
        
        # Check system button
        self.check_system_btn = QPushButton("Check System Compatibility")
        self.check_system_btn.clicked.connect(self.check_system_compatibility)
        system_layout.addWidget(self.check_system_btn)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        widget.setLayout(layout)
        return widget
    
    def create_tips_tab(self) -> QWidget:
        """Create tips tab content"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        tips_text = """
        <h2>Getting Started Tips</h2>
        
        <h3>🎤 Voice Commands</h3>
        <ul>
        <li>Press <strong>Ctrl+Shift+V</strong> to start voice input</li>
        <li>Speak clearly and at a normal pace</li>
        <li>Try commands like "What's the weather?" or "Write a Python function"</li>
        <li>Wait for the confirmation beep before speaking</li>
        </ul>
        
        <h3>💬 Text Chat</h3>
        <ul>
        <li>Type your questions or requests in the chat interface</li>
        <li>Use natural language - the AI understands context</li>
        <li>Ask for code, explanations, or creative content</li>
        </ul>
        
        <h3>🧩 Using Plugins</h3>
        <ul>
        <li>Plugins are automatically loaded from the plugins/ directory</li>
        <li>Some plugins may add new voice commands</li>
        <li>Check the plugin test dialog to see available plugins</li>
        </ul>
        
        <h3>⚙️ Settings</h3>
        <ul>
        <li>Access settings from the menu or press <strong>Ctrl+,</strong></li>
        <li>Customize voice hotkeys, enable/disable features</li>
        <li>Adjust performance settings based on your system</li>
        </ul>
        
        <h3>🔧 Troubleshooting</h3>
        <ul>
        <li>If voice doesn't work, check your microphone settings</li>
        <li>For performance issues, try disabling GPU acceleration</li>
        <li>Check the logs folder for detailed error information</li>
        <li>Restart the application if plugins aren't loading</li>
        </ul>
        """
        
        tips_label = QLabel(tips_text)
        tips_label.setWordWrap(True)
        layout.addWidget(tips_label)
        
        # Documentation links
        docs_group = QGroupBox("Documentation")
        docs_layout = QVBoxLayout()
        
        user_guide_btn = QPushButton("📖 User Guide")
        user_guide_btn.clicked.connect(self.open_user_guide)
        docs_layout.addWidget(user_guide_btn)
        
        dev_guide_btn = QPushButton("🔧 Developer Guide")
        dev_guide_btn.clicked.connect(self.open_dev_guide)
        docs_layout.addWidget(dev_guide_btn)
        
        docs_group.setLayout(docs_layout)
        layout.addWidget(docs_group)
        
        widget.setLayout(layout)
        return widget
    
    def load_config(self):
        """Load current configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Update checkboxes based on current config
                self.voice_hotkey_cb.setChecked(config.get('voice_hotkey_enabled', True))
                self.ar_overlay_cb.setChecked(config.get('ar_overlay_enabled', True))
                self.plugins_cb.setChecked(config.get('plugins_enabled', True))
                self.plugin_test_cb.setChecked(config.get('plugin_test_mode', False))
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration from dialog"""
        try:
            config = {
                'voice_hotkey_enabled': self.voice_hotkey_cb.isChecked(),
                'ar_overlay_enabled': self.ar_overlay_cb.isChecked(),
                'plugins_enabled': self.plugins_cb.isChecked(),
                'plugin_test_mode': self.plugin_test_cb.isChecked(),
                'gpu_acceleration': self.gpu_acceleration_cb.isChecked(),
                'auto_optimize': self.auto_optimize_cb.isChecked(),
                'quick_start_mode': self.quick_start_cb.isChecked(),
                'onboarding_completed': True
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            print("Configuration saved successfully")
            
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def check_system_compatibility(self):
        """Check system compatibility"""
        self.system_status_label.setText("Checking system...")
        
        # Simulate system check
        import platform
        import psutil
        
        try:
            # Check Python version
            python_version = platform.python_version()
            
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            
            # Check GPU availability
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
            except ImportError:
                gpu_available = False
                gpu_name = "PyTorch not available"
            
            # Generate status report
            status_parts = []
            status_parts.append(f"✅ Python {python_version}")
            status_parts.append(f"✅ {cpu_cores} CPU cores")
            status_parts.append(f"✅ {memory_gb:.1f}GB RAM")
            
            if gpu_available:
                status_parts.append(f"✅ GPU: {gpu_name}")
            else:
                status_parts.append("⚠️  GPU: Not available (will use CPU)")
            
            if memory_gb >= 8:
                status_parts.append("✅ Sufficient memory")
            else:
                status_parts.append("⚠️  Low memory (8GB+ recommended)")
            
            status_text = "\n".join(status_parts)
            self.system_status_label.setText(status_text)
            
        except Exception as e:
            self.system_status_label.setText(f"❌ System check failed: {e}")
    
    def update_progress(self):
        """Update progress bar based on current tab"""
        current_tab = self.tab_widget.currentIndex()
        progress = (current_tab + 1) * 25  # 25% per tab
        self.progress_bar.setValue(progress)
        
        # Update button states
        self.back_btn.setEnabled(current_tab > 0)
        if current_tab == self.tab_widget.count() - 1:
            self.next_btn.setText("Finish")
        else:
            self.next_btn.setText("Next")
    
    def next_step(self):
        """Move to next step"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == self.tab_widget.count() - 1:
            # Finish setup
            self.complete_setup()
        else:
            # Move to next tab
            self.tab_widget.setCurrentIndex(current_tab + 1)
    
    def previous_step(self):
        """Move to previous step"""
        current_tab = self.tab_widget.currentIndex()
        if current_tab > 0:
            self.tab_widget.setCurrentIndex(current_tab - 1)
    
    def skip_setup(self):
        """Skip the setup process"""
        self.save_config()
        self.setup_completed.emit({'skipped': True})
        self.accept()
    
    def complete_setup(self):
        """Complete the setup process"""
        self.save_config()
        
        # Emit setup completion signal
        setup_data = {
            'voice_hotkey_enabled': self.voice_hotkey_cb.isChecked(),
            'ar_overlay_enabled': self.ar_overlay_cb.isChecked(),
            'plugins_enabled': self.plugins_cb.isChecked(),
            'plugin_test_mode': self.plugin_test_cb.isChecked(),
            'gpu_acceleration': self.gpu_acceleration_cb.isChecked(),
            'auto_optimize': self.auto_optimize_cb.isChecked(),
            'quick_start_mode': self.quick_start_cb.isChecked(),
            'skipped': False
        }
        
        self.setup_completed.emit(setup_data)
        self.accept()
    
    def open_user_guide(self):
        """Open user guide"""
        try:
            import webbrowser
            # This would open the actual user guide
            webbrowser.open("https://github.com/your-repo/chatgpt-plus-clone/docs/USER_GUIDE.md")
        except Exception as e:
            print(f"Could not open user guide: {e}")
    
    def open_dev_guide(self):
        """Open developer guide"""
        try:
            import webbrowser
            # This would open the actual developer guide
            webbrowser.open("https://github.com/your-repo/chatgpt-plus-clone/docs/DEV_GUIDE.md")
        except Exception as e:
            print(f"Could not open developer guide: {e}")

def show_onboarding_if_needed(parent=None) -> Optional[OnboardingDialog]:
    """Show onboarding dialog if it hasn't been completed"""
    config_file = "config.json"
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check if onboarding has been completed
            if config.get('onboarding_completed', False):
                return None
        else:
            # No config file exists, show onboarding
            pass
            
    except Exception as e:
        print(f"Error checking onboarding status: {e}")
    
    # Show onboarding dialog
    dialog = OnboardingDialog(parent)
    return dialog

if __name__ == "__main__":
    # Test the onboarding dialog
    if PYQT_AVAILABLE:
        import sys
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        dialog = OnboardingDialog()
        dialog.show()
        sys.exit(app.exec())
    else:
        print("PyQt6 not available - cannot test onboarding dialog")