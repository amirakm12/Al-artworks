#!/usr/bin/env python3
"""
Test Voice Hotkey System
Simple test to verify voice hotkey functionality
"""

import sys
import time
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
from PyQt6.QtCore import Qt

# Import our modules
from config_manager import ConfigManager
from voice_hotkey import start_voice_listener, stop_voice_listener
from ui.settings_dialog import SettingsDialog

class VoiceHotkeyTest(QMainWindow):
    """Test window for voice hotkey functionality"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Hotkey Test")
        self.setGeometry(100, 100, 600, 400)
        
        # Initialize config
        self.config = ConfigManager()
        
        # Setup UI
        self.setup_ui()
        
        # Start voice listener
        self.start_voice_listener()
    
    def setup_ui(self):
        """Setup the test interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Voice Hotkey Test")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Press Ctrl+Shift+V to test voice recording.\n"
            "Use the Settings button to toggle voice hotkey on/off."
        )
        instructions.setStyleSheet("margin: 10px; color: #666;")
        layout.addWidget(instructions)
        
        # Settings button
        self.settings_btn = QPushButton("Open Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_btn)
        
        # Status display
        self.status_label = QLabel("Status: Voice hotkey enabled")
        self.status_label.setStyleSheet("margin: 10px; padding: 5px; background-color: #e8f5e8; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Voice commands will appear here...")
        layout.addWidget(self.log_display)
        
        # Test voice button
        self.test_voice_btn = QPushButton("Test Voice Input")
        self.test_voice_btn.clicked.connect(self.test_voice_input)
        layout.addWidget(self.test_voice_btn)
    
    def start_voice_listener(self):
        """Start the voice hotkey listener"""
        try:
            self.voice_listener = start_voice_listener(callback=self.handle_voice_command)
            self.update_status("Voice hotkey enabled")
            self.log_message("Voice hotkey listener started")
        except Exception as e:
            self.update_status(f"Voice hotkey error: {e}")
            self.log_message(f"Error starting voice listener: {e}")
    
    def stop_voice_listener(self):
        """Stop the voice hotkey listener"""
        try:
            stop_voice_listener()
            self.voice_listener = None
            self.update_status("Voice hotkey disabled")
            self.log_message("Voice hotkey listener stopped")
        except Exception as e:
            self.log_message(f"Error stopping voice listener: {e}")
    
    def handle_voice_command(self, text: str):
        """Handle voice command from hotkey"""
        self.log_message(f"🎤 Voice command: {text}")
        self.update_status(f"Last command: {text[:50]}...")
    
    def open_settings(self):
        """Open settings dialog"""
        try:
            dialog = SettingsDialog(parent=self)
            dialog.voice_hotkey_toggled.connect(self.on_voice_hotkey_toggled)
            dialog.exec()
        except Exception as e:
            self.log_message(f"Error opening settings: {e}")
    
    def on_voice_hotkey_toggled(self, enabled: bool):
        """Handle voice hotkey toggle from settings"""
        self.log_message(f"Voice hotkey toggled: {enabled}")
        
        # Update config
        voice_settings = self.config.get_voice_settings()
        voice_settings['enabled'] = enabled
        self.config.set_voice_settings(voice_settings)
        
        # Start or stop voice listener
        if enabled:
            self.start_voice_listener()
        else:
            self.stop_voice_listener()
    
    def test_voice_input(self):
        """Test voice input manually"""
        self.log_message("Testing voice input...")
        # This would trigger the voice recording manually
        if hasattr(self, 'voice_listener') and self.voice_listener:
            self.log_message("Voice listener is active - press Ctrl+Shift+V to test")
        else:
            self.log_message("Voice listener is not active")
    
    def update_status(self, message: str):
        """Update status display"""
        self.status_label.setText(f"Status: {message}")
    
    def log_message(self, message: str):
        """Add message to log display"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle window close"""
        try:
            if hasattr(self, 'voice_listener') and self.voice_listener:
                self.stop_voice_listener()
            self.log_message("Test window closing")
            event.accept()
        except Exception as e:
            self.log_message(f"Error during cleanup: {e}")
            event.accept()

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Hotkey Test")
    
    # Create and show test window
    window = VoiceHotkeyTest()
    window.show()
    
    print("🧪 Voice Hotkey Test Started")
    print("Press Ctrl+Shift+V to test voice recording")
    print("Use the Settings button to toggle voice hotkey")
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()