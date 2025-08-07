#!/usr/bin/env python3
"""
ChatGPT+ Clone - Main Application Entry Point
A comprehensive Windows desktop application with AI capabilities
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSplitter
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont

# Import our modules
from ui.chat_interface import ChatInterface
from ui.tools_panel import ToolsPanel
from ui.file_browser import FileBrowser
from ui.voice_panel import VoicePanel
from llm.agent_orchestrator import AgentOrchestrator
from memory.memory_manager import MemoryManager
from tools.code_executor import CodeExecutor
from tools.web_browser import WebBrowser
from tools.image_editor import ImageEditor
from tools.voice_agent import VoiceAgent
from vs_code_link.vs_code_integration import VSCodeIntegration
from config_manager import ConfigManager
from voice_hotkey import start_voice_listener, stop_voice_listener, update_voice_settings
from plugin_loader import PluginLoader
from ui.settings_dialog import SettingsDialog
from ui.plugin_test_dialog import PluginTestDialog

class ChatGPTPlusClone(QMainWindow):
    """Main application window for ChatGPT+ Clone"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT+ Clone - AI Assistant")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize config manager
        self.config = ConfigManager()
        
        # Initialize core components
        self.memory_manager = MemoryManager()
        self.agent_orchestrator = AgentOrchestrator(self.memory_manager)
        self.vs_code_integration = VSCodeIntegration()
        
        # Initialize plugin system
        self.plugin_loader = PluginLoader()
        self.plugins = []
        self.plugins_active = False
        self.plugin_loader.start_watching(self.reload_plugins)
        
        # Initialize AR overlay
        self.ar_overlay_instance = None
        self.ar_overlay_active = False
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # Initialize voice hotkey listener
        self.setup_voice_hotkey()
        
        # Connect signals
        self.connect_signals()
        
        # Load initial state
        self.load_initial_state()
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Tools and File Browser
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Tools Panel
        self.tools_panel = ToolsPanel(self.agent_orchestrator)
        left_layout.addWidget(self.tools_panel)
        
        # File Browser
        self.file_browser = FileBrowser()
        left_layout.addWidget(self.file_browser)
        
        splitter.addWidget(left_panel)
        
        # Center panel - Chat Interface
        self.chat_interface = ChatInterface(self.agent_orchestrator)
        splitter.addWidget(self.chat_interface)
        
        # Right panel - Voice and VS Code
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Voice Panel
        self.voice_panel = VoicePanel(self.agent_orchestrator)
        right_layout.addWidget(self.voice_panel)
        
        # VS Code Integration
        self.vs_code_widget = self.vs_code_integration.get_widget()
        right_layout.addWidget(self.vs_code_widget)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 800, 300])
    
    def setup_menu(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_chat_action = file_menu.addAction('New Chat')
        new_chat_action.triggered.connect(self.new_chat)
        
        save_chat_action = file_menu.addAction('Save Chat')
        save_chat_action.triggered.connect(self.save_chat)
        
        load_chat_action = file_menu.addAction('Load Chat')
        load_chat_action.triggered.connect(self.load_chat)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        code_interpreter_action = tools_menu.addAction('Code Interpreter')
        code_interpreter_action.triggered.connect(self.open_code_interpreter)
        
        image_editor_action = tools_menu.addAction('Image Editor')
        image_editor_action.triggered.connect(self.open_image_editor)
        
        web_browser_action = tools_menu.addAction('Web Browser')
        web_browser_action.triggered.connect(self.open_web_browser)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        settings_action = settings_menu.addAction('Settings')
        settings_action.triggered.connect(self.open_settings)
        
        plugin_test_action = settings_menu.addAction('Plugin Test Mode')
        plugin_test_action.triggered.connect(self.open_plugin_test)
        
        settings_menu.addSeparator()
        
        model_settings_action = settings_menu.addAction('Model Settings')
        model_settings_action.triggered.connect(self.open_model_settings)
        
        voice_settings_action = settings_menu.addAction('Voice Settings')
        voice_settings_action.triggered.connect(self.open_voice_settings)
    
    def setup_status_bar(self):
        """Setup status bar with system information"""
        self.statusBar().showMessage("Ready - ChatGPT+ Clone v1.0")
        
        # Add memory usage indicator
        self.memory_label = self.statusBar().addPermanentWidget(
            QWidget()
        )
    
    def setup_voice_hotkey(self):
        """Setup global voice hotkey listener"""
        # Check if voice hotkey is enabled in config
        voice_settings = self.config.get_voice_settings()
        voice_enabled = voice_settings.get('enabled', True)
        
        if voice_enabled:
            self.start_voice_listener()
        else:
            self.statusBar().showMessage("Voice hotkey disabled in settings")
    
    def start_voice_listener(self):
        """Start the voice hotkey listener"""
        try:
            self.voice_listener = start_voice_listener(callback=self.handle_voice_command)
            self.statusBar().showMessage("Voice hotkey enabled: Ctrl+Shift+V")
            print("[Main] Voice hotkey listener started")
        except Exception as e:
            self.statusBar().showMessage(f"Voice hotkey error: {e}")
            print(f"[Main] Voice hotkey error: {e}")
    
    def stop_voice_listener(self):
        """Stop the voice hotkey listener"""
        try:
            stop_voice_listener()
            self.voice_listener = None
            self.statusBar().showMessage("Voice hotkey disabled")
            print("[Main] Voice hotkey listener stopped")
        except Exception as e:
            print(f"[Main] Error stopping voice listener: {e}")
    
    def handle_voice_command(self, text: str):
        """Handle voice command from hotkey"""
        print(f"[Main] Voice command received: {text}")
        self.chat_interface.add_user_message(text)
        self.handle_user_message(text)
    
    def connect_signals(self):
        """Connect all signal handlers"""
        # Chat interface signals
        self.chat_interface.message_sent.connect(self.handle_user_message)
        self.chat_interface.file_dropped.connect(self.handle_file_upload)
        
        # Tools panel signals
        self.tools_panel.tool_activated.connect(self.activate_tool)
        
        # File browser signals
        self.file_browser.file_selected.connect(self.handle_file_selection)
        
        # Voice panel signals
        self.voice_panel.voice_input.connect(self.handle_voice_input)
    
    def load_initial_state(self):
        """Load initial application state"""
        try:
            # Load recent chats
            self.load_recent_chats()
            
            # Initialize workspace
            self.initialize_workspace()
            
            # Load user preferences
            self.load_preferences()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading initial state: {e}")
    
    def handle_user_message(self, message: str):
        """Handle user message from chat interface"""
        try:
            # Process message through agent orchestrator
            response = self.agent_orchestrator.process_message(message)
            
            # Update chat interface
            self.chat_interface.add_assistant_message(response)
            
            # Update status
            self.statusBar().showMessage("Message processed successfully")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error processing message: {e}")
    
    def handle_file_upload(self, file_path: str):
        """Handle file upload from drag and drop"""
        try:
            # Process file through agent
            result = self.agent_orchestrator.process_file(file_path)
            
            # Update chat interface with file processing result
            self.chat_interface.add_system_message(f"File processed: {result}")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error processing file: {e}")
    
    def activate_tool(self, tool_name: str):
        """Activate a specific tool"""
        try:
            if tool_name == "code_interpreter":
                self.open_code_interpreter()
            elif tool_name == "image_editor":
                self.open_image_editor()
            elif tool_name == "web_browser":
                self.open_web_browser()
            elif tool_name == "voice":
                self.activate_voice()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error activating tool {tool_name}: {e}")
    
    def activate_voice(self):
        """Activate voice input mode"""
        try:
            self.voice_panel.start_voice_input()
            self.statusBar().showMessage("Voice input activated")
        except Exception as e:
            self.statusBar().showMessage(f"Error activating voice: {e}")
    
    def open_code_interpreter(self):
        """Open code interpreter tool"""
        try:
            self.agent_orchestrator.activate_tool("code_interpreter")
            self.statusBar().showMessage("Code interpreter activated")
        except Exception as e:
            self.statusBar().showMessage(f"Error opening code interpreter: {e}")
    
    def open_image_editor(self):
        """Open image editor tool"""
        try:
            self.agent_orchestrator.activate_tool("image_editor")
            self.statusBar().showMessage("Image editor activated")
        except Exception as e:
            self.statusBar().showMessage(f"Error opening image editor: {e}")
    
    def open_web_browser(self):
        """Open web browser tool"""
        try:
            self.agent_orchestrator.activate_tool("web_browser")
            self.statusBar().showMessage("Web browser activated")
        except Exception as e:
            self.statusBar().showMessage(f"Error opening web browser: {e}")
    
    def new_chat(self):
        """Start a new chat session"""
        self.chat_interface.clear_chat()
        self.memory_manager.clear_session()
        self.statusBar().showMessage("New chat started")
    
    def save_chat(self):
        """Save current chat session"""
        try:
            chat_data = self.chat_interface.get_chat_history()
            self.memory_manager.save_chat_session(chat_data)
            self.statusBar().showMessage("Chat saved successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving chat: {e}")
    
    def load_chat(self):
        """Load a saved chat session"""
        try:
            # This would open a file dialog to select a chat file
            self.statusBar().showMessage("Load chat functionality to be implemented")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading chat: {e}")
    
    def open_model_settings(self):
        """Open model settings dialog"""
        self.statusBar().showMessage("Model settings to be implemented")
    
    def open_settings(self):
        """Open settings dialog"""
        try:
            dialog = SettingsDialog(parent=self)
            dialog.voice_hotkey_toggled.connect(self.on_voice_hotkey_toggled)
            dialog.toggle_changed.connect(self.on_toggle_changed)
            dialog.settings_changed.connect(self.on_settings_changed)
            dialog.exec()
        except Exception as e:
            self.statusBar().showMessage(f"Error opening settings: {e}")
    
    def open_plugin_test(self):
        """Open plugin test dialog"""
        try:
            dialog = PluginTestDialog(self.plugins, parent=self)
            dialog.exec()
        except Exception as e:
            self.statusBar().showMessage(f"Error opening plugin test: {e}")
    
    def on_voice_hotkey_toggled(self, enabled: bool):
        """Handle voice hotkey toggle from settings"""
        print(f"[Main] Voice hotkey toggled: {enabled}")
        
        # Update config
        voice_settings = self.config.get_voice_settings()
        voice_settings['enabled'] = enabled
        self.config.set_voice_settings(voice_settings)
        
        # Start or stop voice listener
        if enabled:
            self.start_voice_listener()
        else:
            self.stop_voice_listener()
    
    def on_toggle_changed(self, toggle_name: str, enabled: bool):
        """Handle any toggle change from settings"""
        print(f"[Main] Toggle changed: {toggle_name} -> {enabled}")
        
        # Update app settings
        app_settings = self.config.get_app_settings()
        app_settings[toggle_name] = enabled
        self.config.set_app_settings(app_settings)
        
        # Apply the toggle change
        if toggle_name == "voice_hotkey_enabled":
            if enabled:
                self.start_voice_listener()
            else:
                self.stop_voice_listener()
                
        elif toggle_name == "enable_plugins":
            if enabled:
                self.enable_plugins()
            else:
                self.disable_plugins()
                
        elif toggle_name == "enable_ar_overlay":
            if enabled:
                self.enable_ar_overlay()
            else:
                self.disable_ar_overlay()
                
        elif toggle_name == "enable_code_interpreter":
            # Toggle code interpreter tool
            if enabled:
                self.agent_orchestrator.enable_tool("code_interpreter")
            else:
                self.agent_orchestrator.disable_tool("code_interpreter")
                
        elif toggle_name == "enable_image_editor":
            # Toggle image editor tool
            if enabled:
                self.agent_orchestrator.enable_tool("image_editor")
            else:
                self.agent_orchestrator.disable_tool("image_editor")
                
        elif toggle_name == "enable_web_browser":
            # Toggle web browser tool
            if enabled:
                self.agent_orchestrator.enable_tool("web_browser")
            else:
                self.agent_orchestrator.disable_tool("web_browser")
                
        elif toggle_name == "enable_memory_system":
            # Toggle memory system
            if enabled:
                self.memory_manager.enable()
            else:
                self.memory_manager.disable()
                
        elif toggle_name == "enable_vs_code_integration":
            # Toggle VS Code integration
            if enabled:
                self.vs_code_integration.enable()
            else:
                self.vs_code_integration.disable()
                
        else:
            print(f"[Main] No handler for toggle: {toggle_name}")
    
    def on_settings_changed(self, settings: dict):
        """Handle settings changes"""
        print(f"[Main] Settings changed: {settings}")
        self.statusBar().showMessage("Settings updated")
    
    def enable_plugins(self):
        """Enable and start all plugins"""
        if not self.plugins_active:
            print("[Main] Loading plugins...")
            try:
                self.plugins = self.plugin_loader.load_plugins()
                
                # Start each plugin if it has a start method
                for plugin in self.plugins:
                    if hasattr(plugin, 'module') and hasattr(plugin['module'], 'on_start'):
                        try:
                            plugin['module'].on_start()
                            print(f"[Main] Started plugin: {plugin.get('name', 'Unknown')}")
                        except Exception as e:
                            print(f"[Main] Error starting plugin {plugin.get('name', 'Unknown')}: {e}")
                
                self.plugins_active = True
                self.statusBar().showMessage(f"Plugins enabled: {len(self.plugins)} loaded")
                print(f"[Main] {len(self.plugins)} plugins loaded and started")
                
            except Exception as e:
                self.statusBar().showMessage(f"Failed to load plugins: {e}")
                print(f"[Main] Failed to load plugins: {e}")
    
    def disable_plugins(self):
        """Disable and stop all plugins"""
        if self.plugins_active:
            print("[Main] Stopping plugins...")
            try:
                # Stop each plugin if it has a stop method
                for plugin in self.plugins:
                    if hasattr(plugin, 'module') and hasattr(plugin['module'], 'on_stop'):
                        try:
                            plugin['module'].on_stop()
                            print(f"[Main] Stopped plugin: {plugin.get('name', 'Unknown')}")
                        except Exception as e:
                            print(f"[Main] Error stopping plugin {plugin.get('name', 'Unknown')}: {e}")
                
                self.plugins = []
                self.plugins_active = False
                self.statusBar().showMessage("Plugins disabled")
                print("[Main] Plugins stopped and unloaded")
                
            except Exception as e:
                self.statusBar().showMessage(f"Failed to stop plugins: {e}")
                print(f"[Main] Failed to stop plugins: {e}")
    
    def enable_ar_overlay(self):
        """Enable AR overlay"""
        if not self.ar_overlay_active:
            print("[Main] Starting AR overlay...")
            try:
                from overlay_ar import AROverlay
                self.ar_overlay_instance = AROverlay()
                self.ar_overlay_instance.show()
                self.ar_overlay_active = True
                self.statusBar().showMessage("AR overlay enabled")
                print("[Main] AR overlay started")
                
            except ImportError:
                self.statusBar().showMessage("AR overlay module not available")
                print("[Main] AR overlay module not available")
            except Exception as e:
                self.statusBar().showMessage(f"Failed to start AR overlay: {e}")
                print(f"[Main] Failed to start AR overlay: {e}")
    
    def disable_ar_overlay(self):
        """Disable AR overlay"""
        if self.ar_overlay_active:
            print("[Main] Stopping AR overlay...")
            try:
                if self.ar_overlay_instance:
                    self.ar_overlay_instance.close()
                    self.ar_overlay_instance = None
                self.ar_overlay_active = False
                self.statusBar().showMessage("AR overlay disabled")
                print("[Main] AR overlay stopped")
                
            except Exception as e:
                self.statusBar().showMessage(f"Failed to stop AR overlay: {e}")
                print(f"[Main] Failed to stop AR overlay: {e}")
    
    def reload_plugins(self):
        """Reload plugins when changes detected"""
        try:
            if self.plugins_active:
                # Reload plugins if they're currently enabled
                self.disable_plugins()
                self.enable_plugins()
            else:
                # Just reload the plugin list without starting them
                self.plugins = self.plugin_loader.load_plugins()
            
            self.statusBar().showMessage(f"Plugins reloaded: {len(self.plugins)} loaded")
            print(f"[Main] Plugins reloaded: {len(self.plugins)} plugins")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error reloading plugins: {e}")
            print(f"[Main] Error reloading plugins: {e}")
    
    def open_voice_settings(self):
        """Open voice settings dialog"""
        self.statusBar().showMessage("Voice settings to be implemented")
    
    def load_recent_chats(self):
        """Load recent chat sessions"""
        pass  # To be implemented
    
    def initialize_workspace(self):
        """Initialize the workspace directory"""
        workspace_path = Path("workspace")
        workspace_path.mkdir(exist_ok=True)
    
    def load_preferences(self):
        """Load user preferences"""
        pass  # To be implemented
    
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop voice listener
            if hasattr(self, 'voice_listener') and self.voice_listener:
                self.stop_voice_listener()
            
            # Stop plugin watcher
            if hasattr(self, 'plugin_loader'):
                self.plugin_loader.stop_watching()
            
            # Save any pending data
            self.save_chat()
            
            print("[Main] Application closing - cleanup completed")
            event.accept()
            
        except Exception as e:
            print(f"[Main] Error during cleanup: {e}")
            event.accept()
    
    def handle_file_selection(self, file_path: str):
        """Handle file selection from file browser"""
        self.chat_interface.add_system_message(f"File selected: {file_path}")
    
    def handle_voice_input(self, text: str):
        """Handle voice input from voice panel"""
        self.chat_interface.add_user_message(text)
        self.handle_user_message(text)

def main():
    """Main application entry point"""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("ChatGPT+ Clone")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ChatGPTPlusClone()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()