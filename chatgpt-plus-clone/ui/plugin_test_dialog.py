"""
Plugin Test Dialog - Plugin Testing Interface
Provides a UI for testing plugins with manual input and real-time responses
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                              QPushButton, QLabel, QComboBox, QGroupBox,
                              QSplitter, QListWidget, QListWidgetItem,
                              QTabWidget, QWidget, QFormLayout, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from typing import Dict, Any, List
import json
from datetime import datetime

class PluginTestDialog(QDialog):
    """Dialog for testing plugins with manual input"""
    
    def __init__(self, plugins: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.plugins = plugins
        self.setup_ui()
        self.update_plugin_list()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Plugin Test Mode - ChatGPT+ Clone")
        self.setGeometry(300, 300, 800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Plugin selection and input
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Output and results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("🧪 Run Test")
        self.test_button.clicked.connect(self.run_test)
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_output)
        self.clear_button.setStyleSheet("""
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
        
        self.close_button = QPushButton("❌ Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
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
        
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with plugin selection and input"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Plugin selection
        plugin_group = QGroupBox("Plugin Selection")
        plugin_layout = QVBoxLayout(plugin_group)
        
        self.plugin_list = QListWidget()
        self.plugin_list.setMaximumHeight(150)
        plugin_layout.addWidget(self.plugin_list)
        
        # Test type selection
        test_group = QGroupBox("Test Type")
        test_layout = QFormLayout(test_group)
        
        self.test_type = QComboBox()
        self.test_type.addItems([
            "Voice Command (on_voice_command)",
            "Message Received (on_message_received)",
            "Tool Executed (on_tool_executed)",
            "File Uploaded (on_file_uploaded)"
        ])
        test_layout.addRow("Hook Type:", self.test_type)
        
        # Input area
        input_group = QGroupBox("Test Input")
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter test command or message here...")
        self.input_text.setMaximumHeight(100)
        input_layout.addWidget(self.input_text)
        
        # Context input
        context_group = QGroupBox("Context (Optional)")
        context_layout = QFormLayout(context_group)
        
        self.context_key = QLineEdit("user_id")
        context_layout.addRow("Context Key:", self.context_key)
        
        self.context_value = QLineEdit("test_user")
        context_layout.addRow("Context Value:", self.context_value)
        
        # Add groups to layout
        layout.addWidget(plugin_group)
        layout.addWidget(test_group)
        layout.addWidget(input_group)
        layout.addWidget(context_group)
        layout.addStretch()
        
        return widget
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with output and results"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create tab widget for different output types
        self.output_tabs = QTabWidget()
        
        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.output_label = QLabel("Plugin Test Results")
        self.output_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        results_layout.addWidget(self.output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Test results will appear here...")
        results_layout.addWidget(self.output_text)
        
        self.output_tabs.addTab(results_tab, "📊 Results")
        
        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        log_label = QLabel("Test Log")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Test log will appear here...")
        log_layout.addWidget(self.log_text)
        
        self.output_tabs.addTab(log_tab, "📝 Log")
        
        # Plugin info tab
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        info_label = QLabel("Plugin Information")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addWidget(info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Plugin information will appear here...")
        info_layout.addWidget(self.info_text)
        
        self.output_tabs.addTab(info_tab, "ℹ️ Info")
        
        layout.addWidget(self.output_tabs)
        
        return widget
    
    def update_plugin_list(self):
        """Update the plugin list with available plugins"""
        self.plugin_list.clear()
        
        for plugin in self.plugins:
            item = QListWidgetItem(f"{plugin['name']} v{plugin['version']}")
            item.setData(Qt.ItemDataRole.UserRole, plugin)
            self.plugin_list.addItem(item)
        
        # Select first plugin by default
        if self.plugin_list.count() > 0:
            self.plugin_list.setCurrentRow(0)
    
    def run_test(self):
        """Run the plugin test"""
        # Get selected plugin
        current_item = self.plugin_list.currentItem()
        if not current_item:
            self.log_message("❌ No plugin selected")
            return
        
        plugin = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Get test input
        test_input = self.input_text.toPlainText().strip()
        if not test_input:
            self.log_message("❌ No test input provided")
            return
        
        # Get test type
        test_type = self.test_type.currentText()
        hook_name = test_type.split("(")[1].split(")")[0]
        
        # Get context
        context_key = self.context_key.text().strip()
        context_value = self.context_value.text().strip()
        context = {}
        if context_key and context_value:
            context[context_key] = context_value
        
        # Log test start
        self.log_message(f"🧪 Starting test for plugin: {plugin['name']}")
        self.log_message(f"📝 Test input: {test_input}")
        self.log_message(f"🔗 Hook type: {hook_name}")
        self.log_message(f"📋 Context: {context}")
        
        # Run test
        try:
            if hook_name in plugin['hooks']:
                hook_func = plugin['hooks'][hook_name]
                
                # Call the hook function
                if hook_name == "on_voice_command":
                    result = hook_func(test_input, context)
                elif hook_name == "on_message_received":
                    result = hook_func(test_input, context)
                elif hook_name == "on_tool_executed":
                    result = hook_func("test_tool", test_input, context)
                elif hook_name == "on_file_uploaded":
                    result = hook_func(test_input, context)
                else:
                    result = hook_func(test_input, context)
                
                # Display results
                if result:
                    self.output_text.append(f"✅ {plugin['name']}: {result}")
                    self.log_message(f"✅ Plugin responded: {result}")
                else:
                    self.output_text.append(f"⚠️ {plugin['name']}: No response")
                    self.log_message(f"⚠️ Plugin returned no response")
                
            else:
                self.output_text.append(f"❌ {plugin['name']}: Hook '{hook_name}' not found")
                self.log_message(f"❌ Hook '{hook_name}' not available in plugin")
                
        except Exception as e:
            error_msg = f"❌ {plugin['name']}: Error - {str(e)}"
            self.output_text.append(error_msg)
            self.log_message(f"❌ Test failed with error: {str(e)}")
        
        # Update plugin info
        self.update_plugin_info(plugin)
    
    def clear_output(self):
        """Clear all output areas"""
        self.output_text.clear()
        self.log_text.clear()
        self.info_text.clear()
        self.log_message("🗑️ Output cleared")
    
    def log_message(self, message: str):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def update_plugin_info(self, plugin: Dict[str, Any]):
        """Update plugin information display"""
        info = f"""Plugin Information:
        
Name: {plugin['name']}
Version: {plugin['version']}
Description: {plugin['description']}
Author: {plugin.get('author', 'Unknown')}

Available Hooks:
"""
        
        for hook_name in plugin['hooks'].keys():
            info += f"  - {hook_name}\n"
        
        if 'permissions' in plugin:
            info += f"\nPermissions:\n"
            for permission in plugin['permissions']:
                info += f"  - {permission}\n"
        
        if 'commands' in plugin:
            info += f"\nCommands:\n"
            for cmd, desc in plugin['commands'].items():
                info += f"  - {cmd}: {desc}\n"
        
        self.info_text.setText(info)
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Ctrl+Enter to run test
            self.run_test()
        else:
            super().keyPressEvent(event)