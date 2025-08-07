"""
Plugin Test Dialog - Plugin Testing and Debugging Interface
Provides a comprehensive interface for testing plugins and viewing their responses
"""

import logging
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                              QTextEdit, QPushButton, QLabel, QComboBox,
                              QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
                              QSplitter, QWidget, QMessageBox, QProgressBar,
                              QCheckBox, QLineEdit, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor

class PluginTestDialog(QDialog):
    """Comprehensive plugin testing dialog"""
    
    # Signals
    plugin_tested = pyqtSignal(str, str)  # plugin_name, result
    plugin_error = pyqtSignal(str, str)   # plugin_name, error
    
    def __init__(self, main_app=None, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.logger = logging.getLogger(__name__)
        
        # Plugin data
        self.plugins = []
        self.current_plugin = None
        self.test_results = {}
        
        # Setup UI
        self.setup_ui()
        self.load_plugins()
        self.connect_signals()
        
        self.logger.info("Plugin test dialog initialized")
    
    def setup_ui(self):
        """Setup the plugin test dialog UI"""
        self.setWindowTitle("Plugin Test Mode - ChatGPT+ Clone")
        self.setGeometry(300, 300, 1000, 700)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Plugin selection and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Test results and logs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        # Status bar
        self.status_label = QLabel("Ready to test plugins")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-weight: bold;
                padding: 5px;
                background-color: #1E1E1E;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def create_left_panel(self):
        """Create the left control panel"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        
        # Plugin selection group
        plugin_group = QGroupBox("Plugin Selection")
        plugin_layout = QFormLayout(plugin_group)
        
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItem("Select a plugin...")
        plugin_layout.addRow("Plugin:", self.plugin_combo)
        
        self.plugin_info_label = QLabel("No plugin selected")
        self.plugin_info_label.setWordWrap(True)
        self.plugin_info_label.setStyleSheet("""
            QLabel {
                background-color: #2D2D30;
                padding: 10px;
                border-radius: 5px;
                color: #FFFFFF;
            }
        """)
        plugin_layout.addRow("Info:", self.plugin_info_label)
        
        layout.addWidget(plugin_group)
        
        # Test configuration group
        config_group = QGroupBox("Test Configuration")
        config_layout = QFormLayout(config_group)
        
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems([
            "Voice Command", "Message", "Tool Execution", "Start/Stop", "Config Test"
        ])
        config_layout.addRow("Test Type:", self.test_type_combo)
        
        self.test_input = QTextEdit()
        self.test_input.setPlaceholderText("Enter test input here...")
        self.test_input.setMaximumHeight(100)
        config_layout.addRow("Test Input:", self.test_input)
        
        self.test_timeout = QSpinBox()
        self.test_timeout.setRange(1, 60)
        self.test_timeout.setValue(10)
        self.test_timeout.setSuffix(" seconds")
        config_layout.addRow("Timeout:", self.test_timeout)
        
        self.auto_test = QCheckBox("Auto-test on plugin change")
        config_layout.addRow("Auto Test:", self.auto_test)
        
        layout.addWidget(config_group)
        
        # Test controls group
        controls_group = QGroupBox("Test Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.run_test_btn = QPushButton("Run Test")
        self.run_test_btn.setStyleSheet("""
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
        controls_layout.addWidget(self.run_test_btn)
        
        self.run_all_btn = QPushButton("Test All Plugins")
        self.run_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E7E34;
            }
        """)
        controls_layout.addWidget(self.run_all_btn)
        
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #545B62;
            }
        """)
        controls_layout.addWidget(self.clear_results_btn)
        
        layout.addWidget(controls_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return left_widget
    
    def create_right_panel(self):
        """Create the right results panel"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Create tab widget for different views
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Test results tab
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        self.results_tabs.addTab(self.results_text, "Test Results")
        
        # Plugin status tab
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(4)
        self.status_table.setHorizontalHeaderLabels([
            "Plugin", "Status", "Last Test", "Result"
        ])
        self.status_table.setStyleSheet("""
            QTableWidget {
                background-color: #2D2D30;
                color: #FFFFFF;
                gridline-color: #3F3F46;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3F3F46;
            }
        """)
        self.results_tabs.addTab(self.status_table, "Plugin Status")
        
        # Debug logs tab
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #00FF00;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.results_tabs.addTab(self.debug_text, "Debug Logs")
        
        return right_widget
    
    def connect_signals(self):
        """Connect all signal handlers"""
        self.plugin_combo.currentTextChanged.connect(self.on_plugin_selected)
        self.run_test_btn.clicked.connect(self.run_single_test)
        self.run_all_btn.clicked.connect(self.run_all_tests)
        self.clear_results_btn.clicked.connect(self.clear_results)
        self.test_type_combo.currentTextChanged.connect(self.on_test_type_changed)
    
    def load_plugins(self):
        """Load available plugins"""
        try:
            if self.main_app and hasattr(self.main_app, 'plugins'):
                self.plugins = self.main_app.plugins
            else:
                # Fallback: try to load plugins directly
                from plugin_loader import load_plugins
                self.plugins = load_plugins()
            
            # Populate plugin combo
            self.plugin_combo.clear()
            self.plugin_combo.addItem("Select a plugin...")
            
            for plugin in self.plugins:
                if hasattr(plugin, 'name'):
                    self.plugin_combo.addItem(plugin.name)
                else:
                    self.plugin_combo.addItem(f"Plugin {len(self.plugins)}")
            
            self.logger.info(f"Loaded {len(self.plugins)} plugins for testing")
            self.update_status(f"Loaded {len(self.plugins)} plugins")
            
        except Exception as e:
            self.logger.error(f"Error loading plugins: {e}")
            self.update_status(f"Error loading plugins: {e}")
    
    def on_plugin_selected(self, plugin_name: str):
        """Handle plugin selection"""
        if plugin_name == "Select a plugin...":
            self.current_plugin = None
            self.plugin_info_label.setText("No plugin selected")
            return
        
        # Find the selected plugin
        for plugin in self.plugins:
            if hasattr(plugin, 'name') and plugin.name == plugin_name:
                self.current_plugin = plugin
                self.update_plugin_info(plugin)
                break
        
        # Auto-test if enabled
        if self.auto_test.isChecked() and self.current_plugin:
            QTimer.singleShot(500, self.run_single_test)
    
    def update_plugin_info(self, plugin):
        """Update plugin information display"""
        try:
            info = f"Plugin: {getattr(plugin, 'name', 'Unknown')}\n"
            info += f"Type: {type(plugin).__name__}\n"
            
            if hasattr(plugin, 'module'):
                info += f"Module: {plugin.module}\n"
            
            if hasattr(plugin, 'code'):
                info += f"Code length: {len(plugin.code)} characters\n"
            
            # Check available methods
            methods = []
            if hasattr(plugin, 'start'):
                methods.append("start()")
            if hasattr(plugin, 'stop'):
                methods.append("stop()")
            if hasattr(plugin, 'call_function'):
                methods.append("call_function()")
            
            info += f"Methods: {', '.join(methods)}"
            
            self.plugin_info_label.setText(info)
            
        except Exception as e:
            self.plugin_info_label.setText(f"Error getting plugin info: {e}")
    
    def on_test_type_changed(self, test_type: str):
        """Handle test type change"""
        # Update placeholder text based on test type
        if test_type == "Voice Command":
            self.test_input.setPlaceholderText("Enter voice command text...")
        elif test_type == "Message":
            self.test_input.setPlaceholderText("Enter message text...")
        elif test_type == "Tool Execution":
            self.test_input.setPlaceholderText("Enter tool parameters (JSON)...")
        elif test_type == "Config Test":
            self.test_input.setPlaceholderText("Enter configuration (JSON)...")
        else:
            self.test_input.setPlaceholderText("Enter test input...")
    
    def run_single_test(self):
        """Run test on selected plugin"""
        if not self.current_plugin:
            QMessageBox.warning(self, "No Plugin", "Please select a plugin to test")
            return
        
        try:
            self.update_status(f"Testing plugin: {getattr(self.current_plugin, 'name', 'Unknown')}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Get test parameters
            test_type = self.test_type_combo.currentText()
            test_input = self.test_input.toPlainText()
            timeout = self.test_timeout.value()
            
            # Run test in separate thread to avoid blocking UI
            QTimer.singleShot(100, lambda: self._execute_test(test_type, test_input, timeout))
            
        except Exception as e:
            self.logger.error(f"Error running test: {e}")
            self.update_status(f"Test error: {e}")
            self.progress_bar.setVisible(False)
    
    def _execute_test(self, test_type: str, test_input: str, timeout: int):
        """Execute the actual test"""
        try:
            plugin_name = getattr(self.current_plugin, 'name', 'Unknown')
            result = None
            error = None
            
            self.progress_bar.setValue(25)
            
            if test_type == "Voice Command":
                result = self._test_voice_command(test_input)
            elif test_type == "Message":
                result = self._test_message(test_input)
            elif test_type == "Tool Execution":
                result = self._test_tool_execution(test_input)
            elif test_type == "Start/Stop":
                result = self._test_start_stop()
            elif test_type == "Config Test":
                result = self._test_config(test_input)
            
            self.progress_bar.setValue(75)
            
            # Record result
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.test_results[plugin_name] = {
                'timestamp': timestamp,
                'test_type': test_type,
                'input': test_input,
                'result': result,
                'error': error,
                'success': error is None
            }
            
            # Update UI
            self._update_results_display()
            self._update_status_table()
            
            if error:
                self.plugin_error.emit(plugin_name, error)
                self.update_status(f"Test failed: {error}")
            else:
                self.plugin_tested.emit(plugin_name, str(result))
                self.update_status(f"Test completed successfully")
            
            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            self.update_status(f"Test execution error: {e}")
            self.progress_bar.setVisible(False)
    
    def _test_voice_command(self, text: str) -> str:
        """Test voice command handling"""
        if hasattr(self.current_plugin, 'call_function'):
            return self.current_plugin.call_function('handle_voice_command', text)
        elif hasattr(self.current_plugin, 'module'):
            if 'handle_voice_command' in self.current_plugin.module:
                return self.current_plugin.module['handle_voice_command'](text, {})
        return "Voice command test not supported"
    
    def _test_message(self, text: str) -> str:
        """Test message handling"""
        if hasattr(self.current_plugin, 'call_function'):
            return self.current_plugin.call_function('handle_message', text)
        elif hasattr(self.current_plugin, 'module'):
            if 'handle_message' in self.current_plugin.module:
                return self.current_plugin.module['handle_message'](text, {})
        return "Message test not supported"
    
    def _test_tool_execution(self, params: str) -> str:
        """Test tool execution"""
        try:
            if params:
                tool_params = json.loads(params)
            else:
                tool_params = {}
            
            if hasattr(self.current_plugin, 'call_function'):
                return self.current_plugin.call_function('handle_tool', tool_params)
            elif hasattr(self.current_plugin, 'module'):
                if 'handle_tool' in self.current_plugin.module:
                    return self.current_plugin.module['handle_tool'](tool_params)
            return "Tool execution test not supported"
        except json.JSONDecodeError:
            return "Invalid JSON parameters"
    
    def _test_start_stop(self) -> str:
        """Test start/stop functionality"""
        try:
            # Test start
            if hasattr(self.current_plugin, 'start'):
                self.current_plugin.start()
            
            # Test stop
            if hasattr(self.current_plugin, 'stop'):
                self.current_plugin.stop()
            
            return "Start/Stop test completed"
        except Exception as e:
            return f"Start/Stop test failed: {e}"
    
    def _test_config(self, config_json: str) -> str:
        """Test configuration handling"""
        try:
            if config_json:
                config = json.loads(config_json)
            else:
                config = {}
            
            if hasattr(self.current_plugin, 'call_function'):
                return self.current_plugin.call_function('handle_config', config)
            elif hasattr(self.current_plugin, 'module'):
                if 'handle_config' in self.current_plugin.module:
                    return self.current_plugin.module['handle_config'](config)
            return "Config test not supported"
        except json.JSONDecodeError:
            return "Invalid JSON configuration"
    
    def run_all_tests(self):
        """Run tests on all plugins"""
        if not self.plugins:
            QMessageBox.warning(self, "No Plugins", "No plugins available for testing")
            return
        
        try:
            self.update_status("Running tests on all plugins...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(self.plugins))
            
            for i, plugin in enumerate(self.plugins):
                self.progress_bar.setValue(i)
                self.current_plugin = plugin
                
                # Run basic test
                test_input = "Test input for all plugins"
                self._execute_test("Message", test_input, 5)
            
            self.progress_bar.setValue(len(self.plugins))
            self.update_status("All plugin tests completed")
            
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.logger.error(f"Error running all tests: {e}")
            self.update_status(f"Error running all tests: {e}")
            self.progress_bar.setVisible(False)
    
    def _update_results_display(self):
        """Update the results text display"""
        self.results_text.clear()
        
        for plugin_name, result in self.test_results.items():
            self.results_text.append(f"=== {plugin_name} ===\n")
            self.results_text.append(f"Time: {result['timestamp']}\n")
            self.results_text.append(f"Test: {result['test_type']}\n")
            self.results_text.append(f"Input: {result['input']}\n")
            self.results_text.append(f"Result: {result['result']}\n")
            if result['error']:
                self.results_text.append(f"Error: {result['error']}\n")
            self.results_text.append(f"Success: {result['success']}\n")
            self.results_text.append("-" * 50 + "\n")
    
    def _update_status_table(self):
        """Update the status table"""
        self.status_table.setRowCount(len(self.test_results))
        
        for row, (plugin_name, result) in enumerate(self.test_results.items()):
            self.status_table.setItem(row, 0, QTableWidgetItem(plugin_name))
            
            status_item = QTableWidgetItem("Success" if result['success'] else "Failed")
            status_item.setBackground(QColor("#28A745" if result['success'] else "#DC3545"))
            self.status_table.setItem(row, 1, status_item)
            
            self.status_table.setItem(row, 2, QTableWidgetItem(result['timestamp']))
            self.status_table.setItem(row, 3, QTableWidgetItem(str(result['result'])[:50]))
    
    def clear_results(self):
        """Clear all test results"""
        self.test_results.clear()
        self.results_text.clear()
        self.status_table.setRowCount(0)
        self.debug_text.clear()
        self.update_status("Results cleared")
    
    def update_status(self, message: str):
        """Update status display"""
        self.status_label.setText(message)
        self.logger.info(f"Plugin Test Status: {message}")
    
    def add_debug_log(self, message: str):
        """Add debug log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.debug_text.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.debug_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.debug_text.setTextCursor(cursor)
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        self.logger.info("Plugin test dialog closed")
        event.accept()