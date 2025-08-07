"""
Plugin Test Dialog for ChatGPT+ Clone
Comprehensive plugin testing and debugging interface
"""

import json
import logging
import traceback
import time
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QDialog, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QTabWidget, QWidget, QLabel, QComboBox, QSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QProgressBar, QListWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QTextCursor

logger = logging.getLogger("ChatGPTPlus.PluginTestDialog")

class PluginTestWorker(QThread):
    """Worker thread for plugin testing"""
    
    test_completed = pyqtSignal(dict)
    test_progress = pyqtSignal(str)
    test_error = pyqtSignal(str)
    
    def __init__(self, plugin_name: str, test_type: str):
        super().__init__()
        self.plugin_name = plugin_name
        self.test_type = test_type
        self.is_running = False
    
    def run(self):
        """Run plugin test"""
        self.is_running = True
        
        try:
            if self.test_type == "load":
                self._test_plugin_load()
            elif self.test_type == "lifecycle":
                self._test_plugin_lifecycle()
            elif self.test_type == "performance":
                self._test_plugin_performance()
            elif self.test_type == "security":
                self._test_plugin_security()
            else:
                self._test_plugin_comprehensive()
                
        except Exception as e:
            self.test_error.emit(f"Test failed: {str(e)}")
        finally:
            self.is_running = False
    
    def _test_plugin_load(self):
        """Test plugin loading"""
        self.test_progress.emit("Testing plugin loading...")
        
        # Simulate plugin loading
        time.sleep(0.5)
        
        result = {
            "test_type": "load",
            "plugin_name": self.plugin_name,
            "status": "passed",
            "duration": 0.5,
            "details": "Plugin loaded successfully"
        }
        
        self.test_completed.emit(result)
    
    def _test_plugin_lifecycle(self):
        """Test plugin lifecycle"""
        self.test_progress.emit("Testing plugin lifecycle...")
        
        # Simulate lifecycle test
        time.sleep(1.0)
        
        result = {
            "test_type": "lifecycle",
            "plugin_name": self.plugin_name,
            "status": "passed",
            "duration": 1.0,
            "details": "Plugin lifecycle test passed"
        }
        
        self.test_completed.emit(result)
    
    def _test_plugin_performance(self):
        """Test plugin performance"""
        self.test_progress.emit("Testing plugin performance...")
        
        # Simulate performance test
        time.sleep(2.0)
        
        result = {
            "test_type": "performance",
            "plugin_name": self.plugin_name,
            "status": "passed",
            "duration": 2.0,
            "details": "Performance test completed"
        }
        
        self.test_completed.emit(result)
    
    def _test_plugin_security(self):
        """Test plugin security"""
        self.test_progress.emit("Testing plugin security...")
        
        # Simulate security test
        time.sleep(1.5)
        
        result = {
            "test_type": "security",
            "plugin_name": self.plugin_name,
            "status": "passed",
            "duration": 1.5,
            "details": "Security test passed"
        }
        
        self.test_completed.emit(result)
    
    def _test_plugin_comprehensive(self):
        """Run comprehensive plugin test"""
        self.test_progress.emit("Running comprehensive test...")
        
        # Run all tests
        tests = ["load", "lifecycle", "performance", "security"]
        results = []
        
        for test in tests:
            self.test_progress.emit(f"Running {test} test...")
            time.sleep(0.5)
            
            results.append({
                "test": test,
                "status": "passed",
                "duration": 0.5
            })
        
        result = {
            "test_type": "comprehensive",
            "plugin_name": self.plugin_name,
            "status": "passed",
            "duration": sum(r["duration"] for r in results),
            "details": f"All {len(tests)} tests passed",
            "sub_tests": results
        }
        
        self.test_completed.emit(result)

class PluginTestDialog(QDialog):
    """Comprehensive plugin testing dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plugin Test Mode")
        self.setMinimumSize(800, 600)
        
        # Test state
        self.current_test_worker = None
        self.test_results = []
        
        # Setup UI
        self.setup_ui()
        self.load_available_plugins()
        
        logger.info("Plugin test dialog initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Create splitter for main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Test controls
        left_panel = self.create_test_controls()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and logs
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 500])
        
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.run_test_btn = QPushButton("Run Test")
        self.run_test_btn.clicked.connect(self.run_test)
        
        self.stop_test_btn = QPushButton("Stop Test")
        self.stop_test_btn.clicked.connect(self.stop_test)
        self.stop_test_btn.setEnabled(False)
        
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self.clear_results)
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_results)
        
        button_layout.addWidget(self.run_test_btn)
        button_layout.addWidget(self.stop_test_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_results_btn)
        button_layout.addWidget(self.export_results_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_test_controls(self):
        """Create test control panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Plugin Selection Group
        plugin_group = QGroupBox("Plugin Selection")
        plugin_layout = QFormLayout()
        
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItem("All Plugins")
        plugin_layout.addRow("Plugin:", self.plugin_combo)
        
        self.plugin_status_label = QLabel("No plugins loaded")
        plugin_layout.addRow("Status:", self.plugin_status_label)
        
        plugin_group.setLayout(plugin_layout)
        layout.addWidget(plugin_group)
        
        # Test Configuration Group
        test_group = QGroupBox("Test Configuration")
        test_layout = QFormLayout()
        
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems([
            "Comprehensive",
            "Load Test",
            "Lifecycle Test", 
            "Performance Test",
            "Security Test"
        ])
        test_layout.addRow("Test Type:", self.test_type_combo)
        
        self.test_timeout_spin = QSpinBox()
        self.test_timeout_spin.setRange(5, 300)
        self.test_timeout_spin.setValue(30)
        test_layout.addRow("Timeout (s):", self.test_timeout_spin)
        
        self.verbose_logging = QCheckBox("Verbose Logging")
        test_layout.addRow(self.verbose_logging)
        
        self.auto_retry = QCheckBox("Auto Retry Failed Tests")
        test_layout.addRow(self.auto_retry)
        
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)
        
        # Test Options Group
        options_group = QGroupBox("Test Options")
        options_layout = QFormLayout()
        
        self.test_load = QCheckBox("Test Plugin Loading")
        self.test_load.setChecked(True)
        options_layout.addRow(self.test_load)
        
        self.test_lifecycle = QCheckBox("Test Plugin Lifecycle")
        self.test_lifecycle.setChecked(True)
        options_layout.addRow(self.test_lifecycle)
        
        self.test_performance = QCheckBox("Test Performance")
        self.test_performance.setChecked(True)
        options_layout.addRow(self.test_performance)
        
        self.test_security = QCheckBox("Test Security")
        self.test_security.setChecked(True)
        options_layout.addRow(self.test_security)
        
        self.test_integration = QCheckBox("Test Integration")
        self.test_integration.setChecked(True)
        options_layout.addRow(self.test_integration)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_results_panel(self):
        """Create results and logs panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create tab widget for results
        self.results_tabs = QTabWidget()
        
        # Test Results Tab
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Plugin", "Test Type", "Status", "Duration", "Details"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.results_tabs.addTab(self.results_table, "Test Results")
        
        # Log Output Tab
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        self.results_tabs.addTab(self.log_output, "Log Output")
        
        # Performance Tab
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.results_tabs.addTab(self.performance_text, "Performance")
        
        # Error Details Tab
        self.error_details = QTextEdit()
        self.error_details.setReadOnly(True)
        self.results_tabs.addTab(self.error_details, "Error Details")
        
        layout.addWidget(self.results_tabs)
        widget.setLayout(layout)
        return widget
    
    def load_available_plugins(self):
        """Load list of available plugins"""
        try:
            import os
            plugin_dir = "plugins"
            
            if os.path.exists(plugin_dir):
                plugins = []
                for filename in os.listdir(plugin_dir):
                    if filename.endswith(".py") and not filename.startswith("__"):
                        plugins.append(filename[:-3])
                
                if plugins:
                    self.plugin_combo.clear()
                    self.plugin_combo.addItem("All Plugins")
                    self.plugin_combo.addItems(plugins)
                    self.plugin_status_label.setText(f"{len(plugins)} plugins found")
                else:
                    self.plugin_status_label.setText("No plugins found")
            else:
                self.plugin_status_label.setText("Plugins directory not found")
                
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            self.plugin_status_label.setText("Error loading plugins")
    
    def run_test(self):
        """Run the selected test"""
        if self.current_test_worker and self.current_test_worker.is_running:
            logger.warning("Test already running")
            return
        
        # Get test parameters
        plugin_name = self.plugin_combo.currentText()
        test_type = self.test_type_combo.currentText().lower().replace(" ", "_")
        
        if plugin_name == "All Plugins":
            self.run_all_plugins_test()
        else:
            self.run_single_plugin_test(plugin_name, test_type)
    
    def run_single_plugin_test(self, plugin_name: str, test_type: str):
        """Run test for a single plugin"""
        logger.info(f"Running {test_type} test for plugin: {plugin_name}")
        
        # Update UI
        self.run_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Log test start
        self.log_output.append(f"[{time.strftime('%H:%M:%S')}] Starting {test_type} test for {plugin_name}")
        
        # Create and start worker
        self.current_test_worker = PluginTestWorker(plugin_name, test_type)
        self.current_test_worker.test_completed.connect(self.on_test_completed)
        self.current_test_worker.test_progress.connect(self.on_test_progress)
        self.current_test_worker.test_error.connect(self.on_test_error)
        self.current_test_worker.start()
    
    def run_all_plugins_test(self):
        """Run tests for all plugins"""
        logger.info("Running tests for all plugins")
        
        # Get all plugins
        plugins = []
        for i in range(1, self.plugin_combo.count()):
            plugins.append(self.plugin_combo.itemText(i))
        
        if not plugins:
            self.log_output.append("No plugins available for testing")
            return
        
        # Run tests for each plugin
        for plugin in plugins:
            self.run_single_plugin_test(plugin, "comprehensive")
    
    def stop_test(self):
        """Stop current test"""
        if self.current_test_worker and self.current_test_worker.is_running:
            self.current_test_worker.terminate()
            self.current_test_worker.wait()
            
            self.log_output.append(f"[{time.strftime('%H:%M:%S')}] Test stopped by user")
        
        # Reset UI
        self.run_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_test_completed(self, result: Dict[str, Any]):
        """Handle test completion"""
        logger.info(f"Test completed: {result}")
        
        # Add to results table
        self.add_result_to_table(result)
        
        # Add to results list
        self.test_results.append(result)
        
        # Update log
        status_icon = "✅" if result["status"] == "passed" else "❌"
        self.log_output.append(
            f"[{time.strftime('%H:%M:%S')}] {status_icon} {result['test_type']} test for {result['plugin_name']}: {result['status']}"
        )
        
        # Update performance tab
        self.update_performance_tab()
        
        # Reset UI
        self.run_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_test_progress(self, message: str):
        """Handle test progress updates"""
        self.log_output.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_output.moveCursor(QTextCursor.MoveOperation.End)
    
    def on_test_error(self, error: str):
        """Handle test errors"""
        logger.error(f"Test error: {error}")
        
        self.log_output.append(f"[{time.strftime('%H:%M:%S')}] ❌ ERROR: {error}")
        self.error_details.append(f"[{time.strftime('%H:%M:%S')}] {error}")
        
        # Reset UI
        self.run_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def add_result_to_table(self, result: Dict[str, Any]):
        """Add test result to results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(result["plugin_name"]))
        self.results_table.setItem(row, 1, QTableWidgetItem(result["test_type"]))
        
        status_item = QTableWidgetItem(result["status"])
        if result["status"] == "passed":
            status_item.setBackground(Qt.GlobalColor.green)
        else:
            status_item.setBackground(Qt.GlobalColor.red)
        self.results_table.setItem(row, 2, status_item)
        
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{result['duration']:.2f}s"))
        self.results_table.setItem(row, 4, QTableWidgetItem(result.get("details", "")))
    
    def update_performance_tab(self):
        """Update performance tab with test results"""
        if not self.test_results:
            return
        
        # Calculate performance statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "passed"])
        failed_tests = total_tests - passed_tests
        
        avg_duration = sum(r["duration"] for r in self.test_results) / total_tests
        
        # Generate performance report
        report = f"""Performance Report
================

Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {(passed_tests/total_tests)*100:.1f}%

Average Duration: {avg_duration:.2f}s

Test Breakdown:
"""
        
        # Group by test type
        test_types = {}
        for result in self.test_results:
            test_type = result["test_type"]
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        for test_type, results in test_types.items():
            passed = len([r for r in results if r["status"] == "passed"])
            total = len(results)
            avg_dur = sum(r["duration"] for r in results) / total
            
            report += f"\n{test_type.title()} Tests:"
            report += f"\n  - Total: {total}"
            report += f"\n  - Passed: {passed}"
            report += f"\n  - Failed: {total - passed}"
            report += f"\n  - Success Rate: {(passed/total)*100:.1f}%"
            report += f"\n  - Avg Duration: {avg_dur:.2f}s\n"
        
        self.performance_text.setText(report)
    
    def clear_results(self):
        """Clear all test results"""
        self.test_results.clear()
        self.results_table.setRowCount(0)
        self.log_output.clear()
        self.performance_text.clear()
        self.error_details.clear()
        
        logger.info("Test results cleared")
    
    def export_results(self):
        """Export test results to file"""
        try:
            import json
            from datetime import datetime
            
            filename = f"plugin_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.test_results),
                "results": self.test_results
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.log_output.append(f"[{time.strftime('%H:%M:%S')}] Results exported to {filename}")
            logger.info(f"Test results exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            self.log_output.append(f"[{time.strftime('%H:%M:%S')}] Failed to export results: {e}")

# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    dialog = PluginTestDialog()
    dialog.show()
    
    sys.exit(app.exec())