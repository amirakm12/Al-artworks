"""
Performance Panel for Athena 3D Avatar UI
Real-time performance monitoring and optimization controls
"""

import sys
import time
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QProgressBar, QPushButton, QComboBox,
                             QSlider, QCheckBox, QGroupBox, QTextEdit, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor

from core.performance_monitor import PerformanceMonitor, MetricType
from utils.logger import setup_logger

class PerformancePanel(QWidget):
    """Performance monitoring and control panel"""
    
    # Signals
    performance_alert = pyqtSignal(str, str)  # metric_name, message
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        super().__init__()
        self.logger = setup_logger()
        self.performance_monitor = performance_monitor
        
        # UI components
        self.metric_labels: Dict[str, QLabel] = {}
        self.metric_bars: Dict[str, QProgressBar] = {}
        self.update_timer = QTimer()
        
        # Initialize UI
        self.setup_ui()
        self.setup_connections()
        self.start_monitoring()
        
    def setup_ui(self):
        """Setup the performance panel UI"""
        try:
            layout = QVBoxLayout()
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # System metrics tab
            system_tab = self.create_system_tab()
            tab_widget.addTab(system_tab, "System")
            
            # Application metrics tab
            app_tab = self.create_application_tab()
            tab_widget.addTab(app_tab, "Application")
            
            # Model metrics tab
            model_tab = self.create_model_tab()
            tab_widget.addTab(model_tab, "Models")
            
            # Rendering metrics tab
            render_tab = self.create_rendering_tab()
            tab_widget.addTab(render_tab, "Rendering")
            
            # Audio metrics tab
            audio_tab = self.create_audio_tab()
            tab_widget.addTab(audio_tab, "Audio")
            
            # Controls tab
            controls_tab = self.create_controls_tab()
            tab_widget.addTab(controls_tab, "Controls")
            
            layout.addWidget(tab_widget)
            self.setLayout(layout)
            
            self.logger.info("Performance panel UI setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup performance panel UI: {e}")
    
    def create_system_tab(self) -> QWidget:
        """Create system metrics tab"""
        try:
            widget = QWidget()
            layout = QGridLayout()
            
            # CPU Usage
            cpu_label = QLabel("CPU Usage:")
            cpu_bar = QProgressBar()
            cpu_bar.setRange(0, 100)
            cpu_value = QLabel("0%")
            layout.addWidget(cpu_label, 0, 0)
            layout.addWidget(cpu_bar, 0, 1)
            layout.addWidget(cpu_value, 0, 2)
            
            # Memory Usage
            memory_label = QLabel("Memory Usage:")
            memory_bar = QProgressBar()
            memory_bar.setRange(0, 12)  # 12GB max
            memory_value = QLabel("0 GB")
            layout.addWidget(memory_label, 1, 0)
            layout.addWidget(memory_bar, 1, 1)
            layout.addWidget(memory_value, 1, 2)
            
            # GPU Usage
            gpu_label = QLabel("GPU Usage:")
            gpu_bar = QProgressBar()
            gpu_bar.setRange(0, 100)
            gpu_value = QLabel("0%")
            layout.addWidget(gpu_label, 2, 0)
            layout.addWidget(gpu_bar, 2, 1)
            layout.addWidget(gpu_value, 2, 2)
            
            # GPU Memory
            gpu_memory_label = QLabel("GPU Memory:")
            gpu_memory_bar = QProgressBar()
            gpu_memory_bar.setRange(0, 8)  # 8GB max
            gpu_memory_value = QLabel("0 GB")
            layout.addWidget(gpu_memory_label, 3, 0)
            layout.addWidget(gpu_memory_bar, 3, 1)
            layout.addWidget(gpu_memory_value, 3, 2)
            
            # Store references
            self.metric_labels[MetricType.CPU_USAGE.value] = cpu_value
            self.metric_bars[MetricType.CPU_USAGE.value] = cpu_bar
            self.metric_labels[MetricType.MEMORY_USAGE.value] = memory_value
            self.metric_bars[MetricType.MEMORY_USAGE.value] = memory_bar
            self.metric_labels[MetricType.GPU_USAGE.value] = gpu_value
            self.metric_bars[MetricType.GPU_USAGE.value] = gpu_bar
            self.metric_labels[MetricType.GPU_MEMORY.value] = gpu_memory_value
            self.metric_bars[MetricType.GPU_MEMORY.value] = gpu_memory_bar
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create system tab: {e}")
            return QWidget()
    
    def create_application_tab(self) -> QWidget:
        """Create application metrics tab"""
        try:
            widget = QWidget()
            layout = QGridLayout()
            
            # FPS
            fps_label = QLabel("FPS:")
            fps_bar = QProgressBar()
            fps_bar.setRange(0, 60)
            fps_value = QLabel("0")
            layout.addWidget(fps_label, 0, 0)
            layout.addWidget(fps_bar, 0, 1)
            layout.addWidget(fps_value, 0, 2)
            
            # Frame Time
            frame_time_label = QLabel("Frame Time:")
            frame_time_bar = QProgressBar()
            frame_time_bar.setRange(0, 50)  # 50ms max
            frame_time_value = QLabel("0 ms")
            layout.addWidget(frame_time_label, 1, 0)
            layout.addWidget(frame_time_bar, 1, 1)
            layout.addWidget(frame_time_value, 1, 2)
            
            # Latency
            latency_label = QLabel("Latency:")
            latency_bar = QProgressBar()
            latency_bar.setRange(0, 300)  # 300ms max
            latency_value = QLabel("0 ms")
            layout.addWidget(latency_label, 2, 0)
            layout.addWidget(latency_bar, 2, 1)
            layout.addWidget(latency_value, 2, 2)
            
            # Throughput
            throughput_label = QLabel("Throughput:")
            throughput_bar = QProgressBar()
            throughput_bar.setRange(0, 1000)
            throughput_value = QLabel("0 ops/s")
            layout.addWidget(throughput_label, 3, 0)
            layout.addWidget(throughput_bar, 3, 1)
            layout.addWidget(throughput_value, 3, 2)
            
            # Store references
            self.metric_labels[MetricType.FPS.value] = fps_value
            self.metric_bars[MetricType.FPS.value] = fps_bar
            self.metric_labels[MetricType.FRAME_TIME.value] = frame_time_value
            self.metric_bars[MetricType.FRAME_TIME.value] = frame_time_bar
            self.metric_labels[MetricType.LATENCY.value] = latency_value
            self.metric_bars[MetricType.LATENCY.value] = latency_bar
            self.metric_labels[MetricType.THROUGHPUT.value] = throughput_value
            self.metric_bars[MetricType.THROUGHPUT.value] = throughput_bar
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create application tab: {e}")
            return QWidget()
    
    def create_model_tab(self) -> QWidget:
        """Create model metrics tab"""
        try:
            widget = QWidget()
            layout = QGridLayout()
            
            # Model Inference Time
            inference_label = QLabel("Inference Time:")
            inference_bar = QProgressBar()
            inference_bar.setRange(0, 200)  # 200ms max
            inference_value = QLabel("0 ms")
            layout.addWidget(inference_label, 0, 0)
            layout.addWidget(inference_bar, 0, 1)
            layout.addWidget(inference_value, 0, 2)
            
            # Model Memory Usage
            model_memory_label = QLabel("Model Memory:")
            model_memory_bar = QProgressBar()
            model_memory_bar.setRange(0, 4)  # 4GB max
            model_memory_value = QLabel("0 GB")
            layout.addWidget(model_memory_label, 1, 0)
            layout.addWidget(model_memory_bar, 1, 1)
            layout.addWidget(model_memory_value, 1, 2)
            
            # Model Accuracy
            accuracy_label = QLabel("Model Accuracy:")
            accuracy_bar = QProgressBar()
            accuracy_bar.setRange(0, 100)
            accuracy_value = QLabel("0%")
            layout.addWidget(accuracy_label, 2, 0)
            layout.addWidget(accuracy_bar, 2, 1)
            layout.addWidget(accuracy_value, 2, 2)
            
            # Store references
            self.metric_labels[MetricType.MODEL_INFERENCE_TIME.value] = inference_value
            self.metric_bars[MetricType.MODEL_INFERENCE_TIME.value] = inference_bar
            self.metric_labels[MetricType.MODEL_MEMORY_USAGE.value] = model_memory_value
            self.metric_bars[MetricType.MODEL_MEMORY_USAGE.value] = model_memory_bar
            self.metric_labels[MetricType.MODEL_ACCURACY.value] = accuracy_value
            self.metric_bars[MetricType.MODEL_ACCURACY.value] = accuracy_bar
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create model tab: {e}")
            return QWidget()
    
    def create_rendering_tab(self) -> QWidget:
        """Create rendering metrics tab"""
        try:
            widget = QWidget()
            layout = QGridLayout()
            
            # Render Time
            render_time_label = QLabel("Render Time:")
            render_time_bar = QProgressBar()
            render_time_bar.setRange(0, 33)  # 33ms max (30 FPS)
            render_time_value = QLabel("0 ms")
            layout.addWidget(render_time_label, 0, 0)
            layout.addWidget(render_time_bar, 0, 1)
            layout.addWidget(render_time_value, 0, 2)
            
            # Draw Calls
            draw_calls_label = QLabel("Draw Calls:")
            draw_calls_bar = QProgressBar()
            draw_calls_bar.setRange(0, 2000)
            draw_calls_value = QLabel("0")
            layout.addWidget(draw_calls_label, 1, 0)
            layout.addWidget(draw_calls_bar, 1, 1)
            layout.addWidget(draw_calls_value, 1, 2)
            
            # Triangle Count
            triangle_count_label = QLabel("Triangle Count:")
            triangle_count_bar = QProgressBar()
            triangle_count_bar.setRange(0, 200000)
            triangle_count_value = QLabel("0")
            layout.addWidget(triangle_count_label, 2, 0)
            layout.addWidget(triangle_count_bar, 2, 1)
            layout.addWidget(triangle_count_value, 2, 2)
            
            # Store references
            self.metric_labels[MetricType.RENDER_TIME.value] = render_time_value
            self.metric_bars[MetricType.RENDER_TIME.value] = render_time_bar
            self.metric_labels[MetricType.DRAW_CALLS.value] = draw_calls_value
            self.metric_bars[MetricType.DRAW_CALLS.value] = draw_calls_bar
            self.metric_labels[MetricType.TRIANGLE_COUNT.value] = triangle_count_value
            self.metric_bars[MetricType.TRIANGLE_COUNT.value] = triangle_count_bar
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create rendering tab: {e}")
            return QWidget()
    
    def create_audio_tab(self) -> QWidget:
        """Create audio metrics tab"""
        try:
            widget = QWidget()
            layout = QGridLayout()
            
            # Audio Latency
            audio_latency_label = QLabel("Audio Latency:")
            audio_latency_bar = QProgressBar()
            audio_latency_bar.setRange(0, 100)  # 100ms max
            audio_latency_value = QLabel("0 ms")
            layout.addWidget(audio_latency_label, 0, 0)
            layout.addWidget(audio_latency_bar, 0, 1)
            layout.addWidget(audio_latency_value, 0, 2)
            
            # Audio Quality
            audio_quality_label = QLabel("Audio Quality:")
            audio_quality_bar = QProgressBar()
            audio_quality_bar.setRange(0, 100)
            audio_quality_value = QLabel("0%")
            layout.addWidget(audio_quality_label, 1, 0)
            layout.addWidget(audio_quality_bar, 1, 1)
            layout.addWidget(audio_quality_value, 1, 2)
            
            # Voice Synthesis Time
            synthesis_time_label = QLabel("Synthesis Time:")
            synthesis_time_bar = QProgressBar()
            synthesis_time_bar.setRange(0, 500)  # 500ms max
            synthesis_time_value = QLabel("0 ms")
            layout.addWidget(synthesis_time_label, 2, 0)
            layout.addWidget(synthesis_time_bar, 2, 1)
            layout.addWidget(synthesis_time_value, 2, 2)
            
            # Store references
            self.metric_labels[MetricType.AUDIO_LATENCY.value] = audio_latency_value
            self.metric_bars[MetricType.AUDIO_LATENCY.value] = audio_latency_bar
            self.metric_labels[MetricType.AUDIO_QUALITY.value] = audio_quality_value
            self.metric_bars[MetricType.AUDIO_QUALITY.value] = audio_quality_bar
            self.metric_labels[MetricType.VOICE_SYNTHESIS_TIME.value] = synthesis_time_value
            self.metric_bars[MetricType.VOICE_SYNTHESIS_TIME.value] = synthesis_time_bar
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create audio tab: {e}")
            return QWidget()
    
    def create_controls_tab(self) -> QWidget:
        """Create performance control tab"""
        try:
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Performance mode selection
            mode_group = QGroupBox("Performance Mode")
            mode_layout = QVBoxLayout()
            
            mode_combo = QComboBox()
            mode_combo.addItems(["Low", "Medium", "High", "Ultra"])
            mode_combo.setCurrentText("Medium")
            mode_layout.addWidget(QLabel("Quality Level:"))
            mode_layout.addWidget(mode_combo)
            
            mode_group.setLayout(mode_layout)
            layout.addWidget(mode_group)
            
            # Threshold controls
            threshold_group = QGroupBox("Alert Thresholds")
            threshold_layout = QGridLayout()
            
            # CPU threshold
            threshold_layout.addWidget(QLabel("CPU Usage (%):"), 0, 0)
            cpu_threshold = QSlider(Qt.Orientation.Horizontal)
            cpu_threshold.setRange(50, 95)
            cpu_threshold.setValue(80)
            threshold_layout.addWidget(cpu_threshold, 0, 1)
            
            # Memory threshold
            threshold_layout.addWidget(QLabel("Memory Usage (GB):"), 1, 0)
            memory_threshold = QSlider(Qt.Orientation.Horizontal)
            memory_threshold.setRange(8, 12)
            memory_threshold.setValue(10)
            threshold_layout.addWidget(memory_threshold, 1, 1)
            
            # FPS threshold
            threshold_layout.addWidget(QLabel("Minimum FPS:"), 2, 0)
            fps_threshold = QSlider(Qt.Orientation.Horizontal)
            fps_threshold.setRange(15, 60)
            fps_threshold.setValue(30)
            threshold_layout.addWidget(fps_threshold, 2, 1)
            
            threshold_group.setLayout(threshold_layout)
            layout.addWidget(threshold_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            refresh_btn = QPushButton("Refresh")
            refresh_btn.clicked.connect(self.refresh_metrics)
            button_layout.addWidget(refresh_btn)
            
            export_btn = QPushButton("Export Report")
            export_btn.clicked.connect(self.export_report)
            button_layout.addWidget(export_btn)
            
            clear_btn = QPushButton("Clear History")
            clear_btn.clicked.connect(self.clear_history)
            button_layout.addWidget(clear_btn)
            
            layout.addLayout(button_layout)
            
            # Performance report text area
            report_group = QGroupBox("Performance Report")
            report_layout = QVBoxLayout()
            
            self.report_text = QTextEdit()
            self.report_text.setMaximumHeight(150)
            report_layout.addWidget(self.report_text)
            
            report_group.setLayout(report_layout)
            layout.addWidget(report_group)
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            self.logger.error(f"Failed to create controls tab: {e}")
            return QWidget()
    
    def setup_connections(self):
        """Setup signal connections"""
        try:
            # Connect update timer
            self.update_timer.timeout.connect(self.update_metrics)
            
            # Connect performance monitor alerts
            for metric_type in MetricType:
                self.performance_monitor.register_alert_callback(metric_type, self.handle_performance_alert)
            
            self.logger.info("Performance panel connections setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup connections: {e}")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        try:
            # Start update timer (update every 500ms)
            self.update_timer.start(500)
            
            # Start performance monitor
            self.performance_monitor.start_monitoring()
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.update_timer.stop()
            self.performance_monitor.stop_monitoring()
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def update_metrics(self):
        """Update all metric displays"""
        try:
            summary = self.performance_monitor.get_performance_summary()
            
            for metric_name, data in summary.items():
                if metric_name in self.metric_labels and metric_name in self.metric_bars:
                    # Update label
                    value_text = f"{data['current']:.1f}{data['unit']}"
                    self.metric_labels[metric_name].setText(value_text)
                    
                    # Update progress bar
                    bar = self.metric_bars[metric_name]
                    current_value = data['current']
                    
                    # Set bar value based on metric type
                    if metric_name in ['fps', 'throughput']:
                        # For metrics where higher is better
                        bar.setValue(int(current_value))
                    else:
                        # For metrics where lower is better, show as percentage of threshold
                        threshold = self.performance_monitor.get_threshold(MetricType(metric_name))
                        if threshold:
                            percentage = min(100, (current_value / threshold) * 100)
                            bar.setValue(int(percentage))
                    
                    # Set bar color based on performance
                    self._set_bar_color(bar, metric_name, data['current'])
            
            # Update performance report
            self.update_performance_report()
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def _set_bar_color(self, bar: QProgressBar, metric_name: str, value: float):
        """Set progress bar color based on performance"""
        try:
            threshold = self.performance_monitor.get_threshold(MetricType(metric_name))
            if not threshold:
                return
            
            # Determine if metric is good or bad
            if metric_name in ['fps', 'throughput']:
                # Higher is better
                is_good = value >= threshold
            else:
                # Lower is better
                is_good = value <= threshold
            
            # Set color
            if is_good:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
                
        except Exception as e:
            self.logger.error(f"Failed to set bar color: {e}")
    
    def update_performance_report(self):
        """Update the performance report text"""
        try:
            report = self.performance_monitor.get_performance_report()
            self.report_text.setPlainText(report)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance report: {e}")
    
    def handle_performance_alert(self, metric_type: MetricType, metric, threshold: float):
        """Handle performance alerts"""
        try:
            alert_message = f"Performance alert: {metric_type.value} = {metric.value}{metric.unit}"
            self.performance_alert.emit(metric_type.value, alert_message)
            
        except Exception as e:
            self.logger.error(f"Failed to handle performance alert: {e}")
    
    def refresh_metrics(self):
        """Manually refresh metrics"""
        try:
            self.update_metrics()
            self.logger.info("Metrics refreshed manually")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh metrics: {e}")
    
    def export_report(self):
        """Export performance report to file"""
        try:
            report = self.performance_monitor.get_performance_report()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"athena_performance_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Performance report exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
    
    def clear_history(self):
        """Clear performance history"""
        try:
            self.performance_monitor.clear_metric_history()
            self.logger.info("Performance history cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear history: {e}")
    
    def cleanup(self):
        """Cleanup performance panel"""
        try:
            self.stop_monitoring()
            self.logger.info("Performance panel cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Performance panel cleanup failed: {e}")