"""
Performance Monitor for Athena 3D Avatar
Comprehensive performance tracking and optimization system
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import psutil
import threading
from dataclasses import dataclass
from enum import Enum
from collections import deque

class MetricType(Enum):
    # System Metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    GPU_MEMORY = "gpu_memory"
    
    # Application Metrics
    FPS = "fps"
    FRAME_TIME = "frame_time"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    
    # Model Metrics
    MODEL_INFERENCE_TIME = "model_inference_time"
    MODEL_MEMORY_USAGE = "model_memory_usage"
    MODEL_ACCURACY = "model_accuracy"
    
    # Rendering Metrics
    RENDER_TIME = "render_time"
    DRAW_CALLS = "draw_calls"
    TRIANGLE_COUNT = "triangle_count"
    
    # Audio Metrics
    AUDIO_LATENCY = "audio_latency"
    AUDIO_QUALITY = "audio_quality"
    VOICE_SYNTHESIS_TIME = "voice_synthesis_time"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    metric_type: MetricType
    value: float
    timestamp: float
    unit: str = ""
    description: str = ""

class PerformanceMonitor:
    """Advanced performance monitoring system for Athena"""
    
    def __init__(self, max_history_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.metrics_history: Dict[MetricType, deque] = {}
        self.max_history_size = max_history_size
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 0.1  # 100ms intervals
        
        # Performance thresholds
        self.thresholds: Dict[MetricType, float] = {}
        self.alert_callbacks: Dict[MetricType, List[callable]] = {}
        
        # Initialize monitoring
        self._initialize_monitoring()
        
    def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        try:
            # Initialize metric history for all types
            for metric_type in MetricType:
                self.metrics_history[metric_type] = deque(maxlen=self.max_history_size)
            
            # Set default thresholds
            self._set_default_thresholds()
            
            self.logger.info("Performance monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitoring: {e}")
    
    def _set_default_thresholds(self):
        """Set default performance thresholds"""
        try:
            # System thresholds
            self.thresholds[MetricType.CPU_USAGE] = 80.0  # 80% CPU usage
            self.thresholds[MetricType.MEMORY_USAGE] = 10.0  # 10GB RAM usage
            self.thresholds[MetricType.GPU_USAGE] = 90.0  # 90% GPU usage
            self.thresholds[MetricType.GPU_MEMORY] = 8.0  # 8GB GPU memory
            
            # Application thresholds
            self.thresholds[MetricType.FPS] = 30.0  # Minimum 30 FPS
            self.thresholds[MetricType.FRAME_TIME] = 33.33  # Maximum 33.33ms frame time
            self.thresholds[MetricType.LATENCY] = 250.0  # Maximum 250ms latency
            
            # Model thresholds
            self.thresholds[MetricType.MODEL_INFERENCE_TIME] = 100.0  # 100ms inference time
            self.thresholds[MetricType.MODEL_MEMORY_USAGE] = 2.0  # 2GB model memory
            
            # Rendering thresholds
            self.thresholds[MetricType.RENDER_TIME] = 16.67  # 16.67ms render time
            self.thresholds[MetricType.DRAW_CALLS] = 1000  # 1000 draw calls
            self.thresholds[MetricType.TRIANGLE_COUNT] = 100000  # 100k triangles
            
            # Audio thresholds
            self.thresholds[MetricType.AUDIO_LATENCY] = 50.0  # 50ms audio latency
            self.thresholds[MetricType.VOICE_SYNTHESIS_TIME] = 200.0  # 200ms synthesis time
            
        except Exception as e:
            self.logger.error(f"Failed to set default thresholds: {e}")
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self.monitor_thread.start()
                self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        try:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check thresholds and trigger alerts
                self._check_thresholds()
                
                # Wait for next interval
                time.sleep(self.monitor_interval)
                
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric(MetricType.CPU_USAGE, cpu_percent, "%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            self.record_metric(MetricType.MEMORY_USAGE, memory_gb, "GB")
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                self.record_metric(MetricType.GPU_USAGE, gpu_usage, "%")
                
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                self.record_metric(MetricType.GPU_MEMORY, gpu_memory, "GB")
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def record_metric(self, metric_type: MetricType, value: float, unit: str = "", 
                     description: str = ""):
        """Record a performance metric"""
        try:
            metric = PerformanceMetric(
                metric_type=metric_type,
                value=value,
                timestamp=time.time(),
                unit=unit,
                description=description
            )
            
            self.metrics_history[metric_type].append(metric)
            
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_type.value}: {e}")
    
    def _check_thresholds(self):
        """Check performance thresholds and trigger alerts"""
        try:
            for metric_type, threshold in self.thresholds.items():
                if metric_type in self.metrics_history and self.metrics_history[metric_type]:
                    latest_metric = self.metrics_history[metric_type][-1]
                    
                    # Check if threshold is exceeded
                    if self._is_threshold_exceeded(latest_metric, threshold):
                        self._trigger_alert(metric_type, latest_metric, threshold)
                        
        except Exception as e:
            self.logger.error(f"Failed to check thresholds: {e}")
    
    def _is_threshold_exceeded(self, metric: PerformanceMetric, threshold: float) -> bool:
        """Check if a metric exceeds its threshold"""
        try:
            # For metrics where lower is better (FPS, etc.)
            if metric.metric_type in [MetricType.FPS, MetricType.THROUGHPUT]:
                return metric.value < threshold
            # For metrics where higher is worse (CPU, memory, latency, etc.)
            else:
                return metric.value > threshold
                
        except Exception as e:
            self.logger.error(f"Failed to check threshold: {e}")
            return False
    
    def _trigger_alert(self, metric_type: MetricType, metric: PerformanceMetric, threshold: float):
        """Trigger performance alert"""
        try:
            alert_message = f"Performance alert: {metric_type.value} = {metric.value}{metric.unit} (threshold: {threshold})"
            self.logger.warning(alert_message)
            
            # Call registered alert callbacks
            if metric_type in self.alert_callbacks:
                for callback in self.alert_callbacks[metric_type]:
                    try:
                        callback(metric_type, metric, threshold)
                    except Exception as e:
                        self.logger.error(f"Alert callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
    
    def register_alert_callback(self, metric_type: MetricType, callback: callable):
        """Register a callback for performance alerts"""
        try:
            if metric_type not in self.alert_callbacks:
                self.alert_callbacks[metric_type] = []
            self.alert_callbacks[metric_type].append(callback)
            
        except Exception as e:
            self.logger.error(f"Failed to register alert callback: {e}")
    
    def get_metric_history(self, metric_type: MetricType, 
                          duration_seconds: Optional[float] = None) -> List[PerformanceMetric]:
        """Get metric history for a specific type"""
        try:
            if metric_type not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[metric_type])
            
            # Filter by duration if specified
            if duration_seconds:
                cutoff_time = time.time() - duration_seconds
                history = [metric for metric in history if metric.timestamp >= cutoff_time]
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get metric history: {e}")
            return []
    
    def get_latest_metric(self, metric_type: MetricType) -> Optional[PerformanceMetric]:
        """Get the latest metric for a specific type"""
        try:
            if metric_type in self.metrics_history and self.metrics_history[metric_type]:
                return self.metrics_history[metric_type][-1]
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest metric: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {}
            
            for metric_type in MetricType:
                history = self.get_metric_history(metric_type, duration_seconds=60)  # Last minute
                
                if history:
                    values = [metric.value for metric in history]
                    summary[metric_type.value] = {
                        'current': values[-1] if values else 0.0,
                        'average': np.mean(values) if values else 0.0,
                        'min': np.min(values) if values else 0.0,
                        'max': np.max(values) if values else 0.0,
                        'count': len(values),
                        'unit': history[0].unit if history else ""
                    }
                else:
                    summary[metric_type.value] = {
                        'current': 0.0,
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0,
                        'unit': ""
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        try:
            summary = self.get_performance_summary()
            report_lines = ["=== Athena Performance Report ==="]
            
            for metric_name, data in summary.items():
                if data['count'] > 0:
                    report_lines.append(
                        f"{metric_name}: {data['current']:.2f}{data['unit']} "
                        f"(avg: {data['average']:.2f}, min: {data['min']:.2f}, max: {data['max']:.2f})"
                    )
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return "Error generating performance report"
    
    def set_threshold(self, metric_type: MetricType, threshold: float):
        """Set a performance threshold"""
        try:
            self.thresholds[metric_type] = threshold
            self.logger.info(f"Set threshold for {metric_type.value}: {threshold}")
            
        except Exception as e:
            self.logger.error(f"Failed to set threshold: {e}")
    
    def get_threshold(self, metric_type: MetricType) -> Optional[float]:
        """Get a performance threshold"""
        return self.thresholds.get(metric_type)
    
    def clear_metric_history(self, metric_type: Optional[MetricType] = None):
        """Clear metric history"""
        try:
            if metric_type:
                if metric_type in self.metrics_history:
                    self.metrics_history[metric_type].clear()
            else:
                for history in self.metrics_history.values():
                    history.clear()
                    
            self.logger.info("Metric history cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear metric history: {e}")
    
    def cleanup(self):
        """Cleanup performance monitor resources"""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Clear all data
            self.clear_metric_history()
            self.alert_callbacks.clear()
            
            self.logger.info("Performance monitor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Performance monitor cleanup failed: {e}")