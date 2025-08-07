import asyncio
import logging
import time
import psutil
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

log = logging.getLogger("ProfilerDashboard")

@dataclass
class SystemStats:
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_used_gb: Optional[float]
    gpu_memory_total_gb: Optional[float]
    gpu_temperature: Optional[float]
    timestamp: float

class ProfilerDashboard(QMainWindow):
    """Real-time system monitoring dashboard"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatGPT+ Profiler Dashboard")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize UI
        self.init_ui()
        
        # Monitoring state
        self.running = False
        self.stats_history = []
        self.max_history = 100
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
        
        log.info("Profiler Dashboard initialized")

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("ChatGPT+ System Monitor")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # CPU Usage
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_bar)
        
        # Memory Usage
        self.memory_label = QLabel("Memory Usage: 0%")
        self.memory_bar = QProgressBar()
        self.memory_bar.setMaximum(100)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.memory_bar)
        
        # GPU Usage (if available)
        self.gpu_label = QLabel("GPU Usage: N/A")
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setMaximum(100)
        layout.addWidget(self.gpu_label)
        layout.addWidget(self.gpu_bar)
        
        # GPU Memory
        self.gpu_memory_label = QLabel("GPU Memory: N/A")
        self.gpu_memory_bar = QProgressBar()
        self.gpu_memory_bar.setMaximum(100)
        layout.addWidget(self.gpu_memory_label)
        layout.addWidget(self.gpu_memory_bar)
        
        # Disk Usage
        self.disk_label = QLabel("Disk Usage: 0%")
        self.disk_bar = QProgressBar()
        self.disk_bar.setMaximum(100)
        layout.addWidget(self.disk_label)
        layout.addWidget(self.disk_bar)
        
        # Process Table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(4)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %"])
        layout.addWidget(self.process_table)
        
        # Status
        self.status_label = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label)

    def update_display(self):
        """Update the display with current system stats"""
        try:
            stats = self.get_system_stats()
            self.update_stats_display(stats)
            self.update_process_table()
            self.status_label.setText(f"Status: Updated at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            log.error(f"Error updating display: {e}")
            self.status_label.setText(f"Status: Error - {str(e)}")

    def get_system_stats(self) -> SystemStats:
        """Get current system statistics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/') if hasattr(psutil, 'disk_usage') else None
        
        # GPU stats (if available)
        gpu_utilization = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None
        gpu_temperature = None
        
        if torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization()
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Try to get temperature via nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_temperature = float(result.stdout.strip())
                except:
                    pass
                    
            except Exception as e:
                log.warning(f"Failed to get GPU stats: {e}")
        
        stats = SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent if disk else 0.0,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_temperature=gpu_temperature,
            timestamp=time.time()
        )
        
        # Store in history
        self.stats_history.append(stats)
        if len(self.stats_history) > self.max_history:
            self.stats_history.pop(0)
        
        return stats

    def update_stats_display(self, stats: SystemStats):
        """Update the display with system statistics"""
        # CPU
        self.cpu_label.setText(f"CPU Usage: {stats.cpu_percent:.1f}%")
        self.cpu_bar.setValue(int(stats.cpu_percent))
        
        # Memory
        self.memory_label.setText(f"Memory Usage: {stats.memory_percent:.1f}% ({stats.memory_used_gb:.1f}GB / {stats.memory_total_gb:.1f}GB)")
        self.memory_bar.setValue(int(stats.memory_percent))
        
        # GPU
        if stats.gpu_utilization is not None:
            self.gpu_label.setText(f"GPU Usage: {stats.gpu_utilization:.1f}%")
            self.gpu_bar.setValue(int(stats.gpu_utilization))
        else:
            self.gpu_label.setText("GPU Usage: N/A")
            self.gpu_bar.setValue(0)
        
        # GPU Memory
        if stats.gpu_memory_used_gb is not None and stats.gpu_memory_total_gb is not None:
            gpu_memory_percent = (stats.gpu_memory_used_gb / stats.gpu_memory_total_gb) * 100
            self.gpu_memory_label.setText(f"GPU Memory: {gpu_memory_percent:.1f}% ({stats.gpu_memory_used_gb:.1f}GB / {stats.gpu_memory_total_gb:.1f}GB)")
            self.gpu_memory_bar.setValue(int(gpu_memory_percent))
        else:
            self.gpu_memory_label.setText("GPU Memory: N/A")
            self.gpu_memory_bar.setValue(0)
        
        # Disk
        self.disk_label.setText(f"Disk Usage: {stats.disk_usage_percent:.1f}%")
        self.disk_bar.setValue(int(stats.disk_usage_percent))

    def update_process_table(self):
        """Update the process table with top processes"""
        try:
            # Get top processes by CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage and take top 10
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            top_processes = processes[:10]
            
            # Update table
            self.process_table.setRowCount(len(top_processes))
            for i, proc in enumerate(top_processes):
                self.process_table.setItem(i, 0, QTableWidgetItem(str(proc['pid'])))
                self.process_table.setItem(i, 1, QTableWidgetItem(proc['name']))
                self.process_table.setItem(i, 2, QTableWidgetItem(f"{proc['cpu_percent']:.1f}%"))
                self.process_table.setItem(i, 3, QTableWidgetItem(f"{proc['memory_percent']:.1f}%"))
                
        except Exception as e:
            log.error(f"Error updating process table: {e}")

    def get_stats_history(self, limit: int = 50) -> List[SystemStats]:
        """Get recent system stats history"""
        return self.stats_history[-limit:]

    def get_average_stats(self, window: int = 10) -> Dict[str, float]:
        """Get average system stats over a window"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-window:]
        
        return {
            "avg_cpu": sum(s.cpu_percent for s in recent_stats) / len(recent_stats),
            "avg_memory": sum(s.memory_percent for s in recent_stats) / len(recent_stats),
            "avg_gpu": sum(s.gpu_utilization for s in recent_stats if s.gpu_utilization) / max(len([s for s in recent_stats if s.gpu_utilization]), 1),
            "avg_disk": sum(s.disk_usage_percent for s in recent_stats) / len(recent_stats)
        }

    def start(self):
        """Start the profiler dashboard"""
        self.running = True
        self.show()
        log.info("Profiler Dashboard started")

    def stop(self):
        """Stop the profiler dashboard"""
        self.running = False
        self.close()
        log.info("Profiler Dashboard stopped")

class ProfilerDashboard:
    """Wrapper class for the profiler dashboard"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.app = None
        self.window = None
        self.running = False
        
        if self.enabled:
            log.info("Profiler Dashboard initialized")
        else:
            log.info("Profiler Dashboard initialized but disabled")

    def start(self):
        """Start the profiler dashboard"""
        if not self.enabled:
            log.info("Profiler Dashboard is disabled, not starting")
            return
        
        try:
            # Create QApplication if not exists
            if not QApplication.instance():
                self.app = QApplication([])
            else:
                self.app = QApplication.instance()
            
            # Create dashboard window
            self.window = ProfilerDashboard()
            self.window.start()
            
            self.running = True
            log.info("Profiler Dashboard started successfully")
            
        except Exception as e:
            log.error(f"Failed to start Profiler Dashboard: {e}")

    def stop(self):
        """Stop the profiler dashboard"""
        if self.enabled and self.running:
            if self.window:
                self.window.stop()
            self.running = False
            log.info("Profiler Dashboard stopped")

    def get_status(self) -> dict:
        """Get dashboard status"""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "window_created": self.window is not None
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    dashboard = ProfilerDashboard(enabled=True)
    
    try:
        dashboard.start()
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Stopping Profiler Dashboard...")
        dashboard.stop()