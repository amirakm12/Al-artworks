import logging
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
import sys
import time

log = logging.getLogger("AROverlay")

class ARVisualizationWidget(QWidget):
    """Custom widget for AR visualizations"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.1);")
        
        # Visualization data
        self.ai_status = "Idle"
        self.gpu_utilization = 0.0
        self.memory_usage = 0.0
        self.active_plugins = []
        self.voice_active = False
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_visualization)
        self.animation_timer.start(100)  # 10 FPS
        
        log.info("AR Visualization Widget initialized")

    def paintEvent(self, event):
        """Custom painting for AR visualizations"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 50))
        
        # Draw AI status indicator
        self._draw_ai_status(painter)
        
        # Draw GPU utilization
        self._draw_gpu_utilization(painter)
        
        # Draw memory usage
        self._draw_memory_usage(painter)
        
        # Draw voice activity indicator
        self._draw_voice_indicator(painter)
        
        # Draw plugin status
        self._draw_plugin_status(painter)

    def _draw_ai_status(self, painter: QPainter):
        """Draw AI status indicator"""
        painter.setPen(QPen(QColor(0, 255, 0) if self.ai_status == "Active" else QColor(255, 255, 0), 2))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(10, 30, f"AI: {self.ai_status}")

    def _draw_gpu_utilization(self, painter: QPainter):
        """Draw GPU utilization bar"""
        x, y = 10, 50
        width, height = 200, 20
        
        # Background
        painter.fillRect(x, y, width, height, QColor(50, 50, 50))
        
        # Utilization bar
        utilization_width = int(width * self.gpu_utilization)
        color = QColor(0, 255, 0) if self.gpu_utilization < 0.7 else QColor(255, 165, 0) if self.gpu_utilization < 0.9 else QColor(255, 0, 0)
        painter.fillRect(x, y, utilization_width, height, color)
        
        # Text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(x + width + 10, y + 15, f"GPU: {self.gpu_utilization:.1%}")

    def _draw_memory_usage(self, painter: QPainter):
        """Draw memory usage bar"""
        x, y = 10, 80
        width, height = 200, 20
        
        # Background
        painter.fillRect(x, y, width, height, QColor(50, 50, 50))
        
        # Memory bar
        memory_width = int(width * self.memory_usage)
        color = QColor(0, 255, 0) if self.memory_usage < 0.7 else QColor(255, 165, 0) if self.memory_usage < 0.9 else QColor(255, 0, 0)
        painter.fillRect(x, y, memory_width, height, color)
        
        # Text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(x + width + 10, y + 15, f"RAM: {self.memory_usage:.1%}")

    def _draw_voice_indicator(self, painter: QPainter):
        """Draw voice activity indicator"""
        x, y = 10, 110
        size = 20
        
        if self.voice_active:
            # Animated voice indicator
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.drawEllipse(x, y, size, size)
            
            # Animated rings
            for i in range(3):
                alpha = int(255 * (1 - i * 0.3))
                painter.setPen(QPen(QColor(0, 255, 0, alpha), 1))
                painter.drawEllipse(x - i * 5, y - i * 5, size + i * 10, size + i * 10)
        else:
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.drawEllipse(x, y, size, size)
        
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(x + size + 10, y + 15, "Voice")

    def _draw_plugin_status(self, painter: QPainter):
        """Draw plugin status"""
        x, y = 10, 140
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(x, y, f"Plugins: {len(self.active_plugins)} active")

    def update_visualization(self):
        """Update visualization data"""
        # This would be called by external components to update the display
        self.update()

    def set_ai_status(self, status: str):
        """Set AI status"""
        self.ai_status = status

    def set_gpu_utilization(self, utilization: float):
        """Set GPU utilization (0.0 to 1.0)"""
        self.gpu_utilization = max(0.0, min(1.0, utilization))

    def set_memory_usage(self, usage: float):
        """Set memory usage (0.0 to 1.0)"""
        self.memory_usage = max(0.0, min(1.0, usage))

    def set_voice_active(self, active: bool):
        """Set voice activity state"""
        self.voice_active = active

    def set_active_plugins(self, plugins: list):
        """Set active plugins list"""
        self.active_plugins = plugins

class AROverlay:
    """AR Overlay manager"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.app = None
        self.widget = None
        self.running = False
        
        if self.enabled:
            log.info("AR Overlay initialized")
        else:
            log.info("AR Overlay initialized but disabled")

    def start(self):
        """Start the AR overlay"""
        if not self.enabled:
            log.info("AR Overlay is disabled, not starting")
            return
        
        try:
            # Create QApplication if not exists
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            # Create visualization widget
            self.widget = ARVisualizationWidget()
            self.widget.setGeometry(100, 100, 400, 200)
            self.widget.show()
            
            self.running = True
            log.info("AR Overlay started successfully")
            
            # Start event loop in separate thread
            import threading
            self.qt_thread = threading.Thread(target=self._qt_event_loop, daemon=True)
            self.qt_thread.start()
            
        except Exception as e:
            log.error(f"Failed to start AR Overlay: {e}")

    def _qt_event_loop(self):
        """Qt event loop in separate thread"""
        try:
            self.app.exec()
        except Exception as e:
            log.error(f"Qt event loop error: {e}")

    def stop(self):
        """Stop the AR overlay"""
        if self.enabled and self.running:
            if self.widget:
                self.widget.close()
            if self.app:
                self.app.quit()
            self.running = False
            log.info("AR Overlay stopped")

    def update_stats(self, gpu_util: float = 0.0, memory_usage: float = 0.0, 
                   ai_status: str = "Idle", voice_active: bool = False, 
                   active_plugins: list = None):
        """Update overlay statistics"""
        if self.enabled and self.widget:
            self.widget.set_gpu_utilization(gpu_util)
            self.widget.set_memory_usage(memory_usage)
            self.widget.set_ai_status(ai_status)
            self.widget.set_voice_active(voice_active)
            if active_plugins:
                self.widget.set_active_plugins(active_plugins)

    def get_status(self) -> dict:
        """Get overlay status"""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "widget_created": self.widget is not None
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    overlay = AROverlay(enabled=True)
    
    try:
        overlay.start()
        
        # Simulate updates
        import time
        for i in range(10):
            overlay.update_stats(
                gpu_util=i * 0.1,
                memory_usage=0.5 + i * 0.05,
                ai_status="Active" if i % 2 == 0 else "Idle",
                voice_active=i % 3 == 0,
                active_plugins=[f"plugin_{i}"]
            )
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Stopping AR Overlay...")
        overlay.stop()