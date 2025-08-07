"""
AR/3D Overlay System - Futuristic HUD Interface
Provides holographic-style overlay with 3D effects and visual enhancements
"""

import sys
import math
import time
from typing import List, Tuple, Optional
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, 
                              QGraphicsItem, QGraphicsTextItem, QGraphicsEllipseItem,
                              QGraphicsRectItem, QGraphicsPolygonItem, QWidget,
                              QVBoxLayout, QHBoxLayout, QLabel, QPushButton)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PyQt6.QtGui import (QPen, QBrush, QColor, QFont, QPainter, QPainterPath,
                         QLinearGradient, QRadialGradient, QFontMetrics)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from OpenGL import GL
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

class HolographicText(QGraphicsTextItem):
    """Holographic-style text with glow effects"""
    
    def __init__(self, text: str, parent: QGraphicsItem = None):
        super().__init__(text, parent)
        self.setDefaultTextColor(QColor(0, 255, 255))  # Cyan
        self.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        # Add glow effect
        self.glow_effect = self.graphicsEffect()
        if self.glow_effect:
            self.glow_effect.setColor(QColor(0, 255, 255, 50))
            self.glow_effect.setBlurRadius(10)
    
    def paint(self, painter: QPainter, option, widget):
        # Create glow effect
        painter.setPen(QPen(QColor(0, 255, 255, 100), 3))
        painter.setBrush(QBrush(QColor(0, 255, 255, 20)))
        
        # Draw glow
        rect = self.boundingRect()
        painter.drawRoundedRect(rect, 5, 5)
        
        # Draw text
        super().paint(painter, option, widget)

class NeuralNode(QGraphicsEllipseItem):
    """Neural network node visualization"""
    
    def __init__(self, x: float, y: float, radius: float = 10, parent: QGraphicsItem = None):
        super().__init__(-radius, -radius, radius * 2, radius * 2, parent)
        self.setPos(x, y)
        
        # Setup appearance
        self.setBrush(QBrush(QColor(0, 255, 255, 150)))
        self.setPen(QPen(QColor(0, 255, 255), 2))
        
        # Animation properties
        self.pulse_animation = QPropertyAnimation(self, b"scale")
        self.pulse_animation.setDuration(2000)
        self.pulse_animation.setStartValue(1.0)
        self.pulse_animation.setEndValue(1.3)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop
        
        # Start pulsing
        self.pulse_animation.start()
    
    def setScale(self, scale: float):
        """Override setScale to handle animation"""
        self.setTransform(self.transform().scale(scale, scale))

class DataFlowLine(QGraphicsItem):
    """Animated data flow line between nodes"""
    
    def __init__(self, start_pos: QPointF, end_pos: QPointF, parent: QGraphicsItem = None):
        super().__init__(parent)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.animation_progress = 0.0
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS
        
        # Setup appearance
        self.setPen(QPen(QColor(0, 255, 255, 200), 2))
    
    def update_animation(self):
        """Update animation progress"""
        self.animation_progress += 0.05
        if self.animation_progress > 1.0:
            self.animation_progress = 0.0
        self.update()
    
    def boundingRect(self) -> QRectF:
        """Return bounding rectangle"""
        return QRectF(self.start_pos, self.end_pos)
    
    def paint(self, painter: QPainter, option, widget):
        """Paint the animated data flow line"""
        # Calculate current position based on animation
        current_x = self.start_pos.x() + (self.end_pos.x() - self.start_pos.x()) * self.animation_progress
        current_y = self.start_pos.y() + (self.end_pos.y() - self.start_pos.y()) * self.animation_progress
        
        # Draw the line
        painter.drawLine(self.start_pos, QPointF(current_x, current_y))
        
        # Draw moving dot
        painter.setBrush(QBrush(QColor(0, 255, 255)))
        painter.drawEllipse(QPointF(current_x, current_y), 3, 3)

class AROverlay(QGraphicsView):
    """Main AR overlay window with 3D effects"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 ATHENA - AI Assistant Overlay")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTop | Qt.WindowType.FramelessWindowHint)
        
        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Setup OpenGL viewport if available
        if OPENGL_AVAILABLE:
            self.setViewport(QOpenGLWidget())
        
        # Setup appearance
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0, 180)))  # Semi-transparent black
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Initialize components
        self.neural_nodes = []
        self.data_flows = []
        self.text_elements = []
        
        # Setup UI
        self.setup_ui()
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        
        # Status
        self.is_active = False
    
    def start(self):
        """Start the AR overlay"""
        self.is_active = True
        self.animation_timer.start(16)  # ~60 FPS
        self.showFullScreen()
        print("[AR Overlay] Started")
    
    def stop(self):
        """Stop the AR overlay"""
        self.is_active = False
        self.animation_timer.stop()
        self.hide()
        print("[AR Overlay] Stopped")
    
    def setup_ui(self):
        """Setup the overlay UI elements"""
        # Add main title
        title = HolographicText("🧠 ATHENA AI ASSISTANT")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setPos(50, 50)
        self.scene.addItem(title)
        self.text_elements.append(title)
        
        # Add status text
        status = HolographicText("System Online - Ready for Voice Input")
        status.setFont(QFont("Arial", 14))
        status.setPos(50, 100)
        self.scene.addItem(status)
        self.text_elements.append(status)
        
        # Create neural network visualization
        self.create_neural_network()
        
        # Add tool indicators
        self.create_tool_indicators()
        
        # Add data visualization
        self.create_data_visualization()
    
    def create_neural_network(self):
        """Create animated neural network visualization"""
        # Create nodes in a grid pattern
        for i in range(5):
            for j in range(5):
                x = 200 + i * 80
                y = 200 + j * 60
                node = NeuralNode(x, y, 8)
                self.scene.addItem(node)
                self.neural_nodes.append(node)
        
        # Create data flow lines
        for i in range(len(self.neural_nodes) - 1):
            start_node = self.neural_nodes[i]
            end_node = self.neural_nodes[i + 1]
            
            flow_line = DataFlowLine(start_node.pos(), end_node.pos())
            self.scene.addItem(flow_line)
            self.data_flows.append(flow_line)
    
    def create_tool_indicators(self):
        """Create visual indicators for available tools"""
        tools = [
            ("🎤 Voice", 50, 300),
            ("💻 Code", 200, 300),
            ("🌐 Web", 350, 300),
            ("🎨 Image", 500, 300),
            ("📁 Files", 650, 300)
        ]
        
        for tool_name, x, y in tools:
            tool_text = HolographicText(tool_name)
            tool_text.setFont(QFont("Arial", 12))
            tool_text.setPos(x, y)
            self.scene.addItem(tool_text)
            self.text_elements.append(tool_text)
    
    def create_data_visualization(self):
        """Create data visualization elements"""
        # Add data bars
        for i in range(8):
            x = 50 + i * 100
            y = 400
            height = 20 + (i * 10) % 60
            
            bar = QGraphicsRectItem(x, y - height, 80, height)
            bar.setBrush(QBrush(QColor(0, 255, 255, 100)))
            bar.setPen(QPen(QColor(0, 255, 255), 1))
            self.scene.addItem(bar)
    
    def update_animations(self):
        """Update all animations with enhanced 3D effects"""
        current_time = time.time()
        
        # Update text glow effects with pulsing
        for text in self.text_elements:
            if isinstance(text, HolographicText):
                # Enhanced pulsing effect
                pulse = math.sin(current_time * 2) * 0.5 + 0.5
                alpha = 150 + int(50 * pulse)
                new_color = QColor(0, 255, 255, alpha)
                text.setDefaultTextColor(new_color)
                
                # Add subtle movement
                offset = math.sin(current_time + text.pos().x() * 0.01) * 2
                text.setPos(text.pos().x(), text.pos().y() + offset)
        
        # Update neural network nodes with enhanced effects
        for i, node in enumerate(self.neural_nodes):
            if isinstance(node, NeuralNode):
                # Enhanced pulsing with individual timing
                pulse = math.sin(current_time * 3 + i * 0.5) * 0.5 + 0.5
                node.setScale(1.0 + pulse * 0.3)
                
                # Color cycling effect
                hue = (current_time * 30 + i * 30) % 360
                color = QColor.fromHsv(int(hue), 255, 255, 200)
                node.setBrush(QBrush(color))
        
        # Update data flow lines with enhanced animation
        for i, flow in enumerate(self.data_flows):
            if isinstance(flow, DataFlowLine):
                # Enhanced flow speed
                flow.animation_progress += 0.02 + (i * 0.01)
                if flow.animation_progress > 1.0:
                    flow.animation_progress = 0.0
                
                # Add particle effects
                self._add_particle_effects(flow, current_time)
        
        # Add holographic distortion effects
        self._add_distortion_effects(current_time)
    
    def _add_particle_effects(self, flow_line, current_time):
        """Add particle effects to data flow lines"""
        # This would add floating particles along the data flow
        pass
    
    def _add_distortion_effects(self, current_time):
        """Add holographic distortion effects"""
        # This would add wave-like distortion effects
        pass
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_visibility()
    
    def toggle_visibility(self):
        """Toggle overlay visibility"""
        if self.isVisible():
            self.hide()
        else:
            self.show()
    
    def add_system_message(self, message: str):
        """Add a system message to the overlay"""
        text = HolographicText(f"System: {message}")
        text.setFont(QFont("Arial", 10))
        text.setPos(50, 500 + len(self.text_elements) * 25)
        self.scene.addItem(text)
        self.text_elements.append(text)
    
    def add_voice_indicator(self, is_listening: bool):
        """Add voice activity indicator"""
        if is_listening:
            indicator = HolographicText("🎤 LISTENING...")
            indicator.setDefaultTextColor(QColor(255, 255, 0))  # Yellow
        else:
            indicator = HolographicText("🎤 Ready")
            indicator.setDefaultTextColor(QColor(0, 255, 0))  # Green
        
        indicator.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        indicator.setPos(50, 150)
        self.scene.addItem(indicator)
        self.text_elements.append(indicator)

class OverlayController:
    """Controller for managing the AR overlay"""
    
    def __init__(self):
        self.overlay = None
        self.is_active = False
    
    def start_overlay(self):
        """Start the AR overlay"""
        if not self.is_active:
            self.overlay = AROverlay()
            self.is_active = True
            return self.overlay
        return None
    
    def stop_overlay(self):
        """Stop the AR overlay"""
        if self.is_active and self.overlay:
            self.overlay.close()
            self.overlay = None
            self.is_active = False
    
    def add_message(self, message: str):
        """Add a message to the overlay"""
        if self.overlay:
            self.overlay.add_system_message(message)
    
    def set_voice_status(self, is_listening: bool):
        """Set voice activity status"""
        if self.overlay:
            self.overlay.add_voice_indicator(is_listening)

# Global overlay controller
overlay_controller = OverlayController()

def start_ar_overlay():
    """Start the AR overlay"""
    return overlay_controller.start_overlay()

def stop_ar_overlay():
    """Stop the AR overlay"""
    overlay_controller.stop_overlay()

def add_overlay_message(message: str):
    """Add a message to the overlay"""
    overlay_controller.add_message(message)

def set_voice_status(is_listening: bool):
    """Set voice activity status in overlay"""
    overlay_controller.set_voice_status(is_listening)

if __name__ == "__main__":
    # Test AR overlay
    app = QApplication(sys.argv)
    
    print("Starting AR Overlay...")
    print("Press ESC to exit")
    print("Press SPACE to toggle visibility")
    
    overlay = start_ar_overlay()
    
    # Add some test messages
    if overlay:
        overlay.add_system_message("System initialized")
        overlay.add_system_message("Voice recognition active")
        overlay.add_system_message("Neural network online")
    
    sys.exit(app.exec())