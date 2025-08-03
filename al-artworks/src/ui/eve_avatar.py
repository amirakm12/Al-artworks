"""
Eve avatar widget for Al-artworks.
Displays Eve's 3D avatar using NeRF rendering with voice interaction.
"""

from typing import Optional
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QPointF
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from OpenGL import GL
except ImportError:
    GL = None


class EveAvatarRenderer(QOpenGLWidget):
    """OpenGL widget for rendering Eve's 3D avatar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation = 0.0
        self.scale = 1.0
        self.glow_intensity = 0.5
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(33)  # ~30fps
        
    def initializeGL(self):
        """Initialize OpenGL context."""
        if GL:
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            
    def resizeGL(self, width: int, height: int):
        """Handle widget resize."""
        if GL:
            GL.glViewport(0, 0, width, height)
            
    def paintGL(self):
        """Render Eve's avatar."""
        if not GL:
            # Fallback to 2D rendering
            self.render_2d_avatar()
            return
            
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        # Placeholder for NeRF rendering
        # In production, this would render the actual 3D model
        self.render_placeholder_avatar()
        
    def render_placeholder_avatar(self):
        """Render a placeholder avatar effect."""
        # This is a placeholder for the actual NeRF rendering
        # In production, integrate with the NeRF model
        pass
        
    def render_2d_avatar(self):
        """Fallback 2D avatar rendering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = min(self.width(), self.height()) / 3
        
        # Create celestial glow effect
        for i in range(3):
            glow_radius = radius * (1.5 - i * 0.2)
            gradient = QRadialGradient(center, glow_radius)
            
            alpha = int(50 * self.glow_intensity * (1 - i * 0.3))
            gradient.setColorAt(0, QColor(255, 200, 255, alpha))
            gradient.setColorAt(0.5, QColor(200, 150, 255, alpha // 2))
            gradient.setColorAt(1, QColor(150, 100, 200, 0))
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, glow_radius, glow_radius)
        
        # Draw avatar circle
        painter.setBrush(QBrush(QColor(100, 50, 150, 200)))
        painter.setPen(QPen(QColor(255, 200, 255), 3))
        painter.drawEllipse(center, radius, radius)
        
        # Draw rotating accent
        painter.save()
        painter.translate(center)
        painter.rotate(self.rotation)
        
        accent_pen = QPen(QColor(255, 200, 255, 150), 2)
        painter.setPen(accent_pen)
        painter.drawArc(-radius, -radius, radius * 2, radius * 2, 0, 60 * 16)
        painter.drawArc(-radius, -radius, radius * 2, radius * 2, 120 * 16, 60 * 16)
        painter.drawArc(-radius, -radius, radius * 2, radius * 2, 240 * 16, 60 * 16)
        
        painter.restore()
        
    def update_animation(self):
        """Update animation parameters."""
        self.rotation += 1.0
        if self.rotation >= 360:
            self.rotation = 0
            
        # Pulsing glow effect
        self.glow_intensity = 0.5 + 0.3 * np.sin(self.rotation * np.pi / 180)
        
        self.update()
        
    def paintEvent(self, event):
        """Override paint event for 2D fallback."""
        if not GL:
            self.render_2d_avatar()
        else:
            super().paintEvent(event)


class EveAvatarWidget(QWidget):
    """Main Eve avatar widget with voice interaction controls."""
    
    # Signals
    voice_command = pyqtSignal(str)
    eve_speaking = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_active = False
        self.is_listening = False
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Avatar display
        self.avatar_renderer = EveAvatarRenderer()
        self.avatar_renderer.setMinimumSize(200, 200)
        layout.addWidget(self.avatar_renderer)
        
        # Eve name label
        self.name_label = QLabel("EVE")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 200, 255, 255);
                font-size: 24px;
                font-weight: bold;
                font-family: 'Arial', sans-serif;
                letter-spacing: 3px;
            }
        """)
        layout.addWidget(self.name_label)
        
        # Status label
        self.status_label = QLabel("Celestial Creative Goddess")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(200, 150, 255, 200);
                font-size: 12px;
                font-style: italic;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Voice controls
        voice_layout = QHBoxLayout()
        voice_layout.setSpacing(10)
        
        self.listen_button = QPushButton("🎤 Listen")
        self.listen_button.clicked.connect(self.toggle_listening)
        self.listen_button.setStyleSheet(self._get_voice_button_style())
        voice_layout.addWidget(self.listen_button)
        
        self.speak_button = QPushButton("🔊 Speak")
        self.speak_button.clicked.connect(self.test_speech)
        self.speak_button.setStyleSheet(self._get_voice_button_style())
        voice_layout.addWidget(self.speak_button)
        
        layout.addLayout(voice_layout)
        
        # Activity indicator
        self.activity_indicator = QWidget()
        self.activity_indicator.setFixedHeight(4)
        self.activity_indicator.setStyleSheet("""
            background-color: rgba(255, 200, 255, 100);
            border-radius: 2px;
        """)
        layout.addWidget(self.activity_indicator)
        
        # Breathing animation for activity
        self.breathing_animation = QPropertyAnimation(self.activity_indicator, b"geometry")
        self.breathing_animation.setDuration(2000)
        self.breathing_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.breathing_animation.setLoopCount(-1)  # Infinite loop
        
        self.setLayout(layout)
        
    def _get_voice_button_style(self) -> str:
        """Get style for voice control buttons."""
        return """
        QPushButton {
            background-color: rgba(100, 50, 150, 150);
            color: rgba(255, 200, 255, 255);
            border: 2px solid rgba(150, 100, 200, 200);
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: rgba(150, 100, 200, 200);
            border-color: rgba(255, 200, 255, 255);
        }
        QPushButton:pressed {
            background-color: rgba(200, 150, 255, 255);
        }
        QPushButton:checked {
            background-color: rgba(255, 100, 100, 200);
            border-color: rgba(255, 150, 150, 255);
        }
        """
        
    def activate(self):
        """Activate Eve with greeting animation."""
        self.is_active = True
        self.status_label.setText("✨ Awakening... ✨")
        
        # Start breathing animation
        self._start_breathing_animation()
        
        # Simulate greeting after animation
        QTimer.singleShot(2000, self._greet_user)
        
    def _greet_user(self):
        """Greet the user."""
        self.status_label.setText("Ready to create celestial art")
        self.eve_speaking.emit(True)
        
        # Simulate speaking
        QTimer.singleShot(3000, lambda: self.eve_speaking.emit(False))
        
    def toggle_listening(self):
        """Toggle voice listening mode."""
        self.is_listening = not self.is_listening
        
        if self.is_listening:
            self.listen_button.setText("🔴 Listening...")
            self.listen_button.setChecked(True)
            self.status_label.setText("Listening for commands...")
            self._start_listening()
        else:
            self.listen_button.setText("🎤 Listen")
            self.listen_button.setChecked(False)
            self.status_label.setText("Ready to create celestial art")
            self._stop_listening()
            
    def _start_listening(self):
        """Start listening for voice commands."""
        # In production, this would integrate with Whisper ASR
        # For now, simulate with a timer
        self.voice_timer = QTimer()
        self.voice_timer.timeout.connect(self._simulate_voice_command)
        self.voice_timer.start(5000)  # Simulate command every 5 seconds
        
    def _stop_listening(self):
        """Stop listening for voice commands."""
        if hasattr(self, 'voice_timer'):
            self.voice_timer.stop()
            
    def _simulate_voice_command(self):
        """Simulate receiving a voice command."""
        commands = [
            "Apply cosmic filter",
            "Create new layer",
            "Enhance the image",
            "Add celestial glow",
            "Vectorize this artwork"
        ]
        
        import random
        command = random.choice(commands)
        self.voice_command.emit(command)
        self.status_label.setText(f"Heard: {command}")
        
    def test_speech(self):
        """Test Eve's speech synthesis."""
        self.eve_speaking.emit(True)
        self.status_label.setText("Speaking...")
        
        # In production, this would use Bark TTS
        QTimer.singleShot(2000, self._finish_speaking)
        
    def _finish_speaking(self):
        """Finish speaking."""
        self.eve_speaking.emit(False)
        self.status_label.setText("Ready to create celestial art")
        
    def _start_breathing_animation(self):
        """Start the breathing animation for the activity indicator."""
        width = self.activity_indicator.width()
        height = self.activity_indicator.height()
        
        # Animate width to create breathing effect
        self.breathing_animation.setStartValue(QRect(0, 0, width, height))
        self.breathing_animation.setEndValue(QRect(-10, 0, width + 20, height))
        self.breathing_animation.start()
        
    def set_mood(self, mood: str):
        """Set Eve's mood/state."""
        moods = {
            "happy": "✨ Inspired and Creative ✨",
            "thinking": "🌟 Processing cosmic ideas... 🌟",
            "working": "🎨 Crafting celestial art... 🎨",
            "excited": "💫 Discovering new possibilities! 💫"
        }
        
        self.status_label.setText(moods.get(mood, "Ready to create celestial art"))