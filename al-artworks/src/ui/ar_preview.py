"""
AR Preview component for Al-artworks.
Provides holographic preview effects for augmented reality visualization.
"""

from typing import Optional, Tuple
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QBrush, QPen, QLinearGradient, 
    QRadialGradient, QTransform, QPixmap, QImage
)


class ARPreviewWidget(QWidget):
    """Widget for displaying AR/holographic previews of artwork."""
    
    # Signals
    preview_activated = pyqtSignal(bool)
    preview_mode_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_active = False
        self.preview_mode = "hologram"
        self.animation_phase = 0.0
        self.preview_image: Optional[QImage] = None
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            ARPreviewWidget {
                background-color: rgba(0, 0, 0, 200);
                border: 2px solid rgba(0, 255, 255, 100);
                border-radius: 10px;
            }
        """)
        
    def set_preview_image(self, image: QImage):
        """Set the image to preview in AR mode."""
        self.preview_image = image
        self.update()
        
    def toggle_preview(self):
        """Toggle AR preview on/off."""
        self.preview_active = not self.preview_active
        
        if self.preview_active:
            self.animation_timer.start(33)  # ~30fps
        else:
            self.animation_timer.stop()
            
        self.preview_activated.emit(self.preview_active)
        self.update()
        
    def set_preview_mode(self, mode: str):
        """Set the AR preview mode."""
        self.preview_mode = mode
        self.preview_mode_changed.emit(mode)
        self.update()
        
    def update_animation(self):
        """Update animation parameters."""
        self.animation_phase += 0.05
        if self.animation_phase > 2 * np.pi:
            self.animation_phase -= 2 * np.pi
        self.update()
        
    def paintEvent(self, event):
        """Paint the AR preview."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if not self.preview_active:
            self._draw_inactive_state(painter)
        else:
            if self.preview_mode == "hologram":
                self._draw_holographic_preview(painter)
            elif self.preview_mode == "projection":
                self._draw_projection_preview(painter)
            elif self.preview_mode == "floating":
                self._draw_floating_preview(painter)
                
    def _draw_inactive_state(self, painter: QPainter):
        """Draw the inactive state."""
        center = self.rect().center()
        
        # Draw AR icon
        icon_rect = QRectF(center.x() - 40, center.y() - 40, 80, 80)
        
        # Gradient background
        gradient = QRadialGradient(center, 40)
        gradient.setColorAt(0, QColor(0, 255, 255, 50))
        gradient.setColorAt(1, QColor(0, 255, 255, 0))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(icon_rect)
        
        # AR symbol
        painter.setPen(QPen(QColor(0, 255, 255, 200), 3))
        painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, "AR")
        
        # Label
        painter.setPen(QColor(200, 200, 200))
        label_rect = QRectF(0, center.y() + 60, self.width(), 30)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "Click to activate AR preview")
        
    def _draw_holographic_preview(self, painter: QPainter):
        """Draw holographic effect preview."""
        if not self.preview_image:
            return
            
        center = self.rect().center()
        
        # Calculate preview size
        preview_size = min(self.width() - 40, self.height() - 40)
        preview_rect = QRectF(
            center.x() - preview_size / 2,
            center.y() - preview_size / 2,
            preview_size,
            preview_size
        )
        
        # Draw multiple offset layers for holographic effect
        offsets = [
            (0, 0, 1.0, QColor(0, 255, 255, 100)),
            (-2, -2, 0.8, QColor(255, 0, 255, 80)),
            (2, 2, 0.8, QColor(255, 255, 0, 80))
        ]
        
        for dx, dy, scale, color in offsets:
            # Apply transformation
            painter.save()
            painter.translate(center)
            painter.scale(scale, scale)
            painter.translate(-center)
            
            # Draw with color overlay
            offset_rect = preview_rect.translated(dx, dy)
            painter.setOpacity(0.7)
            painter.drawImage(offset_rect, self.preview_image)
            
            # Add color tint
            painter.fillRect(offset_rect, color)
            
            painter.restore()
            
        # Draw main image
        painter.setOpacity(1.0)
        painter.drawImage(preview_rect, self.preview_image)
        
        # Add holographic scanlines
        self._draw_scanlines(painter, preview_rect)
        
        # Add glitch effect
        if np.sin(self.animation_phase * 5) > 0.8:
            self._draw_glitch_effect(painter, preview_rect)
            
    def _draw_projection_preview(self, painter: QPainter):
        """Draw projection-style preview."""
        if not self.preview_image:
            return
            
        center = self.rect().center()
        
        # Create perspective transformation
        source_rect = QRectF(0, 0, self.preview_image.width(), self.preview_image.height())
        
        # Calculate projection trapezoid
        top_width = self.width() * 0.4
        bottom_width = self.width() * 0.8
        height = self.height() * 0.6
        
        top_left = QPointF(center.x() - top_width / 2, center.y() - height / 2)
        top_right = QPointF(center.x() + top_width / 2, center.y() - height / 2)
        bottom_left = QPointF(center.x() - bottom_width / 2, center.y() + height / 2)
        bottom_right = QPointF(center.x() + bottom_width / 2, center.y() + height / 2)
        
        # Draw projection beam
        beam_gradient = QLinearGradient(center.x(), center.y() - height / 2, 
                                      center.x(), center.y() + height / 2)
        beam_gradient.setColorAt(0, QColor(0, 255, 255, 0))
        beam_gradient.setColorAt(0.5, QColor(0, 255, 255, 50))
        beam_gradient.setColorAt(1, QColor(0, 255, 255, 100))
        
        painter.setBrush(QBrush(beam_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        
        beam_path = [top_left, top_right, bottom_right, bottom_left]
        painter.drawPolygon(beam_path)
        
        # Draw projected image with perspective (simplified)
        painter.setOpacity(0.8)
        target_rect = QRectF(top_left, bottom_right)
        painter.drawImage(target_rect, self.preview_image)
        
        # Add projection grid
        self._draw_projection_grid(painter, top_left, top_right, bottom_left, bottom_right)
        
    def _draw_floating_preview(self, painter: QPainter):
        """Draw floating 3D-style preview."""
        if not self.preview_image:
            return
            
        center = self.rect().center()
        
        # Calculate floating animation
        float_offset = np.sin(self.animation_phase) * 10
        rotation = np.sin(self.animation_phase * 0.5) * 5
        
        # Draw shadow
        shadow_rect = QRectF(
            center.x() - 100,
            center.y() + 80 + float_offset / 2,
            200,
            40
        )
        
        shadow_gradient = QRadialGradient(shadow_rect.center(), 100)
        shadow_gradient.setColorAt(0, QColor(0, 0, 0, 100))
        shadow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(shadow_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(shadow_rect)
        
        # Draw floating image
        painter.save()
        painter.translate(center)
        painter.rotate(rotation)
        painter.translate(0, float_offset)
        painter.translate(-center)
        
        preview_rect = QRectF(
            center.x() - 120,
            center.y() - 120,
            240,
            240
        )
        
        # Add glow effect
        glow_rect = preview_rect.adjusted(-20, -20, 20, 20)
        glow_gradient = QRadialGradient(glow_rect.center(), glow_rect.width() / 2)
        glow_gradient.setColorAt(0, QColor(0, 255, 255, 0))
        glow_gradient.setColorAt(0.7, QColor(0, 255, 255, 50))
        glow_gradient.setColorAt(1, QColor(0, 255, 255, 0))
        
        painter.setBrush(QBrush(glow_gradient))
        painter.drawEllipse(glow_rect)
        
        # Draw image
        painter.setOpacity(0.9)
        painter.drawImage(preview_rect, self.preview_image)
        
        # Add floating particles
        self._draw_floating_particles(painter, center, float_offset)
        
        painter.restore()
        
    def _draw_scanlines(self, painter: QPainter, rect: QRectF):
        """Draw holographic scanlines."""
        painter.setPen(QPen(QColor(0, 255, 255, 30), 1))
        
        # Animated scanline position
        scanline_offset = int(self.animation_phase * 20) % 4
        
        y = rect.top() + scanline_offset
        while y < rect.bottom():
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += 4
            
    def _draw_glitch_effect(self, painter: QPainter, rect: QRectF):
        """Draw random glitch effect."""
        glitch_height = 10
        glitch_y = rect.top() + np.random.random() * (rect.height() - glitch_height)
        
        glitch_rect = QRectF(rect.left(), glitch_y, rect.width(), glitch_height)
        
        # RGB shift effect
        painter.fillRect(glitch_rect.translated(2, 0), QColor(255, 0, 0, 100))
        painter.fillRect(glitch_rect.translated(-2, 0), QColor(0, 255, 255, 100))
        
    def _draw_projection_grid(self, painter: QPainter, tl: QPointF, tr: QPointF, 
                            bl: QPointF, br: QPointF):
        """Draw projection grid lines."""
        painter.setPen(QPen(QColor(0, 255, 255, 50), 1))
        
        # Draw horizontal lines
        for i in range(1, 10):
            t = i / 10.0
            left = QPointF(
                tl.x() + (bl.x() - tl.x()) * t,
                tl.y() + (bl.y() - tl.y()) * t
            )
            right = QPointF(
                tr.x() + (br.x() - tr.x()) * t,
                tr.y() + (br.y() - tr.y()) * t
            )
            painter.drawLine(left, right)
            
        # Draw vertical lines
        for i in range(1, 10):
            t = i / 10.0
            top = QPointF(
                tl.x() + (tr.x() - tl.x()) * t,
                tl.y() + (tr.y() - tl.y()) * t
            )
            bottom = QPointF(
                bl.x() + (br.x() - bl.x()) * t,
                bl.y() + (br.y() - bl.y()) * t
            )
            painter.drawLine(top, bottom)
            
    def _draw_floating_particles(self, painter: QPainter, center: QPointF, offset: float):
        """Draw floating particles around the image."""
        painter.setPen(Qt.PenStyle.NoPen)
        
        particle_count = 20
        for i in range(particle_count):
            angle = (i / particle_count) * 2 * np.pi + self.animation_phase
            radius = 150 + np.sin(angle * 3 + self.animation_phase) * 30
            
            x = center.x() + np.cos(angle) * radius
            y = center.y() + np.sin(angle) * radius + offset
            
            size = 3 + np.sin(angle * 5 + self.animation_phase * 2) * 2
            
            particle_color = QColor(0, 255, 255, int(100 + np.sin(angle * 7) * 50))
            painter.setBrush(QBrush(particle_color))
            painter.drawEllipse(QPointF(x, y), size, size)
            
    def mousePressEvent(self, event):
        """Handle mouse press to toggle preview."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_preview()