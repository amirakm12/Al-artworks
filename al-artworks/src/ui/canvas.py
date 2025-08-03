"""
Image canvas widget with zoom/pan capabilities for Al-artworks.
Supports real-time rendering at 60fps with GPU acceleration.
"""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPainter, QPixmap, QImage, QWheelEvent, QMouseEvent,
    QKeyEvent, QPen, QBrush, QColor, QTransform
)


class ImageLayer:
    """Represents a single image layer."""
    
    def __init__(self, name: str, image: QImage, opacity: float = 1.0):
        self.name = name
        self.image = image
        self.opacity = opacity
        self.visible = True
        self.blend_mode = "normal"
        self.locked = False
        
    def get_pixmap(self) -> QPixmap:
        """Convert image to pixmap with opacity."""
        pixmap = QPixmap.fromImage(self.image)
        if self.opacity < 1.0:
            # Apply opacity
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.DestinationIn)
            painter.fillRect(pixmap.rect(), QColor(0, 0, 0, int(255 * self.opacity)))
            painter.end()
        return pixmap


class ImageCanvas(QGraphicsView):
    """Main image editing canvas with zoom/pan support."""
    
    # Signals
    zoom_changed = pyqtSignal(float)
    selection_changed = pyqtSignal(QRectF)
    layer_changed = pyqtSignal(int)
    canvas_modified = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Canvas properties
        self.canvas_size = (500, 500)
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.2
        
        # Interaction states
        self.is_panning = False
        self.last_mouse_pos = QPointF()
        self.selection_rect = None
        self.is_selecting = False
        
        # Layers
        self.layers: List[ImageLayer] = []
        self.active_layer_index = -1
        
        # Performance optimization
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_canvas)
        self.render_timer.start(16)  # ~60fps
        self.needs_render = True
        
        # Initialize canvas
        self._init_canvas()
        self._setup_view()
        
    def _init_canvas(self):
        """Initialize the graphics scene and canvas."""
        # Create scene
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, *self.canvas_size)
        self.setScene(self.scene)
        
        # Create background
        self.scene.setBackgroundBrush(QBrush(QColor(50, 50, 50)))
        
        # Create canvas item
        self.canvas_pixmap = QPixmap(*self.canvas_size)
        self.canvas_pixmap.fill(Qt.GlobalColor.white)
        self.canvas_item = QGraphicsPixmapItem(self.canvas_pixmap)
        self.scene.addItem(self.canvas_item)
        
        # Create selection rectangle
        self.selection_item = self.scene.addRect(
            QRectF(), 
            QPen(QColor(0, 120, 255), 2, Qt.PenStyle.DashLine),
            QBrush(QColor(0, 120, 255, 30))
        )
        self.selection_item.setVisible(False)
        
        # Add default layer
        default_image = QImage(*self.canvas_size, QImage.Format.Format_ARGB32)
        default_image.fill(Qt.GlobalColor.white)
        self.add_layer("Background", default_image)
        
    def _setup_view(self):
        """Configure view settings for optimal performance."""
        # Rendering hints
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # View settings
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        
        # Enable OpenGL for better performance
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
            self.setViewport(QOpenGLWidget())
        except ImportError:
            pass  # Fall back to default viewport
            
    def add_layer(self, name: str, image: QImage) -> int:
        """Add a new layer to the canvas."""
        layer = ImageLayer(name, image)
        self.layers.append(layer)
        self.active_layer_index = len(self.layers) - 1
        self.needs_render = True
        self.layer_changed.emit(self.active_layer_index)
        return self.active_layer_index
        
    def remove_layer(self, index: int):
        """Remove a layer from the canvas."""
        if 0 <= index < len(self.layers) and len(self.layers) > 1:
            self.layers.pop(index)
            if self.active_layer_index >= len(self.layers):
                self.active_layer_index = len(self.layers) - 1
            self.needs_render = True
            self.layer_changed.emit(self.active_layer_index)
            
    def merge_layers(self, top_index: int, bottom_index: int):
        """Merge two layers together."""
        if (0 <= top_index < len(self.layers) and 
            0 <= bottom_index < len(self.layers) and 
            top_index != bottom_index):
            
            top_layer = self.layers[top_index]
            bottom_layer = self.layers[bottom_index]
            
            # Create merged image
            merged_image = QImage(bottom_layer.image)
            painter = QPainter(merged_image)
            painter.setOpacity(top_layer.opacity)
            painter.drawImage(0, 0, top_layer.image)
            painter.end()
            
            # Update bottom layer and remove top
            bottom_layer.image = merged_image
            self.layers.pop(top_index)
            
            self.needs_render = True
            self.canvas_modified.emit()
            
    def render_canvas(self):
        """Render all visible layers to the canvas."""
        if not self.needs_render:
            return
            
        # Create composite image
        composite = QPixmap(*self.canvas_size)
        composite.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(composite)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw each visible layer
        for layer in self.layers:
            if layer.visible:
                painter.setOpacity(layer.opacity)
                painter.drawPixmap(0, 0, layer.get_pixmap())
                
        painter.end()
        
        # Update canvas
        self.canvas_item.setPixmap(composite)
        self.needs_render = False
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        # Get scroll direction
        delta = event.angleDelta().y()
        
        # Calculate zoom factor
        if delta > 0:
            zoom_factor = self.zoom_step
        else:
            zoom_factor = 1.0 / self.zoom_step
            
        # Apply zoom
        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_level:
            # Get the scene position before zoom
            scene_pos = self.mapToScene(event.position().toPoint())
            
            # Update zoom
            self.zoom_level = new_zoom
            self.resetTransform()
            self.scale(self.zoom_level, self.zoom_level)
            
            # Center on the mouse position
            self.centerOn(scene_pos)
            
            self.zoom_changed.emit(self.zoom_level)
            
        event.accept()
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning and selection."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check if selecting
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.is_selecting = True
                scene_pos = self.mapToScene(event.position().toPoint())
                self.selection_rect = QRectF(scene_pos, scene_pos)
                self.selection_item.setRect(self.selection_rect)
                self.selection_item.setVisible(True)
            else:
                super().mousePressEvent(event)
                
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning and selection."""
        if self.is_panning:
            # Pan the view
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()
            
            # Update scroll bars
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            
        elif self.is_selecting and self.selection_rect:
            # Update selection rectangle
            scene_pos = self.mapToScene(event.position().toPoint())
            self.selection_rect.setBottomRight(scene_pos)
            self.selection_item.setRect(self.selection_rect.normalized())
            
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            if self.selection_rect:
                self.selection_changed.emit(self.selection_rect.normalized())
                
        else:
            super().mouseReleaseEvent(event)
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            # Temporary pan mode
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            
        elif event.key() == Qt.Key.Key_Escape:
            # Clear selection
            self.selection_item.setVisible(False)
            self.selection_rect = None
            
        elif event.key() == Qt.Key.Key_Delete:
            # Delete selection content
            if self.selection_rect and self.active_layer_index >= 0:
                self.delete_selection()
                
        else:
            super().keyPressEvent(event)
            
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release."""
        if event.key() == Qt.Key.Key_Space:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            super().keyReleaseEvent(event)
            
    def delete_selection(self):
        """Delete content in selection area."""
        if self.selection_rect and 0 <= self.active_layer_index < len(self.layers):
            layer = self.layers[self.active_layer_index]
            if not layer.locked:
                # Clear selection area
                painter = QPainter(layer.image)
                painter.setCompositionMode(QPainter.CompositionMode.Clear)
                painter.fillRect(self.selection_rect.toRect(), Qt.GlobalColor.transparent)
                painter.end()
                
                self.needs_render = True
                self.canvas_modified.emit()
                
    def clear(self):
        """Clear the canvas."""
        # Reset to single white layer
        self.layers.clear()
        default_image = QImage(*self.canvas_size, QImage.Format.Format_ARGB32)
        default_image.fill(Qt.GlobalColor.white)
        self.add_layer("Background", default_image)
        
        # Clear selection
        self.selection_item.setVisible(False)
        self.selection_rect = None
        
        # Reset zoom
        self.zoom_level = 1.0
        self.resetTransform()
        
        self.needs_render = True
        self.canvas_modified.emit()
        
    def get_active_layer(self) -> Optional[ImageLayer]:
        """Get the currently active layer."""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
        
    def export_image(self) -> QImage:
        """Export the composite image."""
        composite = QImage(*self.canvas_size, QImage.Format.Format_ARGB32)
        composite.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(composite)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        for layer in self.layers:
            if layer.visible:
                painter.setOpacity(layer.opacity)
                painter.drawImage(0, 0, layer.image)
                
        painter.end()
        return composite