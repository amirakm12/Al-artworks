"""
Main application window for Al-artworks with cosmic theme.
Features Eve as the voice-driven creative goddess.
"""

import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QSplitter, QDockWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QAction, QPalette, QLinearGradient, QColor, 
    QPainter, QBrush, QPaintEvent
)

from .canvas import ImageCanvas
from .toolbar import ToolBar
from .eve_avatar import EveAvatarWidget
from .themes import CosmicTheme


class CosmicBackgroundWidget(QWidget):
    """Widget with animated cosmic gradient background."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gradient_offset = 0.0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_gradient)
        self.animation_timer.start(50)  # 20fps for smooth animation
        
    def update_gradient(self):
        """Animate the gradient slowly."""
        self.gradient_offset += 0.001
        if self.gradient_offset > 1.0:
            self.gradient_offset = 0.0
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the cosmic gradient background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create cosmic gradient from black to purple/white
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        
        # Animated gradient stops for celestial effect
        gradient.setColorAt(0.0, QColor(0, 0, 0))  # Deep space black
        gradient.setColorAt(0.3 + self.gradient_offset * 0.1, QColor(25, 0, 51))  # Dark purple
        gradient.setColorAt(0.5 + self.gradient_offset * 0.1, QColor(51, 0, 102))  # Purple
        gradient.setColorAt(0.7 + self.gradient_offset * 0.1, QColor(102, 51, 153))  # Light purple
        gradient.setColorAt(0.9, QColor(153, 102, 204))  # Cosmic purple
        gradient.setColorAt(1.0, QColor(204, 153, 255))  # Light cosmic
        
        painter.fillRect(self.rect(), QBrush(gradient))


class MainWindow(QMainWindow):
    """Main application window with cosmic theme."""
    
    # Signals
    eve_activated = pyqtSignal()
    voice_command_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Al-artworks - The Birth of Celestial Art")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply cosmic theme
        self.theme = CosmicTheme()
        self.setStyleSheet(self.theme.get_main_window_style())
        
        # Initialize UI components
        self._init_ui()
        self._init_menus()
        self._init_status_bar()
        self._init_eve()
        
        # Start with black screen animation
        self._start_celestial_birth_animation()
        
    def _init_ui(self):
        """Initialize the main UI layout."""
        # Create central widget with cosmic background
        self.cosmic_background = CosmicBackgroundWidget()
        self.setCentralWidget(self.cosmic_background)
        
        # Main layout
        main_layout = QVBoxLayout(self.cosmic_background)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for main content
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel - Tools and Eve
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Add toolbar
        self.toolbar = ToolBar()
        left_layout.addWidget(self.toolbar)
        
        # Add Eve avatar widget
        self.eve_widget = EveAvatarWidget()
        left_layout.addWidget(self.eve_widget)
        left_layout.addStretch()
        
        # Center - Canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(500, 500)
        
        # Add to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.canvas)
        self.main_splitter.setSizes([300, 1100])
        
        # Create dock widgets for additional panels
        self._create_dock_widgets()
        
    def _create_dock_widgets(self):
        """Create dockable widgets for layers, history, etc."""
        # Layers dock
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        layers_widget = QWidget()
        layers_widget.setMinimumWidth(250)
        self.layers_dock.setWidget(layers_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layers_dock)
        
        # Properties dock
        self.properties_dock = QDockWidget("Properties", self)
        self.properties_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        properties_widget = QWidget()
        properties_widget.setMinimumWidth(250)
        self.properties_dock.setWidget(properties_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
    def _init_menus(self):
        """Initialize menu bar with cosmic styling."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.layers_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        
        # Eve menu
        eve_menu = menubar.addMenu("&Eve")
        
        activate_eve_action = QAction("&Activate Eve", self)
        activate_eve_action.setShortcut("Ctrl+E")
        activate_eve_action.triggered.connect(self.activate_eve)
        eve_menu.addAction(activate_eve_action)
        
        voice_command_action = QAction("&Voice Command", self)
        voice_command_action.setShortcut("Space")
        eve_menu.addAction(voice_command_action)
        
    def _init_status_bar(self):
        """Initialize status bar with cosmic styling."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Al-artworks - Eve is ready to assist")
        
    def _init_eve(self):
        """Initialize Eve integration."""
        # Connect Eve signals
        self.eve_widget.voice_command.connect(self.handle_voice_command)
        self.eve_activated.connect(self.eve_widget.activate)
        
    def _start_celestial_birth_animation(self):
        """Start the celestial birth animation - black to cosmic."""
        # Create overlay widget for animation
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("background-color: black;")
        self.overlay.resize(self.size())
        self.overlay.show()
        
        # Fade out animation
        self.fade_animation = QPropertyAnimation(self.overlay, b"windowOpacity")
        self.fade_animation.setDuration(3000)  # 3 seconds
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_animation.finished.connect(self.overlay.deleteLater)
        
        # Start animation after a brief black screen
        QTimer.singleShot(1000, self.fade_animation.start)
        QTimer.singleShot(1500, self.activate_eve)  # Activate Eve during animation
        
    def activate_eve(self):
        """Activate Eve with greeting."""
        self.eve_activated.emit()
        self.status_bar.showMessage("Eve activated - 'Hello, I am Eve, your celestial creative goddess'")
        
    def handle_voice_command(self, command: str):
        """Handle voice commands from Eve."""
        self.voice_command_received.emit(command)
        self.status_bar.showMessage(f"Voice command: {command}")
        
    def new_project(self):
        """Create new project."""
        self.canvas.clear()
        self.status_bar.showMessage("New project created")
        
    def open_project(self):
        """Open existing project."""
        # Will be implemented with file dialog
        self.status_bar.showMessage("Opening project...")
        
    def save_project(self):
        """Save current project."""
        # Will be implemented with file dialog
        self.status_bar.showMessage("Saving project...")
        
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.resize(self.size())


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())