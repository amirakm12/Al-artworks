"""
Cosmic theme definitions for Al-artworks.
Provides dark-to-light gradient themes with neon accents.
"""

from typing import Dict, Tuple
from PyQt6.QtGui import QColor, QLinearGradient, QPalette
from PyQt6.QtCore import QPointF


class CosmicTheme:
    """Cosmic theme provider with celestial gradients and neon accents."""
    
    # Color definitions
    DEEP_SPACE_BLACK = QColor(0, 0, 0)
    DARK_PURPLE = QColor(25, 0, 51)
    COSMIC_PURPLE = QColor(51, 0, 102)
    LIGHT_PURPLE = QColor(102, 51, 153)
    CELESTIAL_PURPLE = QColor(153, 102, 204)
    LIGHT_COSMIC = QColor(204, 153, 255)
    NEON_PINK = QColor(255, 0, 255)
    NEON_BLUE = QColor(0, 255, 255)
    STARLIGHT_WHITE = QColor(255, 255, 255)
    
    def __init__(self):
        """Initialize the cosmic theme."""
        self.current_theme = "dark"
        
    def get_main_window_style(self) -> str:
        """Get the main window stylesheet."""
        return """
        QMainWindow {
            background-color: rgba(0, 0, 0, 255);
        }
        
        QMenuBar {
            background-color: rgba(25, 0, 51, 200);
            color: rgba(255, 200, 255, 255);
            border-bottom: 2px solid rgba(150, 100, 200, 100);
            padding: 4px;
        }
        
        QMenuBar::item:selected {
            background-color: rgba(100, 50, 150, 150);
            border-radius: 4px;
        }
        
        QMenu {
            background-color: rgba(25, 0, 51, 240);
            color: rgba(255, 200, 255, 255);
            border: 2px solid rgba(150, 100, 200, 200);
            border-radius: 8px;
            padding: 4px;
        }
        
        QMenu::item:selected {
            background-color: rgba(100, 50, 150, 200);
            border-radius: 4px;
        }
        
        QStatusBar {
            background-color: rgba(25, 0, 51, 200);
            color: rgba(200, 150, 255, 255);
            border-top: 2px solid rgba(150, 100, 200, 100);
        }
        
        QDockWidget {
            color: rgba(255, 200, 255, 255);
        }
        
        QDockWidget::title {
            background-color: rgba(50, 0, 100, 200);
            padding: 6px;
            border: 2px solid rgba(150, 100, 200, 100);
            border-radius: 4px;
        }
        
        QScrollBar:vertical {
            background-color: rgba(25, 0, 51, 100);
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: rgba(150, 100, 200, 200);
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: rgba(200, 150, 255, 255);
        }
        
        QScrollBar:horizontal {
            background-color: rgba(25, 0, 51, 100);
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: rgba(150, 100, 200, 200);
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: rgba(200, 150, 255, 255);
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        """
        
    def get_cosmic_gradient(self, start: QPointF, end: QPointF, animated_offset: float = 0.0) -> QLinearGradient:
        """Create an animated cosmic gradient."""
        gradient = QLinearGradient(start, end)
        
        # Animated gradient stops for celestial effect
        gradient.setColorAt(0.0, self.DEEP_SPACE_BLACK)
        gradient.setColorAt(0.3 + animated_offset * 0.1, self.DARK_PURPLE)
        gradient.setColorAt(0.5 + animated_offset * 0.1, self.COSMIC_PURPLE)
        gradient.setColorAt(0.7 + animated_offset * 0.1, self.LIGHT_PURPLE)
        gradient.setColorAt(0.9, self.CELESTIAL_PURPLE)
        gradient.setColorAt(1.0, self.LIGHT_COSMIC)
        
        return gradient
        
    def get_neon_glow_style(self, base_color: QColor, glow_color: QColor) -> str:
        """Create a neon glow effect style."""
        return f"""
        background-color: {base_color.name()};
        border: 2px solid {glow_color.name()};
        border-radius: 8px;
        box-shadow: 0 0 10px {glow_color.name()},
                    0 0 20px {glow_color.name()},
                    0 0 30px {glow_color.name()};
        """
        
    def get_holographic_style(self) -> str:
        """Get holographic effect style for AR previews."""
        return """
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(255, 0, 255, 50),
            stop:0.33 rgba(0, 255, 255, 50),
            stop:0.66 rgba(255, 255, 0, 50),
            stop:1 rgba(255, 0, 255, 50));
        border: 2px solid rgba(255, 255, 255, 100);
        border-radius: 10px;
        """
        
    def get_button_style(self, button_type: str = "default") -> str:
        """Get button style based on type."""
        if button_type == "primary":
            return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(200, 150, 255, 200),
                    stop:1 rgba(150, 100, 200, 200));
                color: white;
                border: 2px solid rgba(255, 200, 255, 255);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 200, 255, 255),
                    stop:1 rgba(200, 150, 255, 255));
                border-color: rgba(255, 255, 255, 255);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(150, 100, 200, 255),
                    stop:1 rgba(100, 50, 150, 255));
            }
            """
        elif button_type == "secondary":
            return """
            QPushButton {
                background-color: rgba(50, 0, 100, 150);
                color: rgba(255, 200, 255, 255);
                border: 2px solid rgba(150, 100, 200, 200);
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(100, 50, 150, 200);
                border-color: rgba(200, 150, 255, 255);
            }
            QPushButton:pressed {
                background-color: rgba(150, 100, 200, 255);
            }
            """
        else:  # default
            return """
            QPushButton {
                background-color: rgba(75, 25, 125, 100);
                color: rgba(255, 200, 255, 255);
                border: 1px solid rgba(150, 100, 200, 150);
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: rgba(100, 50, 150, 150);
                border-color: rgba(200, 150, 255, 200);
            }
            QPushButton:pressed {
                background-color: rgba(125, 75, 175, 200);
            }
            """
            
    def get_input_style(self) -> str:
        """Get style for input fields."""
        return """
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: rgba(25, 0, 51, 150);
            color: rgba(255, 200, 255, 255);
            border: 2px solid rgba(150, 100, 200, 150);
            border-radius: 4px;
            padding: 4px;
            selection-background-color: rgba(200, 150, 255, 150);
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: rgba(255, 200, 255, 255);
            background-color: rgba(50, 0, 100, 200);
        }
        """
        
    def get_eve_avatar_style(self) -> str:
        """Get style for Eve's avatar widget."""
        return """
        background: radial-gradient(circle at center,
            rgba(255, 200, 255, 100) 0%,
            rgba(200, 150, 255, 50) 30%,
            rgba(150, 100, 200, 25) 60%,
            transparent 100%);
        border: 3px solid rgba(255, 200, 255, 200);
        border-radius: 10px;
        """
        
    def get_layer_item_style(self) -> str:
        """Get style for layer list items."""
        return """
        QListWidget {
            background-color: rgba(25, 0, 51, 100);
            border: 2px solid rgba(150, 100, 200, 100);
            border-radius: 4px;
        }
        
        QListWidget::item {
            background-color: rgba(50, 0, 100, 100);
            color: rgba(255, 200, 255, 255);
            border: 1px solid rgba(150, 100, 200, 50);
            border-radius: 4px;
            padding: 4px;
            margin: 2px;
        }
        
        QListWidget::item:selected {
            background-color: rgba(150, 100, 200, 200);
            border-color: rgba(255, 200, 255, 255);
        }
        
        QListWidget::item:hover {
            background-color: rgba(100, 50, 150, 150);
            border-color: rgba(200, 150, 255, 200);
        }
        """
        
    def apply_cosmic_palette(self, widget) -> None:
        """Apply cosmic color palette to a widget."""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, self.DARK_PURPLE)
        palette.setColor(QPalette.ColorRole.WindowText, self.LIGHT_COSMIC)
        
        # Base colors
        palette.setColor(QPalette.ColorRole.Base, self.COSMIC_PURPLE)
        palette.setColor(QPalette.ColorRole.AlternateBase, self.LIGHT_PURPLE)
        
        # Text colors
        palette.setColor(QPalette.ColorRole.Text, self.STARLIGHT_WHITE)
        palette.setColor(QPalette.ColorRole.BrightText, self.NEON_PINK)
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, self.COSMIC_PURPLE)
        palette.setColor(QPalette.ColorRole.ButtonText, self.LIGHT_COSMIC)
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, self.CELESTIAL_PURPLE)
        palette.setColor(QPalette.ColorRole.HighlightedText, self.STARLIGHT_WHITE)
        
        widget.setPalette(palette)
        
    def get_animation_duration(self, animation_type: str) -> int:
        """Get animation duration in milliseconds."""
        durations = {
            "fade": 300,
            "slide": 250,
            "glow": 1000,
            "celestial_birth": 3000,
            "eve_greeting": 2000,
            "tool_switch": 150,
            "filter_apply": 100
        }
        return durations.get(animation_type, 200)