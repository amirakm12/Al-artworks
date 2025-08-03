#!/usr/bin/env python3
"""
Al-artworks - The Birth of Celestial Art
A Qt6-based image-editing application featuring Eve, 
a voice-driven hyper-intelligent creative goddess.

Author: Al-artworks Team
Version: 1.0.0
Python: 3.13+
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient

from src.ui.main_window import MainWindow
from src.utils.config import Config
from src.utils.logger import setup_logger


class CosmicSplashScreen(QSplashScreen):
    """Custom splash screen with cosmic theme."""
    
    def __init__(self):
        # Create cosmic gradient pixmap
        pixmap = QPixmap(600, 400)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw cosmic gradient background
        gradient = QLinearGradient(0, 0, 600, 400)
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(0.3, QColor(25, 0, 51))
        gradient.setColorAt(0.5, QColor(51, 0, 102))
        gradient.setColorAt(0.7, QColor(102, 51, 153))
        gradient.setColorAt(1.0, QColor(153, 102, 204))
        
        painter.fillRect(pixmap.rect(), gradient)
        
        # Draw title
        painter.setPen(QColor(255, 200, 255))
        title_font = QFont("Arial", 32, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Al-artworks")
        
        # Draw subtitle
        painter.setPen(QColor(200, 150, 255))
        subtitle_font = QFont("Arial", 16)
        painter.setFont(subtitle_font)
        subtitle_rect = pixmap.rect()
        subtitle_rect.moveTop(subtitle_rect.top() + 60)
        painter.drawText(subtitle_rect, Qt.AlignmentFlag.AlignCenter, 
                        "The Birth of Celestial Art")
        
        # Draw Eve's name
        painter.setPen(QColor(255, 255, 255))
        eve_font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(eve_font)
        eve_rect = pixmap.rect()
        eve_rect.moveTop(eve_rect.top() + 120)
        painter.drawText(eve_rect, Qt.AlignmentFlag.AlignCenter, "EVE")
        
        # Draw loading message
        painter.setPen(QColor(200, 200, 200))
        loading_font = QFont("Arial", 12)
        painter.setFont(loading_font)
        loading_rect = pixmap.rect()
        loading_rect.moveTop(loading_rect.top() + 300)
        painter.drawText(loading_rect, Qt.AlignmentFlag.AlignCenter, 
                        "Awakening the celestial creative goddess...")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)


class AlArtworksApp:
    """Main application class."""
    
    def __init__(self):
        self.app = None
        self.window = None
        self.config = None
        self.logger = None
        
    def run(self):
        """Run the application."""
        # Create Qt application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Al-artworks")
        self.app.setOrganizationName("Al-artworks Team")
        
        # Setup logging
        self.logger = setup_logger()
        self.logger.info("Starting Al-artworks application")
        
        # Load configuration
        self.config = Config()
        self.config.load()
        
        # Show splash screen
        splash = CosmicSplashScreen()
        splash.show()
        self.app.processEvents()
        
        # Initialize main window
        self.window = MainWindow()
        
        # Show main window after splash
        QTimer.singleShot(3000, lambda: self._show_main_window(splash))
        
        # Run application
        return self.app.exec()
        
    def _show_main_window(self, splash):
        """Show main window and close splash screen."""
        self.window.show()
        splash.finish(self.window)
        self.logger.info("Main window displayed")


def main():
    """Main entry point."""
    # Check Python version
    if sys.version_info < (3, 13):
        print("Error: Al-artworks requires Python 3.13 or higher")
        sys.exit(1)
        
    # Create and run application
    app = AlArtworksApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()