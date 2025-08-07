"""
Chat Interface - Main chat UI component
Provides the primary chat interface with message display and input
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                              QLineEdit, QPushButton, QScrollArea, QLabel,
                              QFrame, QSplitter, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette
from typing import Dict, Any, List
import json
from datetime import datetime

class ChatMessage(QFrame):
    """Individual chat message widget"""
    
    def __init__(self, message: str, is_user: bool = True, timestamp: str = None):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setMaximumWidth(600)
        
        # Setup layout
        layout = QVBoxLayout(self)
        
        # Message content
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                border-radius: 10px;
                font-size: 14px;
                line-height: 1.4;
            }
        """)
        
        # Timestamp
        if timestamp:
            self.timestamp_label = QLabel(timestamp)
            self.timestamp_label.setStyleSheet("font-size: 10px; color: gray;")
            layout.addWidget(self.timestamp_label)
        
        layout.addWidget(self.message_label)
        
        # Style based on message type
        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background-color: #007AFF;
                    border-radius: 15px;
                    margin: 5px;
                }
                QLabel {
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #F0F0F0;
                    border-radius: 15px;
                    margin: 5px;
                }
                QLabel {
                    color: black;
                }
            """)

class ChatInterface(QWidget):
    """Main chat interface widget"""
    
    # Signals
    message_sent = pyqtSignal(str)
    file_dropped = pyqtSignal(str)
    
    def __init__(self, agent_orchestrator):
        super().__init__()
        self.agent_orchestrator = agent_orchestrator
        self.messages = []
        
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Chat area
        self.chat_area = QScrollArea()
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch()  # Push messages to top
        self.chat_area.setWidget(self.chat_widget)
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 2px solid #E0E0E0;
                border-radius: 20px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #007AFF;
            }
        """)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
        """)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        
        # Add widgets to main layout
        layout.addWidget(self.chat_area)
        layout.addLayout(input_layout)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def setup_styles(self):
        """Setup application styles"""
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QScrollArea {
                border: none;
                background-color: #FAFAFA;
            }
        """)
    
    def send_message(self):
        """Send the current message"""
        message = self.message_input.text().strip()
        if message:
            self.add_user_message(message)
            self.message_input.clear()
            self.message_sent.emit(message)
    
    def add_user_message(self, message: str):
        """Add a user message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        message_widget = ChatMessage(message, is_user=True, timestamp=timestamp)
        
        # Add to layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        self.messages.append({"type": "user", "content": message, "timestamp": timestamp})
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        message_widget = ChatMessage(message, is_user=False, timestamp=timestamp)
        
        # Add to layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        self.messages.append({"type": "assistant", "content": message, "timestamp": timestamp})
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def add_system_message(self, message: str):
        """Add a system message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        message_widget = ChatMessage(f"System: {message}", is_user=False, timestamp=timestamp)
        message_widget.setStyleSheet("""
            QFrame {
                background-color: #FFF3CD;
                border: 1px solid #FFEAA7;
                border-radius: 15px;
                margin: 5px;
            }
            QLabel {
                color: #856404;
            }
        """)
        
        # Add to layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)
        self.messages.append({"type": "system", "content": message, "timestamp": timestamp})
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the chat area to the bottom"""
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_chat(self):
        """Clear all messages from the chat"""
        # Remove all message widgets
        while self.chat_layout.count() > 1:  # Keep the stretch
            child = self.chat_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.messages.clear()
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history"""
        return self.messages.copy()
    
    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop event for file uploads"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.file_dropped.emit(file_path)
            self.add_system_message(f"File uploaded: {file_path}")
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            # Shift+Enter for new line
            self.message_input.insert("\n")
        else:
            super().keyPressEvent(event)