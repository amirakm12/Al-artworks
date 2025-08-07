"""
Tools Panel - AI Tools and Capabilities Interface
Provides access to different AI tools and capabilities
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFrame, QScrollArea, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from typing import Dict, Any

class ToolButton(QPushButton):
    """Custom tool button with icon and description"""
    
    def __init__(self, name: str, description: str, icon: str = "", tool_name: str = ""):
        super().__init__()
        self.tool_name = tool_name or name.lower().replace(" ", "_")
        self.setup_ui(name, description, icon)
    
    def setup_ui(self, name: str, description: str, icon: str):
        """Setup the button UI"""
        layout = QVBoxLayout(self)
        
        # Icon and name
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 24px; margin: 5px;")
        
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px; margin: 2px;")
        
        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 10px; color: gray; margin: 2px;")
        
        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        layout.addWidget(desc_label)
        
        # Style
        self.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 2px solid #E9ECEF;
                border-radius: 10px;
                padding: 10px;
                min-height: 80px;
                max-height: 80px;
            }
            QPushButton:hover {
                background-color: #E9ECEF;
                border-color: #007AFF;
            }
            QPushButton:pressed {
                background-color: #DEE2E6;
            }
        """)

class ToolsPanel(QWidget):
    """Main tools panel widget"""
    
    # Signals
    tool_activated = pyqtSignal(str)
    
    def __init__(self, agent_orchestrator):
        super().__init__()
        self.agent_orchestrator = agent_orchestrator
        self.tools = {}
        
        self.setup_ui()
        self.setup_tools()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("AI Tools")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #333;
            padding: 10px;
            border-bottom: 2px solid #E9ECEF;
        """)
        layout.addWidget(title)
        
        # Tools area
        self.tools_area = QScrollArea()
        self.tools_widget = QWidget()
        self.tools_layout = QGridLayout(self.tools_widget)
        self.tools_area.setWidget(self.tools_widget)
        self.tools_area.setWidgetResizable(True)
        self.tools_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        layout.addWidget(self.tools_area)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            font-size: 11px;
            color: #6C757D;
            padding: 5px;
            border-top: 1px solid #E9ECEF;
        """)
        layout.addWidget(self.status_label)
    
    def setup_tools(self):
        """Setup available tools"""
        tools_config = [
            {
                "name": "Voice Assistant",
                "description": "Speech recognition and synthesis",
                "icon": "🎤",
                "tool_name": "voice"
            },
            {
                "name": "Code Interpreter",
                "description": "Python code execution",
                "icon": "💻",
                "tool_name": "code_interpreter"
            },
            {
                "name": "Web Browser",
                "description": "Real-time web search",
                "icon": "🌐",
                "tool_name": "web_browser"
            },
            {
                "name": "Image Generator",
                "description": "AI image creation",
                "icon": "🎨",
                "tool_name": "image_editor"
            },
            {
                "name": "File Processor",
                "description": "Upload and analyze files",
                "icon": "📁",
                "tool_name": "file_processor"
            },
            {
                "name": "Memory Manager",
                "description": "Conversation history",
                "icon": "🧠",
                "tool_name": "memory_manager"
            },
            {
                "name": "Plugin System",
                "description": "Custom extensions",
                "icon": "🔌",
                "tool_name": "plugin_system"
            },
            {
                "name": "AR Overlay",
                "description": "3D visual interface",
                "icon": "🎭",
                "tool_name": "ar_overlay"
            }
        ]
        
        # Create tool buttons
        row, col = 0, 0
        max_cols = 2
        
        for tool_config in tools_config:
            button = ToolButton(
                tool_config["name"],
                tool_config["description"],
                tool_config["icon"],
                tool_config["tool_name"]
            )
            button.clicked.connect(lambda checked, tn=tool_config["tool_name"]: self.activate_tool(tn))
            
            self.tools_layout.addWidget(button, row, col)
            self.tools[tool_config["tool_name"]] = button
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Add stretch to push tools to top
        self.tools_layout.setRowStretch(row + 1, 1)
    
    def activate_tool(self, tool_name: str):
        """Activate a specific tool"""
        self.tool_activated.emit(tool_name)
        self.status_label.setText(f"Activated: {tool_name}")
        
        # Visual feedback
        if tool_name in self.tools:
            button = self.tools[tool_name]
            button.setStyleSheet("""
                QPushButton {
                    background-color: #D4EDDA;
                    border: 2px solid #28A745;
                    border-radius: 10px;
                    padding: 10px;
                    min-height: 80px;
                    max-height: 80px;
                }
                QPushButton:hover {
                    background-color: #C3E6CB;
                }
            """)
    
    def update_tool_status(self, tool_name: str, status: str):
        """Update tool status"""
        if tool_name in self.tools:
            button = self.tools[tool_name]
            # Update button appearance based on status
            if status == "active":
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #D4EDDA;
                        border: 2px solid #28A745;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 80px;
                        max-height: 80px;
                    }
                """)
            elif status == "error":
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #F8D7DA;
                        border: 2px solid #DC3545;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 80px;
                        max-height: 80px;
                    }
                """)
            else:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #F8F9FA;
                        border: 2px solid #E9ECEF;
                        border-radius: 10px;
                        padding: 10px;
                        min-height: 80px;
                        max-height: 80px;
                    }
                    QPushButton:hover {
                        background-color: #E9ECEF;
                        border-color: #007AFF;
                    }
                """)
    
    def get_available_tools(self) -> list:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def set_status(self, status: str):
        """Set the status message"""
        self.status_label.setText(status)