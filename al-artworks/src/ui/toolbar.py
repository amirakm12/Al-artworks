"""
Toolbar component for Al-artworks with cosmic-themed tools.
Includes cropping, filters, layers, and other image editing tools.
"""

from typing import Dict, Optional, Callable
from PyQt6.QtWidgets import (
    QToolBar, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QButtonGroup, QFrame, QSizePolicy,
    QColorDialog, QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush


class ToolButton(QPushButton):
    """Custom tool button with cosmic styling."""
    
    def __init__(self, icon_name: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setFixedSize(48, 48)
        
        # Cosmic button style
        self.setStyleSheet("""
            ToolButton {
                background-color: rgba(50, 0, 100, 100);
                border: 2px solid rgba(150, 100, 200, 100);
                border-radius: 8px;
                padding: 4px;
            }
            ToolButton:hover {
                background-color: rgba(100, 50, 150, 150);
                border-color: rgba(200, 150, 255, 200);
            }
            ToolButton:checked {
                background-color: rgba(150, 100, 200, 200);
                border-color: rgba(255, 200, 255, 255);
                border-width: 3px;
            }
        """)
        
        # Create placeholder icon
        self._create_icon(icon_name)
        
    def _create_icon(self, name: str):
        """Create a placeholder icon for the tool."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw cosmic gradient circle
        gradient = painter.brush()
        painter.setBrush(QBrush(QColor(200, 150, 255)))
        painter.drawEllipse(4, 4, 24, 24)
        
        # Draw tool initial
        painter.setPen(QColor(50, 0, 100))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, name[0].upper())
        
        painter.end()
        self.setIcon(QIcon(pixmap))


class FilterSlider(QWidget):
    """Slider widget for filter adjustments."""
    
    value_changed = pyqtSignal(str, int)  # filter_name, value
    
    def __init__(self, name: str, min_val: int = -100, max_val: int = 100, default: int = 0):
        super().__init__()
        self.name = name
        
        layout = QVBoxLayout()
        layout.setSpacing(2)
        
        # Label
        self.label = QLabel(f"{name}: {default}")
        self.label.setStyleSheet("color: rgba(200, 150, 255, 255);")
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.slider.valueChanged.connect(self._on_value_changed)
        
        # Cosmic slider style
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid rgba(150, 100, 200, 100);
                height: 8px;
                background: rgba(50, 0, 100, 100);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(200, 150, 255, 255), stop:1 rgba(150, 100, 200, 255));
                border: 1px solid rgba(255, 200, 255, 255);
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 200, 255, 255), stop:1 rgba(200, 150, 255, 255));
            }
        """)
        
        layout.addWidget(self.slider)
        self.setLayout(layout)
        
    def _on_value_changed(self, value: int):
        """Handle slider value change."""
        self.label.setText(f"{self.name}: {value}")
        self.value_changed.emit(self.name, value)


class ToolBar(QWidget):
    """Main toolbar widget containing all tools."""
    
    # Signals
    tool_selected = pyqtSignal(str)
    filter_changed = pyqtSignal(str, int)
    color_selected = pyqtSignal(QColor)
    layer_action = pyqtSignal(str)  # add, delete, merge, etc.
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        
        # Tool groups
        self.tools: Dict[str, ToolButton] = {}
        self.current_tool = None
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the toolbar UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Add cosmic frame
        self.setStyleSheet("""
            ToolBar {
                background-color: rgba(25, 0, 50, 200);
                border: 2px solid rgba(150, 100, 200, 100);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        # Tool selection section
        tools_label = QLabel("Tools")
        tools_label.setStyleSheet("color: rgba(255, 200, 255, 255); font-size: 16px; font-weight: bold;")
        main_layout.addWidget(tools_label)
        
        # Create tool buttons
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(5)
        
        # Selection tools
        selection_row = QHBoxLayout()
        self._add_tool_button("Select", "Selection Tool", selection_row)
        self._add_tool_button("Lasso", "Lasso Selection", selection_row)
        self._add_tool_button("Magic", "Magic Wand", selection_row)
        self._add_tool_button("Crop", "Crop Tool", selection_row)
        selection_row.addStretch()
        tools_layout.addLayout(selection_row)
        
        # Drawing tools
        drawing_row = QHBoxLayout()
        self._add_tool_button("Brush", "Brush Tool", drawing_row)
        self._add_tool_button("Pencil", "Pencil Tool", drawing_row)
        self._add_tool_button("Eraser", "Eraser Tool", drawing_row)
        self._add_tool_button("Text", "Text Tool", drawing_row)
        drawing_row.addStretch()
        tools_layout.addLayout(drawing_row)
        
        # Transform tools
        transform_row = QHBoxLayout()
        self._add_tool_button("Move", "Move Tool", transform_row)
        self._add_tool_button("Rotate", "Rotate Tool", transform_row)
        self._add_tool_button("Scale", "Scale Tool", transform_row)
        self._add_tool_button("Warp", "Warp Tool", transform_row)
        transform_row.addStretch()
        tools_layout.addLayout(transform_row)
        
        main_layout.addLayout(tools_layout)
        
        # Add separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet("background-color: rgba(150, 100, 200, 100);")
        main_layout.addWidget(separator1)
        
        # Filters section
        filters_label = QLabel("Filters")
        filters_label.setStyleSheet("color: rgba(255, 200, 255, 255); font-size: 16px; font-weight: bold;")
        main_layout.addWidget(filters_label)
        
        # Add filter sliders
        self.brightness_slider = FilterSlider("Brightness", -100, 100, 0)
        self.brightness_slider.value_changed.connect(self.filter_changed)
        main_layout.addWidget(self.brightness_slider)
        
        self.contrast_slider = FilterSlider("Contrast", -100, 100, 0)
        self.contrast_slider.value_changed.connect(self.filter_changed)
        main_layout.addWidget(self.contrast_slider)
        
        self.saturation_slider = FilterSlider("Saturation", -100, 100, 0)
        self.saturation_slider.value_changed.connect(self.filter_changed)
        main_layout.addWidget(self.saturation_slider)
        
        self.hue_slider = FilterSlider("Hue", -180, 180, 0)
        self.hue_slider.value_changed.connect(self.filter_changed)
        main_layout.addWidget(self.hue_slider)
        
        # Add separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: rgba(150, 100, 200, 100);")
        main_layout.addWidget(separator2)
        
        # Layers section
        layers_label = QLabel("Layers")
        layers_label.setStyleSheet("color: rgba(255, 200, 255, 255); font-size: 16px; font-weight: bold;")
        main_layout.addWidget(layers_label)
        
        # Layer controls
        layer_controls = QHBoxLayout()
        
        add_layer_btn = QPushButton("Add")
        add_layer_btn.clicked.connect(lambda: self.layer_action.emit("add"))
        add_layer_btn.setStyleSheet(self._get_button_style())
        layer_controls.addWidget(add_layer_btn)
        
        delete_layer_btn = QPushButton("Delete")
        delete_layer_btn.clicked.connect(lambda: self.layer_action.emit("delete"))
        delete_layer_btn.setStyleSheet(self._get_button_style())
        layer_controls.addWidget(delete_layer_btn)
        
        merge_layer_btn = QPushButton("Merge")
        merge_layer_btn.clicked.connect(lambda: self.layer_action.emit("merge"))
        merge_layer_btn.setStyleSheet(self._get_button_style())
        layer_controls.addWidget(merge_layer_btn)
        
        layer_controls.addStretch()
        main_layout.addLayout(layer_controls)
        
        # Blend mode
        blend_layout = QHBoxLayout()
        blend_label = QLabel("Blend:")
        blend_label.setStyleSheet("color: rgba(200, 150, 255, 255);")
        blend_layout.addWidget(blend_label)
        
        self.blend_mode = QComboBox()
        self.blend_mode.addItems(["Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light"])
        self.blend_mode.setStyleSheet(self._get_combo_style())
        blend_layout.addWidget(self.blend_mode)
        blend_layout.addStretch()
        
        main_layout.addLayout(blend_layout)
        
        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setStyleSheet("color: rgba(200, 150, 255, 255);")
        opacity_layout.addWidget(opacity_label)
        
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.setStyleSheet(self._get_spin_style())
        opacity_layout.addWidget(self.opacity_spin)
        opacity_layout.addStretch()
        
        main_layout.addLayout(opacity_layout)
        
        # Add separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setStyleSheet("background-color: rgba(150, 100, 200, 100);")
        main_layout.addWidget(separator3)
        
        # Color section
        color_label = QLabel("Color")
        color_label.setStyleSheet("color: rgba(255, 200, 255, 255); font-size: 16px; font-weight: bold;")
        main_layout.addWidget(color_label)
        
        # Color picker button
        self.color_button = QPushButton()
        self.color_button.setFixedSize(60, 60)
        self.color_button.clicked.connect(self._pick_color)
        self._update_color_button(QColor(255, 255, 255))
        main_layout.addWidget(self.color_button, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # Add stretch at bottom
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
    def _add_tool_button(self, name: str, tooltip: str, layout: QHBoxLayout):
        """Add a tool button to the toolbar."""
        button = ToolButton(name, tooltip)
        button.clicked.connect(lambda: self._on_tool_selected(name))
        self.tools[name] = button
        layout.addWidget(button)
        
        # Create button group for exclusive selection
        if not hasattr(self, 'tool_group'):
            self.tool_group = QButtonGroup()
        self.tool_group.addButton(button)
        
    def _on_tool_selected(self, tool_name: str):
        """Handle tool selection."""
        self.current_tool = tool_name
        self.tool_selected.emit(tool_name)
        
    def _pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Color")
        if color.isValid():
            self._update_color_button(color)
            self.color_selected.emit(color)
            
    def _update_color_button(self, color: QColor):
        """Update color button appearance."""
        self.color_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color.name()};
                border: 3px solid rgba(150, 100, 200, 255);
                border-radius: 8px;
            }}
            QPushButton:hover {{
                border-color: rgba(255, 200, 255, 255);
            }}
        """)
        
    def _get_button_style(self) -> str:
        """Get cosmic button style."""
        return """
            QPushButton {
                background-color: rgba(100, 50, 150, 150);
                color: rgba(255, 200, 255, 255);
                border: 2px solid rgba(150, 100, 200, 200);
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(150, 100, 200, 200);
                border-color: rgba(255, 200, 255, 255);
            }
            QPushButton:pressed {
                background-color: rgba(200, 150, 255, 255);
            }
        """
        
    def _get_combo_style(self) -> str:
        """Get cosmic combo box style."""
        return """
            QComboBox {
                background-color: rgba(50, 0, 100, 150);
                color: rgba(255, 200, 255, 255);
                border: 2px solid rgba(150, 100, 200, 200);
                border-radius: 4px;
                padding: 2px 4px;
            }
            QComboBox:hover {
                border-color: rgba(255, 200, 255, 255);
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid rgba(255, 200, 255, 255);
                margin-right: 4px;
            }
        """
        
    def _get_spin_style(self) -> str:
        """Get cosmic spin box style."""
        return """
            QSpinBox {
                background-color: rgba(50, 0, 100, 150);
                color: rgba(255, 200, 255, 255);
                border: 2px solid rgba(150, 100, 200, 200);
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:hover {
                border-color: rgba(255, 200, 255, 255);
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: rgba(100, 50, 150, 150);
                border: none;
                width: 16px;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid rgba(255, 200, 255, 255);
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid rgba(255, 200, 255, 255);
            }
        """
        
    def reset_filters(self):
        """Reset all filters to default values."""
        self.brightness_slider.slider.setValue(0)
        self.contrast_slider.slider.setValue(0)
        self.saturation_slider.slider.setValue(0)
        self.hue_slider.slider.setValue(0)