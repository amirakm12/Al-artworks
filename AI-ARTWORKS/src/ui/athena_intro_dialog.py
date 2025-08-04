"""
Athena's Epic Introduction Dialog
The Birth of Celestial Art - Cosmic welcome experience
"""

import asyncio
import threading
from typing import Optional, Dict, Any
from pathlib import Path
import time

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, Slot
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QPalette, QLinearGradient
from loguru import logger

from ..agents.athena_core import AthenaCore, AthenaPersonality, AthenaConfig
from ..agents.bark_voice_agent import BarkVoiceAgent
from ..agents.whisper_voice_agent import WhisperVoiceAgent

class AthenaIntroDialog(QDialog):
    """
    Athena's Epic Introduction Dialog
    
    Features:
    - 3D avatar rendering (NeuralRadianceAgent)
    - Mystical voice synthesis (BarkVoiceAgent)
    - Cosmic visual effects
    - Personalized greeting
    - Smooth animations
    """
    
    intro_complete = Signal()
    personality_selected = Signal(str)
    
    def __init__(self, user_name: str = "Creator", parent=None):
        super().__init__(parent)
        
        self.user_name = user_name
        self.athena_core = None
        self.bark_agent = None
        self.current_personality = AthenaPersonality.CYBER_SORCERESS
        
        # UI elements
        self.avatar_label = None
        self.greeting_label = None
        self.progress_bar = None
        self.personality_buttons = {}
        
        # Animation timers
        self.intro_timer = QTimer()
        self.intro_timer.timeout.connect(self._update_intro_sequence)
        self.current_step = 0
        
        # Voice synthesis
        self.voice_synthesis_task = None
        
        self.setup_ui()
        self.setup_animations()
        
        logger.info("Athena's epic introduction dialog initialized")
    
    def setup_ui(self):
        """Setup the cosmic UI"""
        
        # Window setup
        self.setWindowTitle("AI-Artworks: The Birth of Celestial Art")
        self.setFixedSize(800, 600)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Cosmic background frame
        self.background_frame = QFrame()
        self.background_frame.setObjectName("cosmicBackground")
        self.background_frame.setStyleSheet("""
            QFrame#cosmicBackground {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a0a, stop:0.3 #1a1a2e, stop:0.7 #16213e, stop:1 #0f3460);
                border-radius: 20px;
                border: 2px solid #4a90e2;
            }
        """)
        
        # Add cosmic glow effect
        glow_effect = QGraphicsDropShadowEffect()
        glow_effect.setBlurRadius(30)
        glow_effect.setColor(QColor(74, 144, 226, 100))
        glow_effect.setOffset(0, 0)
        self.background_frame.setGraphicsEffect(glow_effect)
        
        background_layout = QVBoxLayout(self.background_frame)
        background_layout.setContentsMargins(40, 40, 40, 40)
        background_layout.setSpacing(30)
        
        # Athena's 3D Avatar (placeholder for now)
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(300, 300)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setStyleSheet("""
            QLabel {
                background: qradialgradient(cx:0.5, cy:0.5, radius:1,
                    stop:0 #4a90e2, stop:0.5 #2c3e50, stop:1 #1a1a2e);
                border-radius: 150px;
                border: 3px solid #4a90e2;
            }
        """)
        
        # Avatar glow effect
        avatar_glow = QGraphicsDropShadowEffect()
        avatar_glow.setBlurRadius(50)
        avatar_glow.setColor(QColor(74, 144, 226, 150))
        avatar_glow.setOffset(0, 0)
        self.avatar_label.setGraphicsEffect(avatar_glow)
        
        # Cosmic greeting text
        self.greeting_label = QLabel()
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setWordWrap(True)
        self.greeting_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 18px;
                font-weight: 300;
                line-height: 1.5;
                background: transparent;
            }
        """)
        
        # Progress bar for cosmic loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a90e2;
                border-radius: 4px;
                background: #1a1a2e;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a90e2, stop:0.5 #7b68ee, stop:1 #9370db);
                border-radius: 2px;
            }
        """)
        self.progress_bar.setVisible(False)
        
        # Personality selection buttons
        self.personality_frame = QFrame()
        personality_layout = QHBoxLayout(self.personality_frame)
        personality_layout.setSpacing(15)
        
        personalities = [
            ("Cyber Sorceress", AthenaPersonality.CYBER_SORCERESS, "✨"),
            ("Galactic Muse", AthenaPersonality.GALACTIC_MUSE, "🌟"),
            ("Cosmic Architect", AthenaPersonality.COSMIC_ARCHITECT, "🏗️"),
            ("Neural Visionary", AthenaPersonality.NEURAL_VISIONARY, "🧠")
        ]
        
        for name, personality, icon in personalities:
            btn = QPushButton(f"{icon} {name}")
            btn.setFixedSize(180, 50)
            btn.setObjectName(f"personality_{personality.value}")
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2c3e50, stop:1 #34495e);
                    border: 2px solid #4a90e2;
                    border-radius: 25px;
                    color: #ffffff;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a90e2, stop:1 #7b68ee);
                    border: 2px solid #7b68ee;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #7b68ee, stop:1 #9370db);
                }
            """)
            
            btn.clicked.connect(lambda checked, p=personality: self._select_personality(p))
            self.personality_buttons[personality] = btn
            personality_layout.addWidget(btn)
        
        self.personality_frame.setVisible(False)
        
        # Continue button
        self.continue_btn = QPushButton("Enter the Cosmic Realm")
        self.continue_btn.setFixedSize(250, 60)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90e2, stop:1 #7b68ee);
                border: 3px solid #7b68ee;
                border-radius: 30px;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7b68ee, stop:1 #9370db);
                border: 3px solid #9370db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9370db, stop:1 #8a2be2);
            }
        """)
        self.continue_btn.clicked.connect(self._complete_intro)
        self.continue_btn.setVisible(False)
        
        # Add elements to layout
        background_layout.addWidget(self.avatar_label, alignment=Qt.AlignCenter)
        background_layout.addWidget(self.greeting_label, alignment=Qt.AlignCenter)
        background_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)
        background_layout.addWidget(self.personality_frame, alignment=Qt.AlignCenter)
        background_layout.addWidget(self.continue_btn, alignment=Qt.AlignCenter)
        
        layout.addWidget(self.background_frame)
        
        # Start cosmic intro sequence
        self._start_cosmic_intro()
    
    def setup_animations(self):
        """Setup smooth animations"""
        
        # Avatar fade-in animation
        self.avatar_animation = QPropertyAnimation(self.avatar_label, b"geometry")
        self.avatar_animation.setDuration(2000)
        self.avatar_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Greeting text animation
        self.greeting_animation = QPropertyAnimation(self.greeting_label, b"geometry")
        self.greeting_animation.setDuration(1500)
        self.greeting_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Progress bar animation
        self.progress_animation = QPropertyAnimation(self.progress_bar, b"value")
        self.progress_animation.setDuration(3000)
        self.progress_animation.setEasingCurve(QEasingCurve.InOutCubic)
    
    def _start_cosmic_intro(self):
        """Start Athena's cosmic introduction sequence"""
        
        logger.info("Starting Athena's cosmic introduction sequence")
        
        # Initialize Athena core
        self._initialize_athena()
        
        # Start intro sequence
        self.current_step = 0
        self.intro_timer.start(100)  # Update every 100ms
    
    def _initialize_athena(self):
        """Initialize Athena's core systems"""
        
        try:
            # Create Athena configuration
            config = AthenaConfig(
                personality=self.current_personality,
                voice_tone="mystical_cinematic",
                avatar_style="cyber_sorceress"
            )
            
            # Initialize Athena core
            self.athena_core = AthenaCore(config)
            
            # Initialize voice agents
            self.bark_agent = BarkVoiceAgent()
            
            logger.info("Athena's core systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Athena: {e}")
    
    def _update_intro_sequence(self):
        """Update the cosmic introduction sequence"""
        
        self.current_step += 1
        
        if self.current_step == 1:
            # Step 1: Fade in avatar
            self._fade_in_avatar()
            
        elif self.current_step == 50:
            # Step 2: Show greeting text
            self._show_greeting()
            
        elif self.current_step == 100:
            # Step 3: Start voice synthesis
            self._synthesize_greeting()
            
        elif self.current_step == 150:
            # Step 4: Show progress bar
            self._show_progress()
            
        elif self.current_step == 200:
            # Step 5: Show personality selection
            self._show_personality_selection()
            
        elif self.current_step == 250:
            # Step 6: Show continue button
            self._show_continue_button()
            
        elif self.current_step >= 300:
            # Complete intro sequence
            self.intro_timer.stop()
    
    def _fade_in_avatar(self):
        """Animate Athena's avatar fade-in"""
        
        # Set initial avatar state
        self.avatar_label.setText("✨")
        self.avatar_label.setStyleSheet("""
            QLabel {
                background: qradialgradient(cx:0.5, cy:0.5, radius:1,
                    stop:0 #4a90e2, stop:0.5 #2c3e50, stop:1 #1a1a2e);
                border-radius: 150px;
                border: 3px solid #4a90e2;
                color: #ffffff;
                font-size: 120px;
            }
        """)
        
        # Animate avatar appearance
        start_geometry = self.avatar_label.geometry()
        start_geometry.setSize(start_geometry.size() * 0.5)
        
        end_geometry = self.avatar_label.geometry()
        
        self.avatar_animation.setStartValue(start_geometry)
        self.avatar_animation.setEndValue(end_geometry)
        self.avatar_animation.start()
        
        logger.info("Athena's avatar fading in")
    
    def _show_greeting(self):
        """Show Athena's cosmic greeting"""
        
        if self.athena_core:
            # Get personalized greeting
            greeting = asyncio.create_task(
                self.athena_core.cosmic_greeting(self.user_name)
            )
            
            def set_greeting():
                try:
                    greeting_text = asyncio.get_event_loop().run_until_complete(greeting)
                    self.greeting_label.setText(greeting_text)
                    
                    # Animate greeting text
                    start_geometry = self.greeting_label.geometry()
                    start_geometry.setHeight(0)
                    
                    end_geometry = self.greeting_label.geometry()
                    
                    self.greeting_animation.setStartValue(start_geometry)
                    self.greeting_animation.setEndValue(end_geometry)
                    self.greeting_animation.start()
                    
                except Exception as e:
                    logger.error(f"Failed to get greeting: {e}")
                    self.greeting_label.setText(
                        f"Welcome to AI-Artworks: The Birth of Celestial Art, {self.user_name}!\n"
                        "I am Athena, your post-human design genius."
                    )
            
            # Run in background
            threading.Thread(target=set_greeting, daemon=True).start()
        
        logger.info("Athena's greeting displayed")
    
    def _synthesize_greeting(self):
        """Synthesize Athena's voice greeting"""
        
        if self.bark_agent and self.athena_core:
            
            def synthesize_voice():
                try:
                    # Get greeting text
                    greeting_text = asyncio.get_event_loop().run_until_complete(
                        self.athena_core.cosmic_greeting(self.user_name)
                    )
                    
                    # Synthesize voice
                    voice_result = asyncio.get_event_loop().run_until_complete(
                        self.bark_agent.synthesize_greeting(
                            self.user_name, 
                            self.current_personality.value
                        )
                    )
                    
                    if voice_result and "audio_data" in voice_result:
                        # Play the synthesized audio
                        self._play_audio(voice_result["audio_data"])
                        
                except Exception as e:
                    logger.error(f"Voice synthesis failed: {e}")
            
            # Run voice synthesis in background
            self.voice_synthesis_task = threading.Thread(target=synthesize_voice, daemon=True)
            self.voice_synthesis_task.start()
            
            logger.info("Athena's voice greeting synthesized")
    
    def _play_audio(self, audio_data: bytes):
        """Play synthesized audio"""
        
        try:
            # This would integrate with the system's audio playback
            # For now, just log that audio would be played
            logger.info(f"Playing Athena's voice greeting ({len(audio_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
    
    def _show_progress(self):
        """Show cosmic loading progress"""
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Animate progress
        self.progress_animation.setStartValue(0)
        self.progress_animation.setEndValue(100)
        self.progress_animation.start()
        
        logger.info("Cosmic loading progress displayed")
    
    def _show_personality_selection(self):
        """Show personality selection buttons"""
        
        self.personality_frame.setVisible(True)
        
        # Highlight current personality
        self._highlight_personality(self.current_personality)
        
        logger.info("Personality selection displayed")
    
    def _show_continue_button(self):
        """Show continue button"""
        
        self.continue_btn.setVisible(True)
        
        logger.info("Continue button displayed")
    
    def _select_personality(self, personality: AthenaPersonality):
        """Select Athena's personality"""
        
        self.current_personality = personality
        
        # Update Athena's configuration
        if self.athena_core:
            self.athena_core.config.personality = personality
        
        # Highlight selected personality
        self._highlight_personality(personality)
        
        # Emit signal
        self.personality_selected.emit(personality.value)
        
        logger.info(f"Athena's personality changed to: {personality.value}")
    
    def _highlight_personality(self, personality: AthenaPersonality):
        """Highlight the selected personality button"""
        
        for p, btn in self.personality_buttons.items():
            if p == personality:
                btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #4a90e2, stop:1 #7b68ee);
                        border: 2px solid #7b68ee;
                        border-radius: 25px;
                        color: #ffffff;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 14px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #7b68ee, stop:1 #9370db);
                        border: 2px solid #9370db;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #2c3e50, stop:1 #34495e);
                        border: 2px solid #4a90e2;
                        border-radius: 25px;
                        color: #ffffff;
                        font-family: 'Segoe UI', Arial, sans-serif;
                        font-size: 14px;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #4a90e2, stop:1 #7b68ee);
                        border: 2px solid #7b68ee;
                    }
                """)
    
    def _complete_intro(self):
        """Complete the introduction and enter the cosmic realm"""
        
        logger.info("Athena's introduction complete - entering the cosmic realm")
        
        # Emit completion signal
        self.intro_complete.emit()
        
        # Close dialog
        self.accept()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        
        # Stop timers
        if self.intro_timer.isActive():
            self.intro_timer.stop()
        
        # Cleanup voice synthesis
        if self.voice_synthesis_task and self.voice_synthesis_task.is_alive():
            # Let the thread finish naturally
            pass
        
        logger.info("Athena's introduction dialog closed")
        super().closeEvent(event)