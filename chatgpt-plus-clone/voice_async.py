"""
Async Voice Processing System for ChatGPT+ Clone
Non-blocking voice capture and processing for smooth UI experience
"""

import asyncio
import threading
import queue
import time
import logging
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import whisper
from device_utils import get_best_device, safe_model_load, PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Audio data chunk with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int

@dataclass
class VoiceSession:
    """Voice session configuration and state"""
    is_active: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    audio_chunks: list = None
    transcription: str = ""
    confidence: float = 0.0

class AsyncVoiceProcessor:
    """Asynchronous voice processing with non-blocking audio capture"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_duration: float = 0.5,
                 silence_threshold: float = 0.01,
                 silence_duration: float = 2.0):
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        # Audio processing queues
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Voice session state
        self.current_session = VoiceSession()
        self.silence_start = None
        self.last_audio_time = None
        
        # Processing threads
        self.audio_thread = None
        self.processing_thread = None
        self.is_running = False
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize Whisper model
        self.whisper_model = None
        self.device = get_best_device()
        self._load_whisper_model()
        
        # Callbacks
        self.on_transcription_ready: Optional[Callable] = None
        self.on_session_start: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info(f"AsyncVoiceProcessor initialized on device: {self.device}")
    
    def _load_whisper_model(self):
        """Load Whisper model with device optimization"""
        try:
            logger.info("Loading Whisper model...")
            self.performance_monitor.start_timing()
            
            # Load model with device optimization
            self.whisper_model = whisper.load_model("base")
            self.device = safe_model_load(self.whisper_model, self.device)
            
            self.performance_monitor.end_timing("Whisper model loading")
            logger.info(f"Whisper model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            if self.on_error:
                self.on_error(f"Whisper model loading failed: {e}")
    
    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio callback for sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to float32 and normalize
        audio_data = indata.astype(np.float32)
        
        # Create audio chunk
        chunk = AudioChunk(
            data=audio_data.copy(),
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            channels=self.channels
        )
        
        # Add to processing queue
        self.audio_queue.put(chunk)
    
    def _audio_capture_thread(self):
        """Audio capture thread"""
        try:
            logger.info("Starting audio capture thread...")
            
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                blocksize=int(self.sample_rate * self.chunk_duration)
            ) as stream:
                
                logger.info("Audio stream started")
                
                while self.is_running:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                
                logger.info("Audio capture thread stopped")
                
        except Exception as e:
            logger.error(f"Audio capture thread error: {e}")
            if self.on_error:
                self.on_error(f"Audio capture failed: {e}")
    
    def _processing_thread(self):
        """Audio processing thread"""
        try:
            logger.info("Starting audio processing thread...")
            
            while self.is_running:
                try:
                    # Get audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Process the audio chunk
                    self._process_audio_chunk(chunk)
                    
                except queue.Empty:
                    # Check for session timeout
                    self._check_session_timeout()
                    continue
                    
        except Exception as e:
            logger.error(f"Audio processing thread error: {e}")
            if self.on_error:
                self.on_error(f"Audio processing failed: {e}")
    
    def _process_audio_chunk(self, chunk: AudioChunk):
        """Process individual audio chunk"""
        try:
            # Calculate audio level
            audio_level = np.sqrt(np.mean(chunk.data ** 2))
            
            # Update session state
            if not self.current_session.is_active:
                if audio_level > self.silence_threshold:
                    self._start_session(chunk.timestamp)
            
            if self.current_session.is_active:
                # Add chunk to session
                if self.current_session.audio_chunks is None:
                    self.current_session.audio_chunks = []
                
                self.current_session.audio_chunks.append(chunk)
                self.last_audio_time = chunk.timestamp
                
                # Check for silence
                if audio_level <= self.silence_threshold:
                    if self.silence_start is None:
                        self.silence_start = chunk.timestamp
                    elif chunk.timestamp - self.silence_start > self.silence_duration:
                        self._end_session(chunk.timestamp)
                else:
                    self.silence_start = None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _start_session(self, timestamp: float):
        """Start a new voice session"""
        logger.info("Starting voice session...")
        
        self.current_session = VoiceSession(
            is_active=True,
            start_time=timestamp,
            audio_chunks=[]
        )
        
        if self.on_session_start:
            self.on_session_start()
    
    def _end_session(self, timestamp: float):
        """End current voice session and transcribe"""
        if not self.current_session.is_active:
            return
        
        logger.info("Ending voice session...")
        
        self.current_session.is_active = False
        self.current_session.end_time = timestamp
        
        # Transcribe audio
        self._transcribe_session()
    
    def _transcribe_session(self):
        """Transcribe the current session"""
        if not self.current_session.audio_chunks:
            logger.warning("No audio chunks to transcribe")
            return
        
        try:
            logger.info("Starting transcription...")
            self.performance_monitor.start_timing()
            
            # Concatenate audio chunks
            audio_data = np.concatenate([chunk.data for chunk in self.current_session.audio_chunks])
            
            # Transcribe using Whisper
            if self.whisper_model:
                result = self.whisper_model.transcribe(audio_data)
                
                self.current_session.transcription = result["text"].strip()
                self.current_session.confidence = result.get("confidence", 0.0)
                
                self.performance_monitor.end_timing("Transcription")
                
                logger.info(f"Transcription completed: '{self.current_session.transcription}'")
                
                # Notify callback
                if self.on_transcription_ready:
                    self.on_transcription_ready(self.current_session.transcription)
                
                # End session callback
                if self.on_session_end:
                    self.on_session_end(self.current_session)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if self.on_error:
                self.on_error(f"Transcription failed: {e}")
    
    def _check_session_timeout(self):
        """Check for session timeout due to silence"""
        if (self.current_session.is_active and 
            self.last_audio_time and 
            time.time() - self.last_audio_time > self.silence_duration):
            
            self._end_session(time.time())
    
    def start_listening(self):
        """Start voice listening"""
        if self.is_running:
            logger.warning("Voice processor already running")
            return
        
        logger.info("Starting voice listening...")
        
        self.is_running = True
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()
        
        logger.info("Voice listening started")
    
    def stop_listening(self):
        """Stop voice listening"""
        if not self.is_running:
            return
        
        logger.info("Stopping voice listening...")
        
        self.is_running = False
        
        # End current session if active
        if self.current_session.is_active:
            self._end_session(time.time())
        
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Voice listening stopped")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if not self.current_session.is_active:
            return {"status": "inactive"}
        
        duration = time.time() - self.current_session.start_time
        chunk_count = len(self.current_session.audio_chunks) if self.current_session.audio_chunks else 0
        
        return {
            "status": "active",
            "duration": duration,
            "chunk_count": chunk_count,
            "silence_start": self.silence_start
        }
    
    def set_callbacks(self, 
                     on_transcription_ready: Optional[Callable] = None,
                     on_session_start: Optional[Callable] = None,
                     on_session_end: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set callback functions"""
        self.on_transcription_ready = on_transcription_ready
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        self.on_error = on_error

class VoiceHotkeyManager:
    """Manage global hotkey for voice activation"""
    
    def __init__(self, hotkey: str = "ctrl+shift+v"):
        self.hotkey = hotkey
        self.is_listening = False
        self.voice_processor = None
        self.on_hotkey_pressed: Optional[Callable] = None
        
        # Import keyboard only on Windows
        try:
            import keyboard
            self.keyboard = keyboard
            self.keyboard_available = True
        except ImportError:
            logger.warning("keyboard module not available (Linux/Mac)")
            self.keyboard_available = False
    
    def start_listening(self):
        """Start listening for hotkey"""
        if not self.keyboard_available:
            logger.warning("Hotkey listening not available on this platform")
            return
        
        try:
            self.keyboard.add_hotkey(self.hotkey, self._on_hotkey_pressed)
            self.is_listening = True
            logger.info(f"Hotkey listening started: {self.hotkey}")
        except Exception as e:
            logger.error(f"Failed to start hotkey listening: {e}")
    
    def stop_listening(self):
        """Stop listening for hotkey"""
        if not self.keyboard_available or not self.is_listening:
            return
        
        try:
            self.keyboard.remove_hotkey(self.hotkey)
            self.is_listening = False
            logger.info("Hotkey listening stopped")
        except Exception as e:
            logger.error(f"Failed to stop hotkey listening: {e}")
    
    def _on_hotkey_pressed(self):
        """Handle hotkey press"""
        logger.info(f"Hotkey pressed: {self.hotkey}")
        
        if self.on_hotkey_pressed:
            self.on_hotkey_pressed()
    
    def set_callback(self, callback: Callable):
        """Set hotkey callback"""
        self.on_hotkey_pressed = callback

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create voice processor
    processor = AsyncVoiceProcessor()
    
    # Set callbacks
    def on_transcription(text):
        print(f"Transcription: {text}")
    
    def on_session_start():
        print("Voice session started")
    
    def on_session_end(session):
        print(f"Voice session ended: {session.transcription}")
    
    processor.set_callbacks(
        on_transcription_ready=on_transcription,
        on_session_start=on_session_start,
        on_session_end=on_session_end
    )
    
    # Create hotkey manager
    hotkey_manager = VoiceHotkeyManager()
    
    def on_hotkey():
        if processor.is_running:
            processor.stop_listening()
            print("Voice listening stopped")
        else:
            processor.start_listening()
            print("Voice listening started")
    
    hotkey_manager.set_callback(on_hotkey)
    hotkey_manager.start_listening()
    
    print("Press Ctrl+Shift+V to toggle voice listening")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        processor.stop_listening()
        hotkey_manager.stop_listening()
        print("Exiting...")