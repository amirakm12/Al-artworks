"""
Voice Hotkey System - Global Voice Assistant Activation
Provides global hotkey support for voice interaction with Whisper integration
"""

import keyboard
import threading
import time
import logging
from typing import Optional, Callable
from pathlib import Path
import queue
import tempfile
import os

# Voice processing imports
try:
    import whisper
    import sounddevice as sd
    import numpy as np
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("Voice processing libraries not available")

class VoiceHotkeyManager:
    """Manages global voice hotkey functionality"""
    
    def __init__(self, hotkey: str = "ctrl+shift+v"):
        self.hotkey = hotkey
        self.is_listening = False
        self.voice_thread = None
        self.audio_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
        # Voice processing setup
        if VOICE_AVAILABLE:
            self.whisper_model = None
            self.setup_whisper()
        
        # Callback for voice input
        self.voice_callback = None
        
        # Recording settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # seconds
        self.silence_threshold = 0.01
        self.max_recording_time = 30  # seconds
    
    def setup_whisper(self):
        """Initialize Whisper model"""
        try:
            self.whisper_model = whisper.load_model("base")
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def set_voice_callback(self, callback: Callable[[str], None]):
        """Set callback for voice input processing"""
        self.voice_callback = callback
    
    def start_listening(self):
        """Start listening for the hotkey"""
        try:
            keyboard.add_hotkey(self.hotkey, self.activate_voice_session)
            self.is_listening = True
            self.logger.info(f"Voice hotkey listener started: {self.hotkey}")
            
            # Keep the main thread alive
            keyboard.wait()
            
        except Exception as e:
            self.logger.error(f"Error starting voice hotkey listener: {e}")
    
    def stop_listening(self):
        """Stop listening for the hotkey"""
        try:
            keyboard.remove_hotkey(self.hotkey)
            self.is_listening = False
            self.logger.info("Voice hotkey listener stopped")
        except Exception as e:
            self.logger.error(f"Error stopping voice hotkey listener: {e}")
    
    def activate_voice_session(self):
        """Activate voice recording session"""
        if self.voice_thread and self.voice_thread.is_alive():
            self.logger.info("Voice session already active")
            return
        
        self.voice_thread = threading.Thread(target=self.record_voice)
        self.voice_thread.daemon = True
        self.voice_thread.start()
    
    def record_voice(self):
        """Record voice input and convert to text"""
        try:
            self.logger.info("Starting voice recording...")
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            # Record audio
            audio_data = self.record_audio()
            
            if audio_data is None or len(audio_data) == 0:
                self.logger.warning("No audio recorded")
                return
            
            # Save audio to file
            self.save_audio(audio_data, temp_audio_path)
            
            # Convert to text using Whisper
            if self.whisper_model and VOICE_AVAILABLE:
                text = self.transcribe_audio(temp_audio_path)
                
                if text and text.strip():
                    self.logger.info(f"Transcribed: {text}")
                    
                    # Call callback with transcribed text
                    if self.voice_callback:
                        self.voice_callback(text.strip())
                    else:
                        print(f"Voice input: {text.strip()}")
                else:
                    self.logger.info("No speech detected")
            else:
                self.logger.warning("Whisper not available - cannot transcribe audio")
            
            # Cleanup
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Error in voice recording: {e}")
    
    def record_audio(self) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        try:
            # Calculate chunk size
            chunk_size = int(self.sample_rate * self.chunk_duration)
            
            # Initialize recording
            audio_chunks = []
            silence_frames = 0
            max_silence_frames = int(2.0 / self.chunk_duration)  # 2 seconds of silence
            
            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.warning(f"Audio callback status: {status}")
                
                # Check for silence
                volume_norm = np.linalg.norm(indata) * 10
                if volume_norm < self.silence_threshold:
                    nonlocal silence_frames
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                audio_chunks.append(indata.copy())
            
            # Start recording
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=chunk_size):
                
                self.logger.info("Recording... (speak now)")
                start_time = time.time()
                
                while (time.time() - start_time < self.max_recording_time and 
                       silence_frames < max_silence_frames):
                    time.sleep(self.chunk_duration)
                
                self.logger.info("Recording stopped")
            
            if audio_chunks:
                return np.concatenate(audio_chunks, axis=0)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return None
    
    def save_audio(self, audio_data: np.ndarray, file_path: str):
        """Save audio data to file"""
        try:
            import wave
            
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper"""
        try:
            if not self.whisper_model:
                return ""
            
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()

# Global voice hotkey manager
voice_manager = VoiceHotkeyManager()

def start_voice_session():
    """Start a voice recording session"""
    voice_manager.activate_voice_session()

def set_voice_callback(callback: Callable[[str], None]):
    """Set callback for voice input"""
    voice_manager.set_voice_callback(callback)

def start_voice_hotkey_listener():
    """Start the global voice hotkey listener"""
    voice_manager.start_listening()

def stop_voice_hotkey_listener():
    """Stop the global voice hotkey listener"""
    voice_manager.stop_listening()

def listen_for_hotkey():
    """Legacy function for backward compatibility"""
    print(f"[Voice Assistant] Hotkey listener active — Press {voice_manager.hotkey} to speak.")
    start_voice_hotkey_listener()

if __name__ == "__main__":
    # Test voice hotkey functionality
    print("Starting voice hotkey listener...")
    print(f"Press {voice_manager.hotkey} to activate voice input")
    print("Press Ctrl+C to exit")
    
    try:
        # Set a simple callback for testing
        def test_callback(text):
            print(f"Voice input received: {text}")
        
        voice_manager.set_voice_callback(test_callback)
        voice_manager.start_listening()
        
    except KeyboardInterrupt:
        print("\nStopping voice hotkey listener...")
        voice_manager.cleanup()