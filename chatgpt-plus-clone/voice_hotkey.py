"""
Voice Hotkey Listener - Global Hotkey Management
Handles global hotkey registration and voice activation
"""

import keyboard
import threading
import logging
import time
from typing import Optional, Callable
from config_manager import ConfigManager

class VoiceHotkeyListener:
    """Global hotkey listener for voice activation"""
    
    def __init__(self, hotkey: str = "ctrl+shift+v", voice_callback: Optional[Callable] = None):
        self.hotkey = hotkey
        self.voice_callback = voice_callback
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager()
        
        # Threading
        self.is_running = False
        self.listener_thread = None
        
        # Voice session state
        self.is_recording = False
        self.recording_thread = None
        
        # Load voice settings from config
        self.load_voice_settings()
    
    def load_voice_settings(self):
        """Load voice settings from config"""
        voice_settings = self.config.get_voice_settings()
        self.hotkey = voice_settings.get('hotkey', self.hotkey)
        self.sample_rate = voice_settings.get('sample_rate', 16000)
        self.recording_duration = voice_settings.get('recording_duration', 5)
        self.silence_threshold = voice_settings.get('silence_threshold', 10)
        self.enabled = voice_settings.get('enabled', True)
    
    def start(self):
        """Start the hotkey listener"""
        if not self.enabled:
            self.logger.info("Voice hotkey disabled in config")
            return
        
        if self.is_running:
            self.logger.warning("Voice hotkey listener already running")
            return
        
        try:
            # Register the hotkey
            keyboard.add_hotkey(self.hotkey, self._on_hotkey_pressed)
            self.is_running = True
            
            # Start listener thread
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
            
            self.logger.info(f"🎙️ Voice hotkey listener started - Press {self.hotkey} to speak")
            
        except Exception as e:
            self.logger.error(f"Failed to start voice hotkey listener: {e}")
    
    def stop(self):
        """Stop the hotkey listener"""
        if not self.is_running:
            return
        
        try:
            # Unregister hotkey
            keyboard.remove_hotkey(self.hotkey)
            self.is_running = False
            
            # Stop any active recording
            if self.is_recording:
                self._stop_recording()
            
            self.logger.info("Voice hotkey listener stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping voice hotkey listener: {e}")
    
    def _listen_loop(self):
        """Main listener loop"""
        while self.is_running:
            try:
                time.sleep(0.1)  # Small delay to prevent high CPU usage
            except KeyboardInterrupt:
                break
        
        self.logger.debug("Voice hotkey listener loop ended")
    
    def _on_hotkey_pressed(self):
        """Handle hotkey press with enhanced robustness"""
        # Prevent multiple simultaneous recordings
        if self.is_recording:
            self.logger.debug("Already recording, ignoring hotkey press")
            return
        
        # Check if we're in a valid state
        if not self.is_running:
            self.logger.warning("Voice listener not running, ignoring hotkey")
            return
        
        # Add debouncing to prevent rapid-fire presses
        current_time = time.time()
        if hasattr(self, '_last_press_time'):
            if current_time - self._last_press_time < 0.5:  # 500ms debounce
                self.logger.debug("Hotkey press debounced")
                return
        
        self._last_press_time = current_time
        
        # Start recording in a separate thread with error handling
        try:
            self.recording_thread = threading.Thread(
                target=self._start_voice_session, 
                daemon=True,
                name=f"VoiceSession-{current_time}"
            )
            self.recording_thread.start()
            
            # Set a timeout for the recording thread
            self.recording_timeout = threading.Timer(30.0, self._timeout_recording)
            self.recording_timeout.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start voice session: {e}")
            self.is_recording = False
    
    def _timeout_recording(self):
        """Timeout handler for recording sessions"""
        if self.is_recording:
            self.logger.warning("Voice recording session timed out")
            self._stop_recording()
    
    def _stop_recording(self):
        """Stop current recording with enhanced cleanup"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Cancel timeout timer
        if hasattr(self, 'recording_timeout'):
            self.recording_timeout.cancel()
        
        # Wait for recording thread to finish (with timeout)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
            if self.recording_thread.is_alive():
                self.logger.warning("Recording thread did not finish gracefully")
        
        self.logger.info("Voice recording stopped")
    
    def _start_voice_session(self):
        """Start a voice recording session"""
        try:
            self.is_recording = True
            self.logger.info("🎤 Starting voice session...")
            
            # Visual feedback (could be enhanced with UI notifications)
            print(f"\n🎤 Voice session started - Press {self.hotkey} again to stop")
            
            # Import voice agent here to avoid circular imports
            try:
                from voice_agent import VoiceAgent
                voice_agent = VoiceAgent()
                
                # Record audio
                audio_data = voice_agent.record_audio(duration=self.recording_duration)
                
                if audio_data is not None:
                    # Transcribe
                    text = voice_agent.transcribe(audio_data)
                    
                    if text and text.strip():
                        self.logger.info(f"🎯 Transcribed: {text}")
                        
                        # Call callback if provided
                        if self.voice_callback:
                            self.voice_callback(text)
                        else:
                            # Default behavior: speak the transcription back
                            voice_agent.speak(f"You said: {text}")
                    else:
                        self.logger.info("No speech detected")
                        if self.config.get_voice_settings().get('tts_enabled', True):
                            voice_agent.speak("No speech detected")
                else:
                    self.logger.warning("Failed to record audio")
                    
            except ImportError:
                self.logger.error("Voice agent not available")
                print("❌ Voice agent not available - install required dependencies")
            except Exception as e:
                self.logger.error(f"Error in voice session: {e}")
                print(f"❌ Voice session error: {e}")
            
        finally:
            self.is_recording = False
            self.logger.info("🎤 Voice session ended")
    
    def _stop_recording(self):
        """Stop current recording"""
        self.is_recording = False
        self.logger.info("Stopping voice recording")
    
    def set_voice_callback(self, callback: Callable):
        """Set callback function for voice commands"""
        self.voice_callback = callback
        self.logger.info("Voice callback set")
    
    def update_settings(self):
        """Reload settings from config"""
        self.load_voice_settings()
        
        # Restart if settings changed
        if self.is_running:
            self.stop()
            time.sleep(0.1)  # Small delay
            self.start()
    
    def get_status(self) -> dict:
        """Get current status"""
        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "recording": self.is_recording,
            "hotkey": self.hotkey,
            "sample_rate": self.sample_rate,
            "recording_duration": self.recording_duration
        }

# Global instance for easy access
_voice_listener = None

def get_voice_listener() -> Optional[VoiceHotkeyListener]:
    """Get the global voice listener instance"""
    return _voice_listener

def start_voice_listener(callback: Optional[Callable] = None) -> VoiceHotkeyListener:
    """Start the global voice listener"""
    global _voice_listener
    
    if _voice_listener is None:
        _voice_listener = VoiceHotkeyListener()
    
    if callback:
        _voice_listener.set_voice_callback(callback)
    
    _voice_listener.start()
    return _voice_listener

def stop_voice_listener():
    """Stop the global voice listener"""
    global _voice_listener
    
    if _voice_listener:
        _voice_listener.stop()
        _voice_listener = None

def update_voice_settings():
    """Update voice settings from config"""
    global _voice_listener
    
    if _voice_listener:
        _voice_listener.update_settings()

if __name__ == "__main__":
    # Test the voice hotkey listener
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Voice Hotkey Listener...")
    print("Press Ctrl+Shift+V to test voice recording")
    print("Press Ctrl+C to exit")
    
    try:
        listener = VoiceHotkeyListener()
        listener.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping voice listener...")
        if listener:
            listener.stop()
        print("✅ Voice listener stopped")