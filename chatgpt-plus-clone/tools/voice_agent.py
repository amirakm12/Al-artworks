"""
Voice Agent - Whisper + TTS Pipeline
Provides speech-to-text and text-to-speech capabilities
"""

import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import os
import time
from typing import Optional, Callable
import logging

# TTS imports
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS not available - using fallback")

try:
    import bark
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logging.warning("Bark not available - using fallback")

class VoiceAgent:
    """Main voice agent with STT and TTS capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_speaking = False
        
        # Initialize models
        self.setup_models()
        
        # Recording settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5
        self.silence_threshold = 0.01
        self.max_recording_time = 30
        
        # Callbacks
        self.on_transcription = None
        self.on_speech_start = None
        self.on_speech_end = None
    
    def setup_models(self):
        """Initialize Whisper and TTS models"""
        try:
            # Load Whisper model
            self.whisper_model = whisper.load_model("base")
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading Whisper: {e}")
            self.whisper_model = None
        
        try:
            # Load TTS model
            if TTS_AVAILABLE:
                self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
                self.logger.info("✅ TTS model loaded successfully")
            elif BARK_AVAILABLE:
                self.tts = "bark"
                self.logger.info("✅ Bark TTS available")
            else:
                self.tts = None
                self.logger.warning("⚠️ No TTS available - using fallback")
        except Exception as e:
            self.logger.error(f"❌ Error loading TTS: {e}")
            self.tts = None
    
    def audio_callback(self, indata, frames, time, status):
        """Audio recording callback"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Check for silence
            volume_norm = np.linalg.norm(indata) * 10
            if volume_norm > self.silence_threshold:
                self.audio_queue.put(indata.copy())
    
    def record_audio(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        try:
            self.is_recording = True
            self.audio_queue = queue.Queue()
            
            # Calculate chunk size
            chunk_size = int(self.sample_rate * self.chunk_duration)
            
            with sd.InputStream(callback=self.audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=chunk_size):
                
                self.logger.info("🎙️ Recording... (speak now)")
                start_time = time.time()
                
                # Record for specified duration
                while time.time() - start_time < duration:
                    time.sleep(self.chunk_duration)
                
                self.logger.info("🎙️ Recording stopped")
            
            self.is_recording = False
            
            # Collect recorded audio
            audio_chunks = []
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())
            
            if audio_chunks:
                return np.concatenate(audio_chunks, axis=0)
            else:
                self.logger.warning("⚠️ No audio recorded")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error recording audio: {e}")
            self.is_recording = False
            return None
    
    def transcribe(self, audio_data: Optional[np.ndarray] = None) -> str:
        """Transcribe audio to text using Whisper"""
        try:
            if audio_data is None:
                # Record audio if not provided
                audio_data = self.record_audio()
            
            if audio_data is None or len(audio_data) == 0:
                return ""
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            # Save audio data
            sd.write(temp_audio_path, audio_data, self.sample_rate)
            
            # Transcribe with Whisper
            if self.whisper_model:
                result = self.whisper_model.transcribe(temp_audio_path)
                transcription = result["text"].strip()
                
                # Cleanup
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                
                self.logger.info(f"🎤 Transcribed: '{transcription}'")
                
                # Call callback if set
                if self.on_transcription:
                    self.on_transcription(transcription)
                
                return transcription
            else:
                self.logger.error("❌ Whisper model not available")
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Error transcribing audio: {e}")
            return ""
    
    def speak(self, text: str) -> bool:
        """Convert text to speech and play it"""
        try:
            if not text.strip():
                return False
            
            self.logger.info(f"🗣️ Speaking: '{text}'")
            
            if self.on_speech_start:
                self.on_speech_start(text)
            
            self.is_speaking = True
            
            if self.tts == "bark" and BARK_AVAILABLE:
                # Use Bark for TTS
                audio_array = bark.generate_audio(text)
                sd.play(audio_array, samplerate=24000)
                sd.wait()
                
            elif self.tts and TTS_AVAILABLE:
                # Use TTS library
                wav = self.tts.tts(text)
                sd.play(wav, samplerate=22050)
                sd.wait()
                
            else:
                # Fallback - just print the text
                self.logger.warning("⚠️ No TTS available - printing text")
                print(f"🗣️ [TTS] {text}")
                time.sleep(len(text) * 0.1)  # Simulate speaking time
            
            self.is_speaking = False
            
            if self.on_speech_end:
                self.on_speech_end(text)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error speaking text: {e}")
            self.is_speaking = False
            return False
    
    def voice_command_loop(self, callback: Callable[[str], str]):
        """Continuous voice command loop"""
        self.logger.info("🎤 Starting voice command loop...")
        
        try:
            while True:
                # Record and transcribe
                transcription = self.transcribe()
                
                if transcription:
                    # Process with callback
                    response = callback(transcription)
                    
                    if response:
                        # Speak response
                        self.speak(response)
                
                # Small delay between commands
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("🎤 Voice command loop stopped")
        except Exception as e:
            self.logger.error(f"❌ Error in voice command loop: {e}")
    
    def set_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback for transcription events"""
        self.on_transcription = callback
    
    def set_speech_callbacks(self, start_callback: Callable[[str], None], 
                           end_callback: Callable[[str], None]):
        """Set callbacks for speech events"""
        self.on_speech_start = start_callback
        self.on_speech_end = end_callback
    
    def is_available(self) -> bool:
        """Check if voice agent is available"""
        return self.whisper_model is not None
    
    def get_status(self) -> dict:
        """Get voice agent status"""
        return {
            "whisper_available": self.whisper_model is not None,
            "tts_available": self.tts is not None,
            "is_recording": self.is_recording,
            "is_speaking": self.is_speaking,
            "sample_rate": self.sample_rate
        }

# Global voice agent instance
voice_agent = VoiceAgent()

def transcribe():
    """Global transcribe function"""
    return voice_agent.transcribe()

def speak(text: str):
    """Global speak function"""
    return voice_agent.speak(text)

def record_audio(duration: float = 5.0):
    """Global record audio function"""
    return voice_agent.record_audio(duration)

def voice_command_loop(callback: Callable[[str], str]):
    """Global voice command loop"""
    return voice_agent.voice_command_loop(callback)

if __name__ == "__main__":
    # Test voice agent
    print("🎤 Testing Voice Agent...")
    
    # Test recording and transcription
    print("Speak something for 5 seconds...")
    audio = record_audio(5)
    
    if audio is not None:
        text = transcribe()
        print(f"Transcribed: '{text}'")
        
        # Test TTS
        if text:
            speak(f"You said: {text}")
    else:
        print("No audio recorded")