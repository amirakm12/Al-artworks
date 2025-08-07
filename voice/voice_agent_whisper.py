import asyncio
import logging
import torch
import whisper
import sounddevice as sd
import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

log = logging.getLogger("VoiceAgent")

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    timestamp: float

class AsyncVoiceAgent:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.running = False
        self.model = None
        self.stream = None
        self.buffer = None
        self.buffer_size = 16000 * 5  # 5 seconds buffer
        self.sample_rate = 16000
        self.channels = 1
        
        # Callbacks
        self.on_transcription: Optional[Callable] = None
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        
        # State
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.speech_timeout = 2.0  # seconds of silence to end speech
        
        log.info(f"Initializing Whisper voice agent with model size: {model_size}")
        self._load_model()

    def _load_model(self):
        """Load Whisper model on the specified device"""
        try:
            log.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            log.info("Whisper model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load Whisper model: {e}")
            raise

    def set_callbacks(self, on_transcription: Optional[Callable] = None,
                     on_speech_start: Optional[Callable] = None,
                     on_speech_end: Optional[Callable] = None):
        """Set callback functions for voice events"""
        self.on_transcription = on_transcription
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio capture"""
        if status:
            log.warning(f"Audio callback status: {status}")
        
        # Add audio data to buffer
        if self.buffer is not None:
            # Roll buffer and add new data
            self.buffer = np.roll(self.buffer, -frames)
            self.buffer[-frames:] = indata[:, 0]

    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio using Whisper"""
        if self.model is None:
            log.error("Whisper model not loaded")
            return None
        
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.model.transcribe, audio_data)
            
            text = result.get("text", "").strip()
            if text:
                confidence = result.get("avg_logprob", 0.0)
                language = result.get("language", "unknown")
                
                transcription = TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language=language,
                    timestamp=asyncio.get_event_loop().time()
                )
                
                log.info(f"Transcription: '{text}' (confidence: {confidence:.2f})")
                return transcription
            
        except Exception as e:
            log.error(f"Transcription failed: {e}")
        
        return None

    async def _process_audio_buffer(self):
        """Process the audio buffer for speech detection and transcription"""
        if self.buffer is None:
            return
        
        # Calculate audio level (RMS)
        rms = np.sqrt(np.mean(self.buffer**2))
        current_time = asyncio.get_event_loop().time()
        
        # Simple speech detection based on audio level
        is_speaking_now = rms > 0.01  # Adjust threshold as needed
        
        if is_speaking_now and not self.is_speaking:
            # Speech started
            self.is_speaking = True
            self.last_speech_time = current_time
            
            if self.on_speech_start:
                try:
                    await self.on_speech_start(current_time)
                except Exception as e:
                    log.error(f"Error in speech start callback: {e}")
            
            log.info("Speech detected")
        
        elif is_speaking_now:
            # Continue speaking
            self.last_speech_time = current_time
        
        elif self.is_speaking and (current_time - self.last_speech_time) > self.speech_timeout:
            # Speech ended
            self.is_speaking = False
            
            if self.on_speech_end:
                try:
                    await self.on_speech_end(current_time)
                except Exception as e:
                    log.error(f"Error in speech end callback: {e}")
            
            # Transcribe the speech
            transcription = await self._transcribe_audio(self.buffer)
            if transcription and self.on_transcription:
                try:
                    await self.on_transcription(transcription)
                except Exception as e:
                    log.error(f"Error in transcription callback: {e}")
            
            log.info("Speech ended, transcription completed")

    async def start(self):
        """Start the async voice agent with real-time audio processing"""
        if self.running:
            log.warning("Voice agent already running")
            return
        
        self.running = True
        log.info("Starting voice agent with real-time audio processing")
        
        try:
            # Initialize audio buffer
            self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
            
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                callback=self._audio_callback,
                blocksize=1600  # 0.1 seconds
            ) as stream:
                self.stream = stream
                log.info("Audio stream started")
                
                # Main processing loop
                while self.running:
                    await self._process_audio_buffer()
                    await asyncio.sleep(0.1)  # Process every 100ms
                    
        except Exception as e:
            log.error(f"Voice agent error: {e}")
        finally:
            self.running = False
            log.info("Voice agent stopped")

    async def stop(self):
        """Stop the async voice agent"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        log.info("Voice agent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current voice agent status"""
        return {
            "running": self.running,
            "is_speaking": self.is_speaking,
            "model_size": self.model_size,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "speech_timeout": self.speech_timeout
        }

    def set_parameters(self, speech_timeout: Optional[float] = None,
                      buffer_size: Optional[int] = None):
        """Update voice agent parameters"""
        if speech_timeout is not None:
            self.speech_timeout = speech_timeout
        
        if buffer_size is not None:
            self.buffer_size = buffer_size
            if self.buffer is not None:
                self.buffer = np.zeros(self.buffer_size, dtype=np.int16)

    async def transcribe_file(self, audio_file_path: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file"""
        if self.model is None:
            log.error("Whisper model not loaded")
            return None
        
        try:
            log.info(f"Transcribing audio file: {audio_file_path}")
            
            # Run transcription in thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.model.transcribe, audio_file_path)
            
            text = result.get("text", "").strip()
            if text:
                confidence = result.get("avg_logprob", 0.0)
                language = result.get("language", "unknown")
                
                transcription = TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language=language,
                    timestamp=asyncio.get_event_loop().time()
                )
                
                log.info(f"File transcription: '{text}' (confidence: {confidence:.2f})")
                return transcription
            
        except Exception as e:
            log.error(f"File transcription failed: {e}")
        
        return None

# Example callbacks
async def example_transcription_callback(transcription: TranscriptionResult):
    """Example callback for transcription results"""
    log.info(f"Transcription received: '{transcription.text}'")
    log.info(f"Confidence: {transcription.confidence:.2f}")
    log.info(f"Language: {transcription.language}")

async def example_speech_start_callback(timestamp: float):
    """Example callback for speech start"""
    log.info(f"Speech started at {timestamp}")

async def example_speech_end_callback(timestamp: float):
    """Example callback for speech end"""
    log.info(f"Speech ended at {timestamp}")

# Example usage
async def main():
    """Example main function showing how to use AsyncVoiceAgent with Whisper"""
    logging.basicConfig(level=logging.INFO)
    
    # Create voice agent
    agent = AsyncVoiceAgent(model_size="base")
    
    # Set callbacks
    agent.set_callbacks(
        on_transcription=example_transcription_callback,
        on_speech_start=example_speech_start_callback,
        on_speech_end=example_speech_end_callback
    )
    
    try:
        log.info("Starting voice agent... Press Ctrl+C to stop")
        await agent.start()
    except KeyboardInterrupt:
        log.info("Stopping voice agent...")
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())