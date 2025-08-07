import asyncio
import sounddevice as sd
import numpy as np
import logging
import time
import queue
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from threading import Thread

logger = logging.getLogger("AsyncVoiceInput")

@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float

@dataclass
class VoiceSession:
    is_active: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    audio_chunks: list = None
    transcription: str = ""
    confidence: float = 0.0

class AsyncVoiceInput:
    def __init__(self, callback: Callable, samplerate: int = 16000, channels: int = 1, 
                 blocksize: int = 1024, silence_threshold: float = 0.01, 
                 silence_duration: float = 2.0, max_session_duration: float = 30.0):
        self.callback = callback
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_session_duration = max_session_duration
        
        # Audio processing state
        self.audio_queue = queue.Queue()
        self.stream = None
        self.loop = None
        self.is_running = False
        
        # Voice session management
        self.current_session = VoiceSession()
        self.last_audio_time = 0.0
        self.vad = VoiceActivityDetector(silence_threshold, silence_duration)
        
        # Threading
        self.audio_thread = None
        self.processing_task = None

    async def start(self):
        """Start async voice input processing"""
        if self.is_running:
            logger.warning("AsyncVoiceInput already running")
            return
        
        self.is_running = True
        self.loop = asyncio.get_running_loop()
        
        # Start audio capture in separate thread
        self.audio_thread = Thread(target=self._audio_capture_thread, daemon=True)
        self.audio_thread.start()
        
        # Start audio processing loop
        self.processing_task = asyncio.create_task(self._process_audio_loop())
        
        logger.info("AsyncVoiceInput started")

    async def stop(self):
        """Stop async voice input processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # End current session if active
        if self.current_session.is_active:
            await self._end_session(time.time())
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AsyncVoiceInput stopped")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio capture - runs in audio thread"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Create audio chunk with metadata
        chunk = AudioChunk(
            data=indata.copy(),
            timestamp=time.time(),
            sample_rate=self.samplerate,
            channels=self.channels,
            duration=frames / self.samplerate
        )
        
        # Put chunk in queue for processing
        self.audio_queue.put(chunk)

    def _audio_capture_thread(self):
        """Audio capture thread - runs sounddevice stream"""
        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._audio_callback,
                dtype='float32'
            ) as stream:
                self.stream = stream
                while self.is_running:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
        except Exception as e:
            logger.error(f"Audio capture thread error: {e}")

    async def _process_audio_loop(self):
        """Main audio processing loop - runs in asyncio event loop"""
        while self.is_running:
            try:
                # Get audio chunk from queue with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                await self._process_audio_chunk(chunk)
            except queue.Empty:
                # Check for session timeout when no audio
                await self._check_session_timeout()
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.01)

    async def _process_audio_chunk(self, chunk: AudioChunk):
        """Process individual audio chunk for voice activity detection"""
        # Update last audio time
        self.last_audio_time = chunk.timestamp
        
        # Check for voice activity
        is_speech = self.vad.detect_voice_activity(chunk.data)
        
        if is_speech and not self.current_session.is_active:
            # Start new voice session
            await self._start_session(chunk.timestamp)
        
        if self.current_session.is_active:
            # Add chunk to current session
            if self.current_session.audio_chunks is None:
                self.current_session.audio_chunks = []
            self.current_session.audio_chunks.append(chunk)
            
            # Check if session should end (silence or max duration)
            if not is_speech:
                silence_elapsed = chunk.timestamp - self.last_audio_time
                if silence_elapsed >= self.silence_duration:
                    await self._end_session(chunk.timestamp)
            else:
                # Check max session duration
                session_duration = chunk.timestamp - self.current_session.start_time
                if session_duration >= self.max_session_duration:
                    await self._end_session(chunk.timestamp)

    async def _start_session(self, timestamp: float):
        """Start a new voice session"""
        self.current_session = VoiceSession(
            is_active=True,
            start_time=timestamp,
            audio_chunks=[]
        )
        
        # Notify callback of session start
        await self.callback({
            "event": "session_start",
            "timestamp": timestamp,
            "session": self.current_session
        })
        
        logger.info("Voice session started")

    async def _end_session(self, timestamp: float):
        """End current voice session and process audio"""
        if not self.current_session.is_active:
            return
        
        self.current_session.is_active = False
        self.current_session.end_time = timestamp
        
        # Process session audio
        await self._process_session_audio()

    async def _process_session_audio(self):
        """Process completed voice session audio"""
        if not self.current_session.audio_chunks:
            return
        
        # Concatenate all audio chunks
        audio_data = []
        for chunk in self.current_session.audio_chunks:
            audio_data.append(chunk.data.flatten())
        
        if audio_data:
            full_audio = np.concatenate(audio_data)
            
            # Notify callback with session complete event
            await self.callback({
                "event": "session_complete",
                "audio_data": full_audio,
                "sample_rate": self.samplerate,
                "session": self.current_session
            })
            
            logger.info(f"Voice session completed: {len(full_audio)} samples")

    async def _check_session_timeout(self):
        """Check if current session should timeout due to silence"""
        if self.current_session.is_active:
            current_time = time.time()
            silence_elapsed = current_time - self.last_audio_time
            
            if silence_elapsed >= self.silence_duration:
                await self._end_session(current_time)

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "is_active": self.current_session.is_active,
            "start_time": self.current_session.start_time,
            "duration": time.time() - self.current_session.start_time if self.current_session.start_time else 0,
            "chunk_count": len(self.current_session.audio_chunks) if self.current_session.audio_chunks else 0
        }

    def set_parameters(self, silence_threshold: Optional[float] = None, 
                      silence_duration: Optional[float] = None, 
                      max_session_duration: Optional[float] = None):
        """Update VAD parameters"""
        if silence_threshold is not None:
            self.silence_threshold = silence_threshold
            self.vad.silence_threshold = silence_threshold
        
        if silence_duration is not None:
            self.silence_duration = silence_duration
            self.vad.silence_duration = silence_duration
        
        if max_session_duration is not None:
            self.max_session_duration = max_session_duration

class VoiceActivityDetector:
    """Advanced voice activity detection"""
    
    def __init__(self, silence_threshold: float = 0.01, silence_duration: float = 2.0):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.last_speech_time = 0.0
        self.speech_buffer = []
    
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        # Calculate RMS (Root Mean Square) of audio chunk
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Simple threshold-based VAD
        is_speech = rms > self.silence_threshold
        
        # Update speech buffer for more robust detection
        self.speech_buffer.append(is_speech)
        if len(self.speech_buffer) > 10:  # Keep last 10 chunks
            self.speech_buffer.pop(0)
        
        # Consider speech if majority of recent chunks contain speech
        speech_ratio = sum(self.speech_buffer) / len(self.speech_buffer)
        return speech_ratio > 0.5