"""
Async Voice Input - Non-blocking voice capture with sounddevice + asyncio
Real-time audio processing with voice activity detection
"""

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
    """Audio data chunk with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float

@dataclass
class VoiceSession:
    """Voice session configuration and state"""
    is_active: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    audio_chunks: list = None
    transcription: str = ""
    confidence: float = 0.0

class AsyncVoiceInput:
    """Asynchronous voice input with real-time processing"""
    
    def __init__(self, 
                 callback: Callable,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 blocksize: int = 1024,
                 silence_threshold: float = 0.01,
                 silence_duration: float = 2.0,
                 max_session_duration: float = 30.0):
        
        self.callback = callback  # Async function to process audio chunks
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_session_duration = max_session_duration
        
        # Audio processing state
        self.stream = None
        self.loop = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        
        # Voice session state
        self.current_session = VoiceSession()
        self.silence_start = None
        self.last_audio_time = None
        
        # Processing thread
        self.processing_thread = None
        
        logger.info(f"AsyncVoiceInput initialized: {sample_rate}Hz, {channels}ch, {blocksize} blocksize")
    
    async def start(self):
        """Start async voice input"""
        if self.is_running:
            logger.warning("AsyncVoiceInput already running")
            return
        
        self.loop = asyncio.get_running_loop()
        self.is_running = True
        
        # Start audio capture thread
        self.processing_thread = Thread(target=self._audio_capture_thread, daemon=True)
        self.processing_thread.start()
        
        # Start audio processing
        asyncio.create_task(self._process_audio_loop())
        
        logger.info("AsyncVoiceInput started")
    
    async def stop(self):
        """Stop async voice input"""
        if not self.is_running:
            return
        
        logger.info("Stopping AsyncVoiceInput...")
        self.is_running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # End current session if active
        if self.current_session.is_active:
            await self._end_session(time.time())
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("AsyncVoiceInput stopped")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio callback for sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Create audio chunk
        chunk = AudioChunk(
            data=indata.copy(),
            timestamp=time.time(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration=frames / self.sample_rate
        )
        
        # Add to processing queue
        self.audio_queue.put(chunk)
    
    def _audio_capture_thread(self):
        """Audio capture thread"""
        try:
            logger.info("Starting audio capture thread...")
            
            with sd.InputStream(
                callback=self._audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype=np.float32
            ) as stream:
                
                self.stream = stream
                logger.info("Audio stream started")
                
                # Keep thread alive while running
                while self.is_running:
                    time.sleep(0.01)
                
                logger.info("Audio capture thread stopped")
                
        except Exception as e:
            logger.error(f"Audio capture thread error: {e}")
    
    async def _process_audio_loop(self):
        """Process audio chunks asynchronously"""
        try:
            logger.info("Starting audio processing loop...")
            
            while self.is_running:
                try:
                    # Get audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    await self._process_audio_chunk(chunk)
                    
                except queue.Empty:
                    # Check for session timeout
                    await self._check_session_timeout()
                    continue
                    
        except Exception as e:
            logger.error(f"Audio processing loop error: {e}")
    
    async def _process_audio_chunk(self, chunk: AudioChunk):
        """Process individual audio chunk"""
        try:
            # Calculate audio level (RMS)
            audio_level = np.sqrt(np.mean(chunk.data ** 2))
            
            # Update session state
            if not self.current_session.is_active:
                if audio_level > self.silence_threshold:
                    await self._start_session(chunk.timestamp)
            
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
                        await self._end_session(chunk.timestamp)
                else:
                    self.silence_start = None
                
                # Check for max session duration
                if (chunk.timestamp - self.current_session.start_time > 
                    self.max_session_duration):
                    await self._end_session(chunk.timestamp)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _start_session(self, timestamp: float):
        """Start a new voice session"""
        logger.info("Starting voice session...")
        
        self.current_session = VoiceSession(
            is_active=True,
            start_time=timestamp,
            audio_chunks=[]
        )
        
        # Notify callback
        if self.callback:
            await self.callback("session_start", self.current_session)
    
    async def _end_session(self, timestamp: float):
        """End current voice session and process audio"""
        if not self.current_session.is_active:
            return
        
        logger.info("Ending voice session...")
        
        self.current_session.is_active = False
        self.current_session.end_time = timestamp
        
        # Process session audio
        await self._process_session_audio()
    
    async def _process_session_audio(self):
        """Process the complete session audio"""
        if not self.current_session.audio_chunks:
            logger.warning("No audio chunks to process")
            return
        
        try:
            logger.info("Processing session audio...")
            
            # Concatenate audio chunks
            audio_data = np.concatenate([chunk.data for chunk in self.current_session.audio_chunks])
            
            # Calculate session duration
            duration = self.current_session.end_time - self.current_session.start_time
            
            # Create session data
            session_data = {
                'audio_data': audio_data,
                'duration': duration,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'chunk_count': len(self.current_session.audio_chunks)
            }
            
            # Notify callback with session data
            if self.callback:
                await self.callback("session_complete", session_data)
            
            logger.info(f"Session processed: {duration:.2f}s, {len(self.current_session.audio_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing session audio: {e}")
    
    async def _check_session_timeout(self):
        """Check for session timeout due to silence"""
        if (self.current_session.is_active and 
            self.last_audio_time and 
            time.time() - self.last_audio_time > self.silence_duration):
            
            await self._end_session(time.time())
    
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
    
    def set_parameters(self, 
                      silence_threshold: Optional[float] = None,
                      silence_duration: Optional[float] = None,
                      max_session_duration: Optional[float] = None):
        """Update voice input parameters"""
        if silence_threshold is not None:
            self.silence_threshold = silence_threshold
        if silence_duration is not None:
            self.silence_duration = silence_duration
        if max_session_duration is not None:
            self.max_session_duration = max_session_duration
        
        logger.info(f"Parameters updated: threshold={self.silence_threshold}, "
                   f"silence_duration={self.silence_duration}, "
                   f"max_duration={self.max_session_duration}")

class VoiceActivityDetector:
    """Advanced voice activity detection"""
    
    def __init__(self, 
                 energy_threshold: float = 0.01,
                 silence_threshold: float = 0.005,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.5):
        
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        self.speech_start = None
        self.silence_start = None
        self.is_speaking = False
    
    def detect_activity(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Detect voice activity in audio chunk"""
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        current_time = time.time()
        
        # Detect speech start
        if energy > self.energy_threshold and not self.is_speaking:
            if self.speech_start is None:
                self.speech_start = current_time
            elif current_time - self.speech_start > self.min_speech_duration:
                self.is_speaking = True
                self.silence_start = None
        
        # Detect silence
        elif energy < self.silence_threshold and self.is_speaking:
            if self.silence_start is None:
                self.silence_start = current_time
            elif current_time - self.silence_start > self.min_silence_duration:
                self.is_speaking = False
                self.speech_start = None
        
        # Reset if energy is in middle range
        elif self.energy_threshold >= energy >= self.silence_threshold:
            if self.is_speaking:
                self.silence_start = None
            else:
                self.speech_start = None
        
        return {
            'is_speaking': self.is_speaking,
            'energy': energy,
            'speech_start': self.speech_start,
            'silence_start': self.silence_start
        }

# Example usage and testing
async def example_callback(event_type: str, data: Any):
    """Example callback for voice input"""
    if event_type == "session_start":
        print(f"🎤 Voice session started")
    elif event_type == "session_complete":
        print(f"✅ Voice session completed: {data['duration']:.2f}s, {data['chunk_count']} chunks")
        # Here you would typically send audio_data to Whisper for transcription

async def main():
    """Example usage of AsyncVoiceInput"""
    voice_input = AsyncVoiceInput(example_callback)
    
    print("Starting voice input...")
    await voice_input.start()
    
    try:
        # Keep running for 30 seconds
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        print("Stopping voice input...")
    finally:
        await voice_input.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())