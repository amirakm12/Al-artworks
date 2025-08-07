import asyncio
import sounddevice as sd
import numpy as np
import queue
import threading
import logging
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

log = logging.getLogger("VoiceAgent")

@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float

class AsyncVoiceAgent:
    def __init__(self, samplerate: int = 16000, blocksize: int = 1024, channels: int = 1,
                 silence_threshold: float = 0.01, silence_duration: float = 2.0):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        # Audio processing state
        self.audio_queue = queue.Queue()
        self.running = False
        self.loop = None
        self.stream = None
        
        # Voice session management
        self.is_listening = False
        self.session_start_time = None
        self.last_audio_time = None
        self.audio_buffer = []
        
        # Callbacks
        self.on_voice_start: Optional[Callable] = None
        self.on_voice_end: Optional[Callable] = None
        self.on_audio_chunk: Optional[Callable] = None

    def set_callbacks(self, on_voice_start: Optional[Callable] = None,
                     on_voice_end: Optional[Callable] = None,
                     on_audio_chunk: Optional[Callable] = None):
        """Set callback functions for voice events"""
        self.on_voice_start = on_voice_start
        self.on_voice_end = on_voice_end
        self.on_audio_chunk = on_audio_chunk

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio capture - runs in audio thread"""
        if status:
            log.warning(f"Audio callback status: {status}")
        
        # Create audio chunk with metadata
        chunk = AudioChunk(
            data=indata.copy(),
            timestamp=time.time(),
            sample_rate=self.samplerate,
            duration=frames / self.samplerate
        )
        
        # Put chunk in queue for processing
        self.audio_queue.put(chunk)

    async def process_audio(self):
        """Main audio processing loop - runs in asyncio event loop"""
        while self.running:
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                await self._process_audio_chunk(audio_chunk)
            except queue.Empty:
                # Check for session timeout when no audio
                await self._check_session_timeout()
            except Exception as e:
                log.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.01)

    async def _process_audio_chunk(self, chunk: AudioChunk):
        """Process individual audio chunk for voice activity detection"""
        # Update last audio time
        self.last_audio_time = chunk.timestamp
        
        # Check for voice activity (simple RMS-based detection)
        rms = np.sqrt(np.mean(chunk.data**2))
        is_speech = rms > self.silence_threshold
        
        # Call audio chunk callback if set
        if self.on_audio_chunk:
            try:
                await self.on_audio_chunk(chunk, is_speech)
            except Exception as e:
                log.error(f"Error in audio chunk callback: {e}")
        
        if is_speech and not self.is_listening:
            # Start voice session
            await self._start_voice_session(chunk.timestamp)
        
        if self.is_listening:
            # Add chunk to buffer
            self.audio_buffer.append(chunk)
            
            # Check if session should end (silence or max duration)
            if not is_speech:
                silence_elapsed = chunk.timestamp - self.last_audio_time
                if silence_elapsed >= self.silence_duration:
                    await self._end_voice_session(chunk.timestamp)
            else:
                # Check max session duration (30 seconds)
                session_duration = chunk.timestamp - self.session_start_time
                if session_duration >= 30.0:
                    await self._end_voice_session(chunk.timestamp)

    async def _start_voice_session(self, timestamp: float):
        """Start a new voice session"""
        self.is_listening = True
        self.session_start_time = timestamp
        self.audio_buffer = []
        
        log.info("Voice session started")
        
        # Call voice start callback if set
        if self.on_voice_start:
            try:
                await self.on_voice_start(timestamp)
            except Exception as e:
                log.error(f"Error in voice start callback: {e}")

    async def _end_voice_session(self, timestamp: float):
        """End current voice session and process audio"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        session_duration = timestamp - self.session_start_time
        
        log.info(f"Voice session ended after {session_duration:.2f} seconds")
        
        # Process collected audio
        if self.audio_buffer:
            # Concatenate all audio chunks
            audio_data = []
            for chunk in self.audio_buffer:
                audio_data.append(chunk.data.flatten())
            
            if audio_data:
                full_audio = np.concatenate(audio_data)
                
                # Call voice end callback if set
                if self.on_voice_end:
                    try:
                        await self.on_voice_end(full_audio, self.samplerate, session_duration)
                    except Exception as e:
                        log.error(f"Error in voice end callback: {e}")

    async def _check_session_timeout(self):
        """Check if current session should timeout due to silence"""
        if self.is_listening and self.last_audio_time:
            current_time = time.time()
            silence_elapsed = current_time - self.last_audio_time
            
            if silence_elapsed >= self.silence_duration:
                await self._end_voice_session(current_time)

    async def start(self):
        """Start the async voice agent"""
        if self.running:
            log.warning("VoiceAgent already running")
            return
        
        self.running = True
        self.loop = asyncio.get_running_loop()
        
        log.info("Starting AsyncVoiceAgent")
        
        # Start audio stream in separate thread
        def audio_thread():
            try:
                with sd.InputStream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    channels=self.channels,
                    callback=self.audio_callback,
                    dtype='float32'
                ) as stream:
                    self.stream = stream
                    log.info("Audio stream started")
                    while self.running:
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
            except Exception as e:
                log.error(f"Audio thread error: {e}")
        
        # Start audio thread
        audio_thread_obj = threading.Thread(target=audio_thread, daemon=True)
        audio_thread_obj.start()
        
        # Start audio processing loop
        await self.process_audio()

    async def stop(self):
        """Stop the async voice agent"""
        if not self.running:
            return
        
        self.running = False
        
        # End current session if active
        if self.is_listening:
            await self._end_voice_session(time.time())
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        log.info("AsyncVoiceAgent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current voice agent status"""
        return {
            "running": self.running,
            "is_listening": self.is_listening,
            "session_duration": time.time() - self.session_start_time if self.session_start_time else 0,
            "audio_buffer_size": len(self.audio_buffer),
            "queue_size": self.audio_queue.qsize()
        }

    def set_parameters(self, silence_threshold: Optional[float] = None,
                      silence_duration: Optional[float] = None):
        """Update voice detection parameters"""
        if silence_threshold is not None:
            self.silence_threshold = silence_threshold
        
        if silence_duration is not None:
            self.silence_duration = silence_duration

# Example usage and testing
async def example_voice_callback(audio_data: np.ndarray, sample_rate: int, duration: float):
    """Example callback for voice session end"""
    log.info(f"Voice session captured: {len(audio_data)} samples, {duration:.2f}s duration")
    # Here you would send audio_data to Whisper for transcription
    # transcription = await whisper_model.transcribe(audio_data)

async def example_audio_chunk_callback(chunk: AudioChunk, is_speech: bool):
    """Example callback for individual audio chunks"""
    if is_speech:
        log.debug(f"Speech detected: RMS={np.sqrt(np.mean(chunk.data**2)):.4f}")

async def main():
    """Example main function showing how to use AsyncVoiceAgent"""
    logging.basicConfig(level=logging.INFO)
    
    # Create voice agent
    agent = AsyncVoiceAgent(
        samplerate=16000,
        blocksize=1024,
        silence_threshold=0.01,
        silence_duration=2.0
    )
    
    # Set callbacks
    agent.set_callbacks(
        on_voice_end=example_voice_callback,
        on_audio_chunk=example_audio_chunk_callback
    )
    
    try:
        log.info("Starting voice agent... Press Ctrl+C to stop")
        await agent.start()
    except KeyboardInterrupt:
        log.info("Stopping voice agent...")
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())