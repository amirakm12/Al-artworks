import asyncio
import sounddevice as sd
import queue
import numpy as np
import logging
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

log = logging.getLogger("ContinuousVoice")

@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float

class AsyncVoiceInput:
    """Async always-on voice listening with real-time audio processing"""
    
    def __init__(self, samplerate: int = 16000, channels: int = 1, 
                 blocksize: int = 1024, device: Optional[int] = None):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self.q = queue.Queue()
        self.loop = asyncio.get_event_loop()
        self.running = False
        self.stream = None
        self.audio_processor = None
        self.stats = {
            "chunks_processed": 0,
            "total_audio_time": 0.0,
            "start_time": None,
            "last_chunk_time": None
        }

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict[str, Any], status: sd.CallbackFlags):
        """Callback for audio input - called by sounddevice"""
        if status:
            log.warning(f"Audio callback status: {status}")
        
        # Create audio chunk with metadata
        chunk = AudioChunk(
            data=indata.copy(),
            timestamp=time.time(),
            sample_rate=self.samplerate,
            channels=self.channels,
            duration=frames / self.samplerate
        )
        
        # Put in queue for async processing
        try:
            self.q.put_nowait(chunk)
        except queue.Full:
            log.warning("Audio queue full, dropping chunk")

    async def generator(self):
        """Async generator that yields audio chunks"""
        while self.running:
            try:
                # Get chunk from queue with timeout
                chunk = await self.loop.run_in_executor(None, self.q.get, True, 1.0)
                yield chunk
            except queue.Empty:
                # No audio data available, continue
                continue
            except Exception as e:
                log.error(f"Error in audio generator: {e}")
                await asyncio.sleep(0.1)

    async def start_listening(self, audio_processor_callback: Callable[[AudioChunk], None]):
        """Start continuous voice listening with audio processing"""
        self.running = True
        self.audio_processor = audio_processor_callback
        self.stats["start_time"] = time.time()
        
        log.info(f"Starting continuous voice input (device: {self.device}, "
                f"sample_rate: {self.samplerate}, channels: {self.channels})")
        
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                device=self.device,
                callback=self.audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            log.info("Audio stream started successfully")
            
            # Process audio chunks
            async for chunk in self.generator():
                if not self.running:
                    break
                
                try:
                    await self.audio_processor(chunk)
                    self.stats["chunks_processed"] += 1
                    self.stats["total_audio_time"] += chunk.duration
                    self.stats["last_chunk_time"] = chunk.timestamp
                except Exception as e:
                    log.error(f"Error processing audio chunk: {e}")
                
        except Exception as e:
            log.error(f"Error in voice listening: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop voice listening"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            log.info("Audio stream stopped")
        
        log.info(f"Voice listening stopped. Processed {self.stats['chunks_processed']} chunks "
                f"({self.stats['total_audio_time']:.2f}s total audio)")

    def get_stats(self) -> Dict[str, Any]:
        """Get voice input statistics"""
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        return {
            **self.stats,
            "uptime": uptime,
            "chunks_per_second": self.stats["chunks_processed"] / max(uptime, 1),
            "audio_time_per_second": self.stats["total_audio_time"] / max(uptime, 1),
            "running": self.running
        }

    def list_devices(self) -> list:
        """List available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_inputs'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_inputs'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices

    def set_device(self, device_index: int):
        """Set audio input device"""
        devices = self.list_devices()
        device_indices = [d['index'] for d in devices]
        
        if device_index in device_indices:
            self.device = device_index
            log.info(f"Audio device set to index {device_index}")
        else:
            log.error(f"Invalid device index {device_index}. Available devices: {device_indices}")

class VoiceActivityDetector:
    """Simple voice activity detection for continuous listening"""
    
    def __init__(self, threshold: float = 0.01, min_duration: float = 0.5):
        self.threshold = threshold
        self.min_duration = min_duration
        self.speech_start = None
        self.speech_buffer = []
        self.is_speaking = False

    def detect_activity(self, chunk: AudioChunk) -> bool:
        """Detect if audio chunk contains speech"""
        # Calculate RMS (Root Mean Square) of audio
        rms = np.sqrt(np.mean(chunk.data**2))
        
        if rms > self.threshold:
            if not self.is_speaking:
                self.speech_start = chunk.timestamp
                self.speech_buffer = []
                self.is_speaking = True
            
            self.speech_buffer.append(chunk)
            return True
        else:
            if self.is_speaking:
                # Check if speech duration meets minimum
                duration = chunk.timestamp - self.speech_start
                if duration >= self.min_duration:
                    log.info(f"Speech detected: {duration:.2f}s")
                    return True
                else:
                    # Reset if speech was too short
                    self.speech_buffer = []
                    self.is_speaking = False
            
            return False

    def get_speech_audio(self) -> Optional[np.ndarray]:
        """Get concatenated speech audio if available"""
        if self.speech_buffer:
            # Concatenate all speech chunks
            speech_data = np.concatenate([chunk.data for chunk in self.speech_buffer])
            self.speech_buffer = []  # Clear buffer
            return speech_data
        return None

# Example usage
async def example_audio_processor(chunk: AudioChunk):
    """Example audio processor that logs audio levels"""
    rms = np.sqrt(np.mean(chunk.data**2))
    if rms > 0.01:  # Only log when there's significant audio
        log.info(f"Audio chunk: RMS={rms:.4f}, Duration={chunk.duration:.3f}s")

async def example_voice_listening():
    """Example of using the continuous voice input"""
    logging.basicConfig(level=logging.INFO)
    
    # Create voice input
    voice_input = AsyncVoiceInput(samplerate=16000, channels=1, blocksize=1024)
    
    # List available devices
    devices = voice_input.list_devices()
    log.info("Available audio devices:")
    for device in devices:
        log.info(f"  {device['index']}: {device['name']} ({device['channels']} channels)")
    
    # Start listening
    try:
        await voice_input.start_listening(example_audio_processor)
    except KeyboardInterrupt:
        log.info("Stopping voice listening...")
    finally:
        voice_input.stop()
        
        # Print stats
        stats = voice_input.get_stats()
        log.info(f"Final stats: {stats}")

if __name__ == "__main__":
    asyncio.run(example_voice_listening())