import asyncio
import queue
import threading
import logging
import numpy as np
import sounddevice as sd
from TTS.api import TTS  # Assuming Coqui TTS
from typing import Optional, Callable, Dict, Any

log = logging.getLogger("TTSAgent")

class AsyncTTSAgent:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", 
                 sample_rate: int = 22050, device: Optional[str] = None):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize TTS model
        try:
            self.tts = TTS(model_name)
            log.info(f"TTS model loaded: {model_name}")
        except Exception as e:
            log.error(f"Failed to load TTS model {model_name}: {e}")
            self.tts = None
        
        # Audio playback state
        self.audio_queue = queue.Queue()
        self.running = False
        self.play_thread = None
        
        # Callbacks
        self.on_synthesis_start: Optional[Callable] = None
        self.on_synthesis_complete: Optional[Callable] = None
        self.on_playback_start: Optional[Callable] = None
        self.on_playback_complete: Optional[Callable] = None

    def set_callbacks(self, on_synthesis_start: Optional[Callable] = None,
                     on_synthesis_complete: Optional[Callable] = None,
                     on_playback_start: Optional[Callable] = None,
                     on_playback_complete: Optional[Callable] = None):
        """Set callback functions for TTS events"""
        self.on_synthesis_start = on_synthesis_start
        self.on_synthesis_complete = on_synthesis_complete
        self.on_playback_start = on_playback_start
        self.on_playback_complete = on_playback_complete

    async def synthesize_text(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to speech asynchronously
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array, or None if synthesis failed
        """
        if not self.tts:
            log.error("TTS model not loaded")
            return None
        
        if self.on_synthesis_start:
            try:
                await self.on_synthesis_start(text)
            except Exception as e:
                log.error(f"Error in synthesis start callback: {e}")
        
        log.info(f"Synthesizing text: {text[:50]}...")
        
        try:
            # Run TTS synthesis in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(None, self.tts.tts, text)
            
            if self.on_synthesis_complete:
                try:
                    await self.on_synthesis_complete(text, wav)
                except Exception as e:
                    log.error(f"Error in synthesis complete callback: {e}")
            
            log.info(f"Synthesis complete: {len(wav)} samples")
            return wav
            
        except Exception as e:
            log.error(f"TTS synthesis failed: {e}")
            return None

    async def synthesize_and_play(self, text: str):
        """
        Synthesize text and queue it for playback
        
        Args:
            text: Text to synthesize and play
        """
        wav = await self.synthesize_text(text)
        if wav is not None:
            self.audio_queue.put(wav)
            log.info(f"Audio queued for playback: {len(wav)} samples")

    def playback_worker(self):
        """Audio playback worker thread"""
        while self.running:
            try:
                # Get audio from queue with timeout
                wav = self.audio_queue.get(timeout=0.5)
                
                if self.on_playback_start:
                    try:
                        # Run callback in event loop
                        asyncio.run_coroutine_threadsafe(
                            self.on_playback_start(len(wav)), 
                            asyncio.get_event_loop()
                        )
                    except Exception as e:
                        log.error(f"Error in playback start callback: {e}")
                
                log.info(f"Playing audio chunk of length {len(wav)}")
                
                # Play audio using sounddevice
                sd.play(wav, samplerate=self.sample_rate)
                sd.wait()  # Wait for playback to complete
                
                if self.on_playback_complete:
                    try:
                        # Run callback in event loop
                        asyncio.run_coroutine_threadsafe(
                            self.on_playback_complete(len(wav)), 
                            asyncio.get_event_loop()
                        )
                    except Exception as e:
                        log.error(f"Error in playback complete callback: {e}")
                
                log.info("Audio playback completed")
                
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Playback error: {e}")

    async def start(self):
        """Start the async TTS agent"""
        if self.running:
            log.warning("TTSAgent already running")
            return
        
        self.running = True
        
        # Start playback thread
        self.play_thread = threading.Thread(target=self.playback_worker, daemon=True)
        self.play_thread.start()
        
        log.info("TTS Playback thread started")

    async def stop(self):
        """Stop the async TTS agent"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for playback thread to finish
        if self.play_thread:
            self.play_thread.join(timeout=2.0)
            if self.play_thread.is_alive():
                log.warning("Playback thread did not stop gracefully")
        
        log.info("TTS Playback thread stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current TTS agent status"""
        return {
            "running": self.running,
            "model_loaded": self.tts is not None,
            "model_name": self.model_name,
            "queue_size": self.audio_queue.qsize(),
            "sample_rate": self.sample_rate
        }

    async def clear_queue(self):
        """Clear the audio playback queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        log.info("Audio queue cleared")

    async def test_synthesis(self, text: str = "Hello, this is a test of the TTS system."):
        """Test TTS synthesis and playback"""
        log.info("Testing TTS synthesis and playback")
        await self.synthesize_and_play(text)

# Example callbacks
async def example_synthesis_start(text: str):
    """Example callback for synthesis start"""
    log.info(f"Starting synthesis: {text[:30]}...")

async def example_synthesis_complete(text: str, wav: np.ndarray):
    """Example callback for synthesis complete"""
    log.info(f"Synthesis complete: {len(wav)} samples for '{text[:30]}...'")

async def example_playback_start(sample_count: int):
    """Example callback for playback start"""
    log.info(f"Starting playback: {sample_count} samples")

async def example_playback_complete(sample_count: int):
    """Example callback for playback complete"""
    log.info(f"Playback complete: {sample_count} samples")

# Example usage and testing
async def main():
    """Example main function showing how to use AsyncTTSAgent"""
    logging.basicConfig(level=logging.INFO)
    
    # Create TTS agent
    agent = AsyncTTSAgent(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        sample_rate=22050
    )
    
    # Set callbacks
    agent.set_callbacks(
        on_synthesis_start=example_synthesis_start,
        on_synthesis_complete=example_synthesis_complete,
        on_playback_start=example_playback_start,
        on_playback_complete=example_playback_complete
    )
    
    try:
        # Start the agent
        await agent.start()
        
        # Test synthesis and playback
        await agent.test_synthesis("Hello, this is a test of the async TTS playback pipeline.")
        
        # Keep running for a bit to allow playback
        await asyncio.sleep(5)
        
        # Test another synthesis
        await agent.synthesize_and_play("This is another test of the TTS system.")
        
        # Wait for playback to complete
        await asyncio.sleep(3)
        
    except KeyboardInterrupt:
        log.info("Stopping TTS agent...")
    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())