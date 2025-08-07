import asyncio
import logging
import whisper
import sounddevice as sd
import numpy as np
from typing import Optional, Callable, Dict, Any
from voice.voice_agent_whisper import AsyncVoiceAgent
from voice.tts_agent import TTSEngine

log = logging.getLogger("VoiceAgent")

class VoiceAgent:
    """Main voice agent that integrates ASR, TTS, and command processing"""
    
    def __init__(self, plugin_manager, ai_manager=None):
        self.plugin_manager = plugin_manager
        self.ai_manager = ai_manager
        self.running = False
        
        # Voice components
        self.voice_agent = AsyncVoiceAgent()
        self.tts_engine = TTSEngine()
        
        # Voice state
        self.is_listening = False
        self.last_transcription = ""
        self.voice_session_active = False
        
        # Callbacks
        self.on_voice_start: Optional[Callable] = None
        self.on_voice_end: Optional[Callable] = None
        self.on_transcription: Optional[Callable] = None
        self.on_ai_response: Optional[Callable] = None
        
        log.info("VoiceAgent initialized")

    def set_callbacks(self, on_voice_start: Optional[Callable] = None,
                     on_voice_end: Optional[Callable] = None,
                     on_transcription: Optional[Callable] = None,
                     on_ai_response: Optional[Callable] = None):
        """Set callback functions for voice events"""
        self.on_voice_start = on_voice_start
        self.on_voice_end = on_voice_end
        self.on_transcription = on_transcription
        self.on_ai_response = on_ai_response

    async def start(self):
        """Start the voice agent"""
        if self.running:
            log.warning("VoiceAgent already running")
            return
        
        self.running = True
        log.info("Starting VoiceAgent...")
        
        # Start voice agent
        await self.voice_agent.start()
        
        # Set up voice callbacks
        self.voice_agent.set_callbacks(
            on_voice_start=self._on_voice_start,
            on_voice_end=self._on_voice_end,
            on_audio_chunk=self._on_audio_chunk
        )
        
        # Start TTS engine
        await self.tts_engine.start()
        
        log.info("VoiceAgent started successfully")

    async def stop(self):
        """Stop the voice agent"""
        if not self.running:
            return
        
        self.running = False
        log.info("Stopping VoiceAgent...")
        
        # Stop voice agent
        await self.voice_agent.stop()
        
        # Stop TTS engine
        await self.tts_engine.stop()
        
        log.info("VoiceAgent stopped")

    async def _on_voice_start(self, timestamp: float):
        """Called when voice activity starts"""
        self.voice_session_active = True
        self.is_listening = True
        log.info("Voice session started")
        
        if self.on_voice_start:
            try:
                await self.on_voice_start(timestamp)
            except Exception as e:
                log.error(f"Error in voice start callback: {e}")

    async def _on_voice_end(self, timestamp: float):
        """Called when voice activity ends"""
        self.voice_session_active = False
        self.is_listening = False
        log.info("Voice session ended")
        
        if self.on_voice_end:
            try:
                await self.on_voice_end(timestamp)
            except Exception as e:
                log.error(f"Error in voice end callback: {e}")

    async def _on_audio_chunk(self, chunk, is_speech: bool):
        """Called for each audio chunk"""
        if self.on_audio_chunk:
            try:
                await self.on_audio_chunk(chunk, is_speech)
            except Exception as e:
                log.error(f"Error in audio chunk callback: {e}")

    async def process_voice_command(self, text: str) -> str:
        """Process a voice command and return AI response"""
        log.info(f"Processing voice command: {text}")
        
        # Store last transcription
        self.last_transcription = text
        
        # Call transcription callback
        if self.on_transcription:
            try:
                await self.on_transcription(text)
            except Exception as e:
                log.error(f"Error in transcription callback: {e}")
        
        # Try to handle with plugins first
        handled = False
        if self.plugin_manager:
            try:
                handled = await self.plugin_manager.dispatch_voice_command(text)
                log.info(f"Voice command handled by plugins: {handled}")
            except Exception as e:
                log.error(f"Error dispatching to plugins: {e}")
        
        # If not handled by plugins, use AI manager
        if not handled and self.ai_manager:
            try:
                response = await self.ai_manager.generate_response(text)
                log.info(f"AI response generated: {response[:50]}...")
                
                # Call AI response callback
                if self.on_ai_response:
                    await self.on_ai_response(response)
                
                # Speak the response
                await self.speak_response(response)
                
                return response
                
            except Exception as e:
                log.error(f"Error generating AI response: {e}")
                return f"Error processing command: {str(e)}"
        
        return "Command processed"

    async def speak_response(self, text: str):
        """Speak a text response using TTS"""
        if self.tts_engine and self.tts_engine.running:
            try:
                self.tts_engine.synthesize(text)
                log.info(f"TTS synthesis started for: {text[:50]}...")
            except Exception as e:
                log.error(f"Error in TTS synthesis: {e}")
        else:
            log.warning("TTS engine not available for speech synthesis")

    async def transcribe_audio_file(self, file_path: str) -> str:
        """Transcribe an audio file using Whisper"""
        try:
            log.info(f"Transcribing audio file: {file_path}")
            result = await self.voice_agent.transcribe_file(file_path)
            return result.get("text", "").strip()
        except Exception as e:
            log.error(f"Error transcribing audio file: {e}")
            return ""

    def get_status(self) -> Dict[str, Any]:
        """Get voice agent status"""
        return {
            "running": self.running,
            "is_listening": self.is_listening,
            "voice_session_active": self.voice_session_active,
            "last_transcription": self.last_transcription,
            "voice_agent_status": self.voice_agent.get_status() if self.voice_agent else {},
            "tts_status": self.tts_engine.get_status() if self.tts_engine else {}
        }

    def set_voice_parameters(self, silence_threshold: Optional[float] = None,
                           silence_duration: Optional[float] = None):
        """Set voice detection parameters"""
        if self.voice_agent:
            self.voice_agent.set_parameters(silence_threshold, silence_duration)

    async def toggle_listening(self):
        """Toggle voice listening state"""
        if self.is_listening:
            self.is_listening = False
            log.info("Voice listening disabled")
        else:
            self.is_listening = True
            log.info("Voice listening enabled")

# Example usage
async def example_voice_callback(timestamp: float):
    """Example voice start callback"""
    log.info(f"Voice started at {timestamp}")

async def example_transcription_callback(text: str):
    """Example transcription callback"""
    log.info(f"Transcribed: {text}")

async def example_ai_response_callback(response: str):
    """Example AI response callback"""
    log.info(f"AI Response: {response}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock plugin manager and AI manager
    class MockPluginManager:
        async def dispatch_voice_command(self, text):
            return False
    
    class MockAIManager:
        async def generate_response(self, text):
            return f"AI response to: {text}"
    
    async def main():
        plugin_manager = MockPluginManager()
        ai_manager = MockAIManager()
        
        voice_agent = VoiceAgent(plugin_manager, ai_manager)
        
        # Set callbacks
        voice_agent.set_callbacks(
            on_voice_start=example_voice_callback,
            on_transcription=example_transcription_callback,
            on_ai_response=example_ai_response_callback
        )
        
        try:
            await voice_agent.start()
            
            # Keep running
            await asyncio.sleep(30)
            
        finally:
            await voice_agent.stop()
    
    asyncio.run(main())