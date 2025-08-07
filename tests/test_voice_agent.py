import unittest
import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

# Mock Voice Agent for testing
class MockVoiceAgent:
    """Mock voice agent for testing purposes"""
    
    def __init__(self, plugin_manager=None, ai_manager=None):
        self.plugin_manager = plugin_manager
        self.ai_manager = ai_manager
        self.running = False
        self.listening = False
        self.audio_chunks_processed = 0
        self.voice_commands_processed = 0
        self.transcriptions = []
        self.audio_buffer = []
        self.sample_rate = 16000
        self.channels = 1
        self.blocksize = 1024
        
        # Mock audio data
        self.mock_audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    
    async def start(self):
        """Mock start method"""
        self.running = True
        self.listening = True
        return True
    
    async def stop(self):
        """Mock stop method"""
        self.running = False
        self.listening = False
        return True
    
    async def listen(self) -> bytes:
        """Mock listen method - returns audio data"""
        if not self.listening:
            raise RuntimeError("Voice agent not listening")
        
        # Simulate audio capture
        await asyncio.sleep(0.1)  # Simulate processing time
        return self.mock_audio_data.tobytes()
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Mock transcription method"""
        # Simulate transcription
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Mock transcription based on audio data length
        if len(audio_data) > 8000:  # More than 0.5 seconds
            transcription = "Hello world"
        else:
            transcription = ""
        
        self.transcriptions.append(transcription)
        return transcription
    
    async def process_voice_command(self, text: str) -> str:
        """Mock voice command processing"""
        self.voice_commands_processed += 1
        
        # Mock command processing
        if self.plugin_manager:
            handled = await self.plugin_manager.dispatch_voice_command(text)
            if handled:
                return f"Plugin handled: {text}"
        
        if self.ai_manager:
            response = await self.ai_manager.generate_response(text)
            return f"AI response: {response}"
        
        return f"Processed command: {text}"
    
    async def capture_audio_loop(self):
        """Mock audio capture loop"""
        while self.running:
            try:
                # Simulate audio capture
                audio_chunk = await self.listen()
                self.audio_chunks_processed += 1
                
                # Process audio chunk
                if len(audio_chunk) > 0:
                    transcription = await self.transcribe(audio_chunk)
                    if transcription:
                        response = await self.process_voice_command(transcription)
                        # In a real implementation, this would trigger TTS
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                # Log error but continue
                print(f"Error in audio loop: {e}")
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice agent statistics"""
        return {
            "running": self.running,
            "listening": self.listening,
            "audio_chunks_processed": self.audio_chunks_processed,
            "voice_commands_processed": self.voice_commands_processed,
            "transcriptions_count": len(self.transcriptions),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "blocksize": self.blocksize
        }

# Mock TTS Engine
class MockTTSEngine:
    """Mock TTS engine for testing"""
    
    def __init__(self):
        self.speaking = False
        self.speech_queue = asyncio.Queue()
        self.synthesized_texts = []
        self.audio_generated = 0
    
    async def start(self):
        """Mock TTS start"""
        self.speaking = True
        return True
    
    async def stop(self):
        """Mock TTS stop"""
        self.speaking = False
        return True
    
    async def synthesize(self, text: str) -> np.ndarray:
        """Mock speech synthesis"""
        if not self.speaking:
            raise RuntimeError("TTS engine not started")
        
        # Simulate synthesis time
        await asyncio.sleep(0.1)
        
        # Generate mock audio
        audio_length = len(text) * 100  # Rough estimate
        audio = np.random.randn(audio_length).astype(np.float32)
        
        self.synthesized_texts.append(text)
        self.audio_generated += 1
        
        return audio
    
    async def speak(self, text: str):
        """Mock speak method"""
        audio = await self.synthesize(text)
        # In a real implementation, this would play the audio
        await asyncio.sleep(0.05)  # Simulate playback time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        return {
            "speaking": self.speaking,
            "synthesized_texts_count": len(self.synthesized_texts),
            "audio_generated": self.audio_generated,
            "queue_size": self.speech_queue.qsize()
        }

class TestVoiceAgent(unittest.TestCase):
    """Test suite for Voice Agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_manager = Mock()
        self.ai_manager = Mock()
        self.voice_agent = MockVoiceAgent(self.plugin_manager, self.ai_manager)
    
    def test_voice_agent_initialization(self):
        """Test voice agent initialization"""
        self.assertIsNotNone(self.voice_agent)
        self.assertFalse(self.voice_agent.running)
        self.assertFalse(self.voice_agent.listening)
        self.assertEqual(self.voice_agent.audio_chunks_processed, 0)
    
    async def test_start_stop(self):
        """Test start and stop functionality"""
        # Test start
        result = await self.voice_agent.start()
        self.assertTrue(result)
        self.assertTrue(self.voice_agent.running)
        self.assertTrue(self.voice_agent.listening)
        
        # Test stop
        result = await self.voice_agent.stop()
        self.assertTrue(result)
        self.assertFalse(self.voice_agent.running)
        self.assertFalse(self.voice_agent.listening)
    
    async def test_listen(self):
        """Test audio listening functionality"""
        # Start the agent
        await self.voice_agent.start()
        
        # Test listening
        audio_data = await self.voice_agent.listen()
        self.assertIsInstance(audio_data, bytes)
        self.assertGreater(len(audio_data), 0)
        
        # Test listening when not started
        await self.voice_agent.stop()
        with self.assertRaises(RuntimeError):
            await self.voice_agent.listen()
    
    async def test_transcribe(self):
        """Test transcription functionality"""
        # Test transcription with audio data
        audio_data = self.voice_agent.mock_audio_data.tobytes()
        transcription = await self.voice_agent.transcribe(audio_data)
        
        self.assertIsInstance(transcription, str)
        self.assertIn(transcription, self.voice_agent.transcriptions)
        
        # Test transcription with empty audio
        empty_transcription = await self.voice_agent.transcribe(b"")
        self.assertEqual(empty_transcription, "")
    
    async def test_process_voice_command(self):
        """Test voice command processing"""
        # Test with plugin manager
        self.plugin_manager.dispatch_voice_command = AsyncMock(return_value=True)
        response = await self.voice_agent.process_voice_command("hello")
        self.assertIn("Plugin handled", response)
        
        # Test with AI manager (when plugin doesn't handle)
        self.plugin_manager.dispatch_voice_command = AsyncMock(return_value=False)
        self.ai_manager.generate_response = AsyncMock(return_value="AI response")
        response = await self.voice_agent.process_voice_command("hello")
        self.assertIn("AI response", response)
        
        # Test without managers
        voice_agent_no_managers = MockVoiceAgent()
        response = await voice_agent_no_managers.process_voice_command("hello")
        self.assertIn("Processed command", response)
    
    async def test_capture_audio_loop(self):
        """Test audio capture loop"""
        # Start the agent
        await self.voice_agent.start()
        
        # Run capture loop for a short time
        loop_task = asyncio.create_task(self.voice_agent.capture_audio_loop())
        await asyncio.sleep(0.2)  # Run for 200ms
        
        # Stop the agent
        await self.voice_agent.stop()
        loop_task.cancel()
        
        # Check that some processing occurred
        self.assertGreater(self.voice_agent.audio_chunks_processed, 0)
    
    def test_get_stats(self):
        """Test statistics gathering"""
        stats = self.voice_agent.get_stats()
        
        self.assertIn("running", stats)
        self.assertIn("listening", stats)
        self.assertIn("audio_chunks_processed", stats)
        self.assertIn("voice_commands_processed", stats)
        self.assertIn("transcriptions_count", stats)
        self.assertIn("sample_rate", stats)
        self.assertIn("channels", stats)
        self.assertIn("blocksize", stats)

@pytest.mark.asyncio
class TestVoiceAgentAsync:
    """Async test suite for Voice Agent"""
    
    @pytest.fixture
    def voice_agent(self):
        """Async fixture for voice agent"""
        return MockVoiceAgent()
    
    async def test_async_voice_operations(self, voice_agent):
        """Test async voice operations"""
        # Start agent
        await voice_agent.start()
        
        # Test concurrent operations
        tasks = [
            voice_agent.listen(),
            voice_agent.transcribe(b"test audio"),
            voice_agent.process_voice_command("hello")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Check results
        self.assertIsInstance(results[0], bytes)  # Audio data
        self.assertIsInstance(results[1], str)    # Transcription
        self.assertIsInstance(results[2], str)    # Command response
    
    async def test_voice_agent_error_handling(self, voice_agent):
        """Test error handling in voice agent"""
        # Test listening without starting
        with self.assertRaises(RuntimeError):
            await voice_agent.listen()
        
        # Test processing with invalid input
        response = await voice_agent.process_voice_command("")
        self.assertIsInstance(response, str)
    
    async def test_voice_agent_integration(self, voice_agent):
        """Test voice agent integration with other components"""
        # Mock plugin manager
        plugin_manager = Mock()
        plugin_manager.dispatch_voice_command = AsyncMock(return_value=True)
        
        # Mock AI manager
        ai_manager = Mock()
        ai_manager.generate_response = AsyncMock(return_value="Test response")
        
        # Create voice agent with managers
        integrated_voice_agent = MockVoiceAgent(plugin_manager, ai_manager)
        await integrated_voice_agent.start()
        
        # Test integrated processing
        response = await integrated_voice_agent.process_voice_command("test command")
        self.assertIn("Plugin handled", response)
        
        # Test AI fallback
        plugin_manager.dispatch_voice_command = AsyncMock(return_value=False)
        response = await integrated_voice_agent.process_voice_command("test command")
        self.assertIn("AI response", response)

class TestTTSEngine:
    """Test suite for TTS Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tts_engine = MockTTSEngine()
    
    async def test_tts_start_stop(self):
        """Test TTS start and stop"""
        # Test start
        result = await self.tts_engine.start()
        self.assertTrue(result)
        self.assertTrue(self.tts_engine.speaking)
        
        # Test stop
        result = await self.tts_engine.stop()
        self.assertTrue(result)
        self.assertFalse(self.tts_engine.speaking)
    
    async def test_tts_synthesize(self):
        """Test TTS synthesis"""
        await self.tts_engine.start()
        
        # Test synthesis
        audio = await self.tts_engine.synthesize("Hello world")
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)
        self.assertEqual(audio.dtype, np.float32)
        
        # Check statistics
        self.assertEqual(self.tts_engine.audio_generated, 1)
        self.assertIn("Hello world", self.tts_engine.synthesized_texts)
    
    async def test_tts_speak(self):
        """Test TTS speak functionality"""
        await self.tts_engine.start()
        
        # Test speak
        await self.tts_engine.speak("Test speech")
        
        # Check that synthesis occurred
        self.assertIn("Test speech", self.tts_engine.synthesized_texts)
    
    async def test_tts_error_handling(self):
        """Test TTS error handling"""
        # Test synthesis without starting
        with self.assertRaises(RuntimeError):
            await self.tts_engine.synthesize("test")
    
    def test_tts_get_stats(self):
        """Test TTS statistics"""
        stats = self.tts_engine.get_stats()
        
        self.assertIn("speaking", stats)
        self.assertIn("synthesized_texts_count", stats)
        self.assertIn("audio_generated", stats)
        self.assertIn("queue_size", stats)

class TestVoiceAgentIntegration:
    """Integration tests for Voice Agent with real components"""
    
    def test_voice_agent_with_real_audio(self):
        """Test voice agent with real audio processing"""
        # This would test with real audio processing libraries
        # Implementation depends on your actual audio processing
        pass
    
    def test_voice_agent_with_real_transcription(self):
        """Test voice agent with real transcription"""
        # This would test with real Whisper or other ASR
        # Implementation depends on your actual transcription
        pass

# Performance tests
class TestVoiceAgentPerformance:
    """Performance tests for Voice Agent"""
    
    async def test_voice_processing_performance(self):
        """Test voice processing performance"""
        voice_agent = MockVoiceAgent()
        await voice_agent.start()
        
        import time
        start_time = time.time()
        
        # Process many voice commands
        for i in range(100):
            await voice_agent.process_voice_command(f"command {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 5.0)  # Less than 5 seconds for 100 commands
        self.assertEqual(voice_agent.voice_commands_processed, 100)
    
    async def test_concurrent_voice_operations(self):
        """Test concurrent voice operations"""
        voice_agent = MockVoiceAgent()
        await voice_agent.start()
        
        import time
        start_time = time.time()
        
        # Create many concurrent operations
        tasks = []
        for i in range(50):
            tasks.append(voice_agent.listen())
            tasks.append(voice_agent.transcribe(b"test audio"))
            tasks.append(voice_agent.process_voice_command(f"command {i}"))
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 10.0)  # Less than 10 seconds for 150 operations

# Memory tests
class TestVoiceAgentMemory:
    """Memory usage tests for Voice Agent"""
    
    async def test_memory_usage_with_voice_processing(self):
        """Test memory usage during voice processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        voice_agent = MockVoiceAgent()
        await voice_agent.start()
        
        # Process many voice commands
        for i in range(1000):
            await voice_agent.process_voice_command(f"command {i}")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)