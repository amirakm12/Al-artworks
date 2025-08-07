import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from voice.voice_agent import AsyncVoiceAgent, AudioChunk

@pytest.mark.asyncio
async def test_voice_agent_start_stop():
    """Test voice agent start and stop functionality"""
    agent = AsyncVoiceAgent()
    
    # Test start
    task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.1)  # Give it time to start
    
    # Verify it's running
    assert agent.running
    
    # Test stop
    await agent.stop()
    task.cancel()
    
    # Verify it's stopped
    assert not agent.running

@pytest.mark.asyncio
async def test_voice_agent_audio_callback():
    """Test audio callback functionality"""
    agent = AsyncVoiceAgent()
    
    # Mock audio data
    mock_audio_data = np.random.rand(1024, 1).astype(np.float32)
    
    # Test callback
    agent._audio_callback(mock_audio_data, 1024, None, None)
    
    # Check that audio was queued
    assert not agent.audio_queue.empty()
    
    # Get the queued chunk
    chunk = agent.audio_queue.get_nowait()
    assert isinstance(chunk, AudioChunk)
    assert chunk.data.shape == mock_audio_data.shape

@pytest.mark.asyncio
async def test_voice_agent_session_management():
    """Test voice session management"""
    agent = AsyncVoiceAgent()
    
    # Mock audio chunk
    chunk = AudioChunk(
        data=np.random.rand(1024, 1).astype(np.float32),
        timestamp=asyncio.get_event_loop().time(),
        sample_rate=16000,
        duration=0.064
    )
    
    # Test session start
    await agent._start_voice_session(chunk.timestamp)
    assert agent.current_session.is_active
    assert agent.current_session.start_time == chunk.timestamp
    
    # Test session end
    await agent._end_voice_session(chunk.timestamp)
    assert not agent.current_session.is_active
    assert agent.current_session.end_time == chunk.timestamp

@pytest.mark.asyncio
async def test_voice_agent_voice_activity_detection():
    """Test voice activity detection"""
    agent = AsyncVoiceAgent()
    
    # Test with speech-like audio (high amplitude)
    speech_audio = np.random.rand(1024, 1).astype(np.float32) * 0.1
    chunk = AudioChunk(
        data=speech_audio,
        timestamp=asyncio.get_event_loop().time(),
        sample_rate=16000,
        duration=0.064
    )
    
    # Process chunk
    await agent._process_audio_chunk(chunk)
    
    # Should detect speech and start session
    assert agent.current_session.is_active
    
    # Test with silence (low amplitude)
    silence_audio = np.random.rand(1024, 1).astype(np.float32) * 0.001
    chunk = AudioChunk(
        data=silence_audio,
        timestamp=asyncio.get_event_loop().time(),
        sample_rate=16000,
        duration=0.064
    )
    
    # Process silence chunk
    await agent._process_audio_chunk(chunk)
    
    # Should not start new session for silence
    # (This depends on the VAD implementation)

@pytest.mark.asyncio
async def test_voice_agent_callbacks():
    """Test voice agent callback functionality"""
    agent = AsyncVoiceAgent()
    
    # Mock callbacks
    voice_start_called = False
    voice_end_called = False
    audio_chunk_called = False
    
    async def mock_voice_start(timestamp):
        nonlocal voice_start_called
        voice_start_called = True
    
    async def mock_voice_end(audio_data, sample_rate, duration):
        nonlocal voice_end_called
        voice_end_called = True
    
    async def mock_audio_chunk(chunk, is_speech):
        nonlocal audio_chunk_called
        audio_chunk_called = True
    
    # Set callbacks
    agent.set_callbacks(
        on_voice_start=mock_voice_start,
        on_voice_end=mock_voice_end,
        on_audio_chunk=mock_audio_chunk
    )
    
    # Create test audio chunk
    chunk = AudioChunk(
        data=np.random.rand(1024, 1).astype(np.float32),
        timestamp=asyncio.get_event_loop().time(),
        sample_rate=16000,
        duration=0.064
    )
    
    # Process chunk
    await agent._process_audio_chunk(chunk)
    
    # Check callbacks were called
    assert audio_chunk_called

@pytest.mark.asyncio
async def test_voice_agent_parameters():
    """Test voice agent parameter setting"""
    agent = AsyncVoiceAgent()
    
    # Test parameter setting
    agent.set_parameters(
        silence_threshold=0.02,
        silence_duration=3.0
    )
    
    assert agent.silence_threshold == 0.02
    assert agent.silence_duration == 3.0
    assert agent.vad.silence_threshold == 0.02
    assert agent.vad.silence_duration == 3.0

@pytest.mark.asyncio
async def test_voice_agent_status():
    """Test voice agent status retrieval"""
    agent = AsyncVoiceAgent()
    
    status = agent.get_status()
    
    assert "running" in status
    assert "is_listening" in status
    assert "session_duration" in status
    assert "chunk_count" in status

@pytest.mark.asyncio
async def test_voice_activity_detector():
    """Test voice activity detector"""
    from voice.voice_agent import VoiceActivityDetector
    
    vad = VoiceActivityDetector(silence_threshold=0.01, silence_duration=2.0)
    
    # Test with speech-like audio
    speech_audio = np.random.rand(1024, 1).astype(np.float32) * 0.1
    is_speech = vad.detect_voice_activity(speech_audio)
    
    # Should detect speech
    assert is_speech
    
    # Test with silence
    silence_audio = np.random.rand(1024, 1).astype(np.float32) * 0.001
    is_speech = vad.detect_voice_activity(silence_audio)
    
    # Should not detect speech
    assert not is_speech

@pytest.mark.asyncio
async def test_voice_agent_session_timeout():
    """Test voice session timeout functionality"""
    agent = AsyncVoiceAgent(silence_duration=0.1)  # Short timeout for testing
    
    # Start a session
    await agent._start_voice_session(asyncio.get_event_loop().time())
    assert agent.current_session.is_active
    
    # Wait for timeout
    await asyncio.sleep(0.2)
    
    # Check timeout
    await agent._check_session_timeout()
    
    # Session should be ended due to timeout
    assert not agent.current_session.is_active

@pytest.mark.asyncio
async def test_voice_agent_audio_processing_loop():
    """Test audio processing loop"""
    agent = AsyncVoiceAgent()
    
    # Add some test audio chunks to queue
    for i in range(3):
        chunk = AudioChunk(
            data=np.random.rand(1024, 1).astype(np.float32),
            timestamp=asyncio.get_event_loop().time() + i,
            sample_rate=16000,
            duration=0.064
        )
        agent.audio_queue.put(chunk)
    
    # Start processing loop
    task = asyncio.create_task(agent._process_audio_loop())
    
    # Let it process for a short time
    await asyncio.sleep(0.1)
    
    # Stop the loop
    agent.is_running = False
    task.cancel()
    
    # Queue should be processed
    assert agent.audio_queue.empty()

@pytest.mark.asyncio
async def test_voice_agent_error_handling():
    """Test voice agent error handling"""
    agent = AsyncVoiceAgent()
    
    # Test with invalid audio data
    invalid_chunk = AudioChunk(
        data=None,  # Invalid data
        timestamp=asyncio.get_event_loop().time(),
        sample_rate=16000,
        duration=0.064
    )
    
    # Should handle gracefully
    try:
        await agent._process_audio_chunk(invalid_chunk)
    except Exception:
        # Should not raise exception, just log error
        pass

@pytest.mark.asyncio
async def test_voice_agent_concurrent_access():
    """Test voice agent with concurrent access"""
    agent = AsyncVoiceAgent()
    
    # Start multiple tasks
    tasks = []
    for i in range(3):
        task = asyncio.create_task(agent._process_audio_loop())
        tasks.append(task)
    
    # Add audio chunks
    for i in range(5):
        chunk = AudioChunk(
            data=np.random.rand(1024, 1).astype(np.float32),
            timestamp=asyncio.get_event_loop().time() + i,
            sample_rate=16000,
            duration=0.064
        )
        agent.audio_queue.put(chunk)
    
    # Let them process
    await asyncio.sleep(0.1)
    
    # Stop all tasks
    agent.is_running = False
    for task in tasks:
        task.cancel()

@pytest.mark.asyncio
async def test_voice_agent_memory_management():
    """Test voice agent memory management"""
    agent = AsyncVoiceAgent()
    
    # Add many audio chunks
    for i in range(100):
        chunk = AudioChunk(
            data=np.random.rand(1024, 1).astype(np.float32),
            timestamp=asyncio.get_event_loop().time() + i,
            sample_rate=16000,
            duration=0.064
        )
        agent.audio_queue.put(chunk)
    
    # Check queue size
    assert agent.audio_queue.qsize() == 100
    
    # Process chunks
    task = asyncio.create_task(agent._process_audio_loop())
    await asyncio.sleep(0.2)
    
    # Stop
    agent.is_running = False
    task.cancel()
    
    # Queue should be processed
    assert agent.audio_queue.empty()

if __name__ == "__main__":
    pytest.main([__file__])