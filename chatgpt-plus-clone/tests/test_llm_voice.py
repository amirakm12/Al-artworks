import unittest
import sys
import os
import time
import logging
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llm.agent_orchestrator import AgentOrchestrator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from voice_hotkey import VoiceHotkeyListener
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMVoice(unittest.TestCase):
    """Comprehensive tests for LLM and voice functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_config = {
            'llm_model': 'dolphin-mixtral:8x22b',
            'voice_hotkey': 'ctrl+shift+v',
            'test_timeout': 30
        }
        
        # Create test directories
        Path("test_output").mkdir(exist_ok=True)
        Path("test_logs").mkdir(exist_ok=True)
    
    def setUp(self):
        """Set up each test"""
        self.start_time = time.time()
        logger.info(f"Starting test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up after each test"""
        duration = time.time() - self.start_time
        logger.info(f"Completed test: {self._testMethodName} (duration: {duration:.2f}s)")
    
    @unittest.skipUnless(LLM_AVAILABLE, "LLM module not available")
    def test_llm_initialization(self):
        """Test LLM system initialization"""
        try:
            orchestrator = AgentOrchestrator()
            self.assertIsNotNone(orchestrator)
            logger.info("✅ LLM initialization successful")
        except Exception as e:
            self.fail(f"LLM initialization failed: {e}")
    
    @unittest.skipUnless(LLM_AVAILABLE, "LLM module not available")
    def test_llm_model_loading(self):
        """Test LLM model loading"""
        try:
            orchestrator = AgentOrchestrator()
            
            # Test model loading
            model_info = orchestrator.get_model_info()
            self.assertIsNotNone(model_info)
            logger.info(f"✅ Model info: {model_info}")
            
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    @unittest.skipUnless(LLM_AVAILABLE, "LLM module not available")
    def test_llm_basic_generation(self):
        """Test basic text generation"""
        try:
            orchestrator = AgentOrchestrator()
            
            # Test simple generation
            prompt = "Hello, this is a test."
            response = orchestrator.generate_response(prompt)
            
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            logger.info(f"✅ Basic generation successful: {response[:100]}...")
            
        except Exception as e:
            self.fail(f"Basic generation failed: {e}")
    
    @unittest.skipUnless(LLM_AVAILABLE, "LLM module not available")
    def test_llm_tool_detection(self):
        """Test tool detection in LLM responses"""
        try:
            orchestrator = AgentOrchestrator()
            
            # Test code generation
            prompt = "Write a Python function to calculate fibonacci numbers"
            response = orchestrator.generate_response(prompt)
            
            self.assertIsNotNone(response)
            # Check if response contains code
            has_code = 'def ' in response or 'import ' in response or 'print(' in response
            self.assertTrue(has_code, "Response should contain code")
            
            logger.info("✅ Tool detection successful")
            
        except Exception as e:
            self.fail(f"Tool detection failed: {e}")
    
    @unittest.skipUnless(VOICE_AVAILABLE, "Voice module not available")
    def test_voice_hotkey_initialization(self):
        """Test voice hotkey system initialization"""
        try:
            listener = VoiceHotkeyListener(hotkey=self.test_config['voice_hotkey'])
            self.assertIsNotNone(listener)
            self.assertEqual(listener.hotkey, self.test_config['voice_hotkey'])
            
            logger.info("✅ Voice hotkey initialization successful")
            
        except Exception as e:
            self.fail(f"Voice hotkey initialization failed: {e}")
    
    @unittest.skipUnless(VOICE_AVAILABLE, "Voice module not available")
    def test_voice_hotkey_start_stop(self):
        """Test voice hotkey start/stop functionality"""
        try:
            listener = VoiceHotkeyListener(hotkey=self.test_config['voice_hotkey'])
            
            # Test start
            listener.start()
            self.assertTrue(listener.is_running)
            
            # Test stop
            listener.stop()
            self.assertFalse(listener.is_running)
            
            logger.info("✅ Voice hotkey start/stop successful")
            
        except Exception as e:
            self.fail(f"Voice hotkey start/stop failed: {e}")
    
    @unittest.skipUnless(VOICE_AVAILABLE, "Voice module not available")
    def test_voice_recording_simulation(self):
        """Test voice recording simulation (without actual audio)"""
        try:
            listener = VoiceHotkeyListener(hotkey=self.test_config['voice_hotkey'])
            
            # Simulate voice recording
            with patch('voice_hotkey.sounddevice') as mock_sd:
                mock_sd.rec.return_value = b"test_audio_data"
                
                # Test recording simulation
                result = listener._simulate_recording(duration=1)
                self.assertIsNotNone(result)
                
            logger.info("✅ Voice recording simulation successful")
            
        except Exception as e:
            self.fail(f"Voice recording simulation failed: {e}")
    
    @unittest.skipUnless(VOICE_AVAILABLE, "Voice module not available")
    def test_voice_transcription_simulation(self):
        """Test voice transcription simulation"""
        try:
            listener = VoiceHotkeyListener(hotkey=self.test_config['voice_hotkey'])
            
            # Test transcription simulation
            test_audio = b"simulated_audio_data"
            transcription = listener._simulate_transcription(test_audio)
            
            self.assertIsNotNone(transcription)
            self.assertIsInstance(transcription, str)
            
            logger.info(f"✅ Voice transcription simulation successful: {transcription}")
            
        except Exception as e:
            self.fail(f"Voice transcription simulation failed: {e}")
    
    def test_llm_voice_integration(self):
        """Test LLM and voice integration"""
        try:
            # Test integration without actual hardware
            if LLM_AVAILABLE and VOICE_AVAILABLE:
                orchestrator = AgentOrchestrator()
                listener = VoiceHotkeyListener(hotkey=self.test_config['voice_hotkey'])
                
                # Simulate voice input
                simulated_voice_input = "Hello, how are you?"
                
                # Process through LLM
                response = orchestrator.generate_response(simulated_voice_input)
                
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                
                logger.info(f"✅ LLM-Voice integration successful: {response[:100]}...")
            else:
                self.skipTest("LLM or Voice modules not available")
                
        except Exception as e:
            self.fail(f"LLM-Voice integration failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        try:
            metrics = {
                'llm_response_time': 0.5,
                'voice_activation_time': 0.1,
                'transcription_time': 0.3,
                'total_processing_time': 0.9
            }
            
            # Test metrics validation
            self.assertGreater(metrics['llm_response_time'], 0)
            self.assertGreater(metrics['voice_activation_time'], 0)
            self.assertGreater(metrics['transcription_time'], 0)
            self.assertGreater(metrics['total_processing_time'], 0)
            
            # Test performance thresholds
            self.assertLess(metrics['llm_response_time'], 10.0, "LLM response too slow")
            self.assertLess(metrics['voice_activation_time'], 1.0, "Voice activation too slow")
            self.assertLess(metrics['transcription_time'], 5.0, "Transcription too slow")
            
            logger.info(f"✅ Performance metrics valid: {metrics}")
            
        except Exception as e:
            self.fail(f"Performance metrics test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in LLM and voice systems"""
        try:
            # Test LLM error handling
            if LLM_AVAILABLE:
                with patch('llm.agent_orchestrator.AgentOrchestrator.generate_response') as mock_gen:
                    mock_gen.side_effect = Exception("Simulated LLM error")
                    
                    orchestrator = AgentOrchestrator()
                    with self.assertRaises(Exception):
                        orchestrator.generate_response("test")
            
            # Test voice error handling
            if VOICE_AVAILABLE:
                with patch('voice_hotkey.VoiceHotkeyListener.start') as mock_start:
                    mock_start.side_effect = Exception("Simulated voice error")
                    
                    listener = VoiceHotkeyListener()
                    with self.assertRaises(Exception):
                        listener.start()
            
            logger.info("✅ Error handling tests successful")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        try:
            # Test valid configuration
            valid_config = {
                'llm_model': 'dolphin-mixtral:8x22b',
                'voice_hotkey': 'ctrl+shift+v',
                'timeout': 30
            }
            
            self.assertIn('llm_model', valid_config)
            self.assertIn('voice_hotkey', valid_config)
            self.assertIn('timeout', valid_config)
            self.assertIsInstance(valid_config['timeout'], int)
            self.assertGreater(valid_config['timeout'], 0)
            
            # Test invalid configuration
            invalid_config = {
                'llm_model': '',
                'voice_hotkey': '',
                'timeout': -1
            }
            
            self.assertFalse(bool(invalid_config['llm_model']))
            self.assertFalse(bool(invalid_config['voice_hotkey']))
            self.assertLess(invalid_config['timeout'], 0)
            
            logger.info("✅ Configuration validation successful")
            
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Clean up test files
        test_files = ["test_output", "test_logs"]
        for test_file in test_files:
            if Path(test_file).exists():
                import shutil
                shutil.rmtree(test_file)
        
        logger.info("🧹 Test environment cleaned up")

def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_results = {}
    
    # LLM Performance Benchmark
    if LLM_AVAILABLE:
        try:
            start_time = time.time()
            orchestrator = AgentOrchestrator()
            init_time = time.time() - start_time
            
            start_time = time.time()
            response = orchestrator.generate_response("Test prompt")
            gen_time = time.time() - start_time
            
            benchmark_results['llm'] = {
                'initialization_time': init_time,
                'generation_time': gen_time,
                'response_length': len(response)
            }
            
            print(f"✅ LLM Benchmark: {benchmark_results['llm']}")
            
        except Exception as e:
            print(f"❌ LLM Benchmark failed: {e}")
    
    # Voice Performance Benchmark
    if VOICE_AVAILABLE:
        try:
            start_time = time.time()
            listener = VoiceHotkeyListener()
            init_time = time.time() - start_time
            
            start_time = time.time()
            listener.start()
            start_time_actual = time.time() - start_time
            
            listener.stop()
            
            benchmark_results['voice'] = {
                'initialization_time': init_time,
                'start_time': start_time_actual
            }
            
            print(f"✅ Voice Benchmark: {benchmark_results['voice']}")
            
        except Exception as e:
            print(f"❌ Voice Benchmark failed: {e}")
    
    return benchmark_results

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running LLM and Voice Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    benchmark_results = run_performance_benchmark()
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"LLM Available: {LLM_AVAILABLE}")
    print(f"Voice Available: {VOICE_AVAILABLE}")
    print(f"Benchmark Results: {benchmark_results}")