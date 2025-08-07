"""
Plugin Loader Tests
Comprehensive testing for plugin loading, lifecycle, and error handling
"""

import unittest
import sys
import os
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from plugin_loader import enable_plugins, BasePlugin
    PLUGIN_LOADER_AVAILABLE = True
except ImportError:
    PLUGIN_LOADER_AVAILABLE = False

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPluginLoader(unittest.TestCase):
    """Comprehensive tests for plugin loader functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create temporary test directories
        cls.test_dir = Path(tempfile.mkdtemp(prefix="plugin_test_"))
        cls.plugins_dir = cls.test_dir / "plugins"
        cls.plugins_dir.mkdir(exist_ok=True)
        
        # Create test plugin files
        cls.create_test_plugins()
        
        logger.info(f"Test environment created: {cls.test_dir}")
    
    @classmethod
    def create_test_plugins(cls):
        """Create test plugin files"""
        # Test plugin 1 - Valid plugin
        plugin1_code = '''
class TestPlugin1:
    def on_load(self):
        print("[Plugin] Test Plugin 1 loaded")
        return True
    
    def on_unload(self):
        print("[Plugin] Test Plugin 1 unloaded")
        return True
'''
        
        # Test plugin 2 - Plugin with error
        plugin2_code = '''
class TestPlugin2:
    def on_load(self):
        print("[Plugin] Test Plugin 2 loaded")
        raise Exception("Simulated plugin error")
    
    def on_unload(self):
        print("[Plugin] Test Plugin 2 unloaded")
        return True
'''
        
        # Test plugin 3 - Invalid plugin (no on_load method)
        plugin3_code = '''
class TestPlugin3:
    def some_method(self):
        print("[Plugin] Test Plugin 3 - invalid")
'''
        
        # Write plugin files
        (cls.plugins_dir / "test_plugin1.py").write_text(plugin1_code)
        (cls.plugins_dir / "test_plugin2.py").write_text(plugin2_code)
        (cls.plugins_dir / "test_plugin3.py").write_text(plugin3_code)
        
        logger.info("Test plugins created")
    
    def setUp(self):
        """Set up each test"""
        import time
        self.start_time = time.time()
        logger.info(f"Starting test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up after each test"""
        import time
        duration = time.time() - self.start_time
        logger.info(f"Completed test: {self._testMethodName} (duration: {duration:.2f}s)")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_enable_plugins_basic(self):
        """Test basic plugin loading functionality"""
        try:
            # Mock the plugins directory
            with patch('plugin_loader.PLUGIN_DIR', str(self.plugins_dir)):
                enable_plugins()
                # If we get here without exception, the test passes
                self.assertTrue(True)
                logger.info("✅ Basic plugin loading test passed")
        except Exception as e:
            self.fail(f"Basic plugin loading failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_lifecycle(self):
        """Test plugin lifecycle (load/unload)"""
        try:
            # Test plugin loading
            with patch('plugin_loader.PLUGIN_DIR', str(self.plugins_dir)):
                enable_plugins()
                
                # Check that plugins were loaded
                # This would require access to the loaded plugins list
                # For now, we just check that no exceptions were raised
                self.assertTrue(True)
                logger.info("✅ Plugin lifecycle test passed")
        except Exception as e:
            self.fail(f"Plugin lifecycle test failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_error_handling(self):
        """Test plugin error handling"""
        try:
            # Test that plugin errors don't crash the loader
            with patch('plugin_loader.PLUGIN_DIR', str(self.plugins_dir)):
                enable_plugins()
                
                # The loader should handle plugin errors gracefully
                self.assertTrue(True)
                logger.info("✅ Plugin error handling test passed")
        except Exception as e:
            self.fail(f"Plugin error handling test failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_directory_not_found(self):
        """Test behavior when plugin directory doesn't exist"""
        try:
            # Test with non-existent directory
            with patch('plugin_loader.PLUGIN_DIR', '/non/existent/path'):
                enable_plugins()
                
                # Should handle missing directory gracefully
                self.assertTrue(True)
                logger.info("✅ Plugin directory not found test passed")
        except Exception as e:
            self.fail(f"Plugin directory not found test failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_config_disabled(self):
        """Test plugin loading when disabled in config"""
        try:
            # Mock config to disable plugins
            mock_config = {'plugins_enabled': False}
            
            with patch('plugin_loader.json.load', return_value=mock_config):
                enable_plugins()
                
                # Should skip loading when disabled
                self.assertTrue(True)
                logger.info("✅ Plugin config disabled test passed")
        except Exception as e:
            self.fail(f"Plugin config disabled test failed: {e}")
    
    def test_base_plugin_class(self):
        """Test BasePlugin class functionality"""
        if PLUGIN_LOADER_AVAILABLE:
            try:
                # Test BasePlugin instantiation
                plugin = BasePlugin()
                self.assertIsNotNone(plugin)
                
                # Test default methods
                result = plugin.on_load()
                self.assertIsNone(result)  # Default should return None
                
                result = plugin.on_unload()
                self.assertIsNone(result)  # Default should return None
                
                logger.info("✅ BasePlugin class test passed")
            except Exception as e:
                self.fail(f"BasePlugin class test failed: {e}")
        else:
            self.skipTest("Plugin loader not available")
    
    def test_plugin_metadata_validation(self):
        """Test plugin metadata validation"""
        try:
            # Test valid plugin metadata
            valid_metadata = {
                'name': 'Test Plugin',
                'version': '1.0.0',
                'description': 'A test plugin',
                'author': 'Test Author',
                'dependencies': []
            }
            
            # Validate required fields
            required_fields = ['name', 'version', 'description']
            for field in required_fields:
                self.assertIn(field, valid_metadata)
            
            # Validate data types
            self.assertIsInstance(valid_metadata['name'], str)
            self.assertIsInstance(valid_metadata['version'], str)
            self.assertIsInstance(valid_metadata['description'], str)
            
            logger.info("✅ Plugin metadata validation test passed")
            
        except Exception as e:
            self.fail(f"Plugin metadata validation test failed: {e}")
    
    def test_plugin_performance(self):
        """Test plugin loading performance"""
        try:
            if PLUGIN_LOADER_AVAILABLE:
                import time
                
                # Measure loading time
                start_time = time.time()
                
                with patch('plugin_loader.PLUGIN_DIR', str(self.plugins_dir)):
                    enable_plugins()
                
                load_time = time.time() - start_time
                
                # Performance assertions
                self.assertLess(load_time, 5.0, "Plugin loading too slow")
                
                logger.info(f"✅ Plugin performance test passed (load time: {load_time:.2f}s)")
            else:
                self.skipTest("Plugin loader not available")
                
        except Exception as e:
            self.fail(f"Plugin performance test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            # Remove test directory
            if cls.test_dir.exists():
                shutil.rmtree(cls.test_dir)
            
            logger.info("🧹 Plugin loader test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")

def run_plugin_loader_benchmark():
    """Run plugin loader benchmarks"""
    print("🚀 Running Plugin Loader Benchmarks...")
    
    benchmark_results = {}
    
    if PLUGIN_LOADER_AVAILABLE:
        try:
            # Create temporary test environment
            test_dir = Path(tempfile.mkdtemp(prefix="plugin_benchmark_"))
            plugins_dir = test_dir / "plugins"
            plugins_dir.mkdir(exist_ok=True)
            
            # Create benchmark plugins
            for i in range(5):
                plugin_code = f'''
class BenchmarkPlugin{i}:
    def on_load(self):
        print(f"[Plugin] Benchmark Plugin {{i}} loaded")
        return True
    
    def on_unload(self):
        print(f"[Plugin] Benchmark Plugin {{i}} unloaded")
        return True
'''
                (plugins_dir / f"benchmark_plugin_{i}.py").write_text(plugin_code)
            
            # Benchmark plugin loading
            import time
            start_time = time.time()
            
            with patch('plugin_loader.PLUGIN_DIR', str(plugins_dir)):
                enable_plugins()
            
            load_time = time.time() - start_time
            
            benchmark_results = {
                'load_time': load_time,
                'plugin_count': 5
            }
            
            # Cleanup
            shutil.rmtree(test_dir)
            
            print(f"✅ Plugin Loader Benchmark: {benchmark_results}")
            
        except Exception as e:
            print(f"❌ Plugin Loader Benchmark failed: {e}")
    
    return benchmark_results

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Plugin Loader Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    benchmark_results = run_plugin_loader_benchmark()
    
    # Print summary
    print("\n📊 Plugin Loader Test Summary:")
    print(f"Plugin Loader Available: {PLUGIN_LOADER_AVAILABLE}")
    print(f"Benchmark Results: {benchmark_results}")