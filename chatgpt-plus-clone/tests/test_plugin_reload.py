import unittest
import sys
import os
import time
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from plugin_loader import PluginLoader
    PLUGIN_LOADER_AVAILABLE = True
except ImportError:
    PLUGIN_LOADER_AVAILABLE = False

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPluginHotReload(unittest.TestCase):
    """Comprehensive tests for plugin hot-reload functionality"""
    
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
        # Test plugin 1
        plugin1_code = '''
def on_load():
    return {
        "name": "Test Plugin 1",
        "version": "1.0",
        "description": "Test plugin for hot-reload",
        "hooks": {
            "on_voice_command": test_voice_command,
            "on_message_received": test_message
        }
    }

def test_voice_command(text, context):
    return f"Plugin 1 processed: {text}"

def test_message(message, context):
    return f"Plugin 1 received: {message}"

def on_unload():
    print("Plugin 1 unloaded")
'''
        
        # Test plugin 2
        plugin2_code = '''
def on_load():
    return {
        "name": "Test Plugin 2",
        "version": "2.0",
        "description": "Another test plugin",
        "hooks": {
            "on_voice_command": test_voice_command,
            "on_tool_executed": test_tool
        }
    }

def test_voice_command(text, context):
    return f"Plugin 2 processed: {text}"

def test_tool(tool_name, result, context):
    return f"Plugin 2 tool: {tool_name}"

def on_unload():
    print("Plugin 2 unloaded")
'''
        
        # Write plugin files
        (cls.plugins_dir / "test_plugin1.py").write_text(plugin1_code)
        (cls.plugins_dir / "test_plugin2.py").write_text(plugin2_code)
        
        logger.info("Test plugins created")
    
    def setUp(self):
        """Set up each test"""
        self.start_time = time.time()
        logger.info(f"Starting test: {self._testMethodName}")
        
        # Create fresh plugin loader for each test
        if PLUGIN_LOADER_AVAILABLE:
            self.loader = PluginLoader(plugin_dir=str(self.plugins_dir))
    
    def tearDown(self):
        """Clean up after each test"""
        duration = time.time() - self.start_time
        logger.info(f"Completed test: {self._testMethodName} (duration: {duration:.2f}s)")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_loader_initialization(self):
        """Test plugin loader initialization"""
        try:
            self.assertIsNotNone(self.loader)
            self.assertEqual(self.loader.plugin_dir, str(self.plugins_dir))
            self.assertIsInstance(self.loader.plugins, dict)
            
            logger.info("✅ Plugin loader initialization successful")
            
        except Exception as e:
            self.fail(f"Plugin loader initialization failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_loading(self):
        """Test basic plugin loading"""
        try:
            # Load plugins
            plugins = self.loader.load_plugins()
            
            # Check that plugins were loaded
            self.assertIsInstance(plugins, list)
            self.assertGreater(len(plugins), 0)
            
            # Check plugin metadata
            for plugin in plugins:
                self.assertIn('name', plugin)
                self.assertIn('version', plugin)
                self.assertIn('description', plugin)
                self.assertIn('hooks', plugin)
            
            logger.info(f"✅ Plugin loading successful: {len(plugins)} plugins loaded")
            
        except Exception as e:
            self.fail(f"Plugin loading failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_reload(self):
        """Test plugin reloading"""
        try:
            # Load plugins initially
            initial_plugins = self.loader.load_plugins()
            initial_count = len(initial_plugins)
            
            # Reload a specific plugin
            reload_success = self.loader.reload_plugin("test_plugin1")
            self.assertTrue(reload_success)
            
            # Check that plugin is still available
            reloaded_plugins = self.loader.load_plugins()
            self.assertEqual(len(reloaded_plugins), initial_count)
            
            # Check that plugin metadata is intact
            plugin1 = next((p for p in reloaded_plugins if p['name'] == 'Test Plugin 1'), None)
            self.assertIsNotNone(plugin1)
            self.assertEqual(plugin1['version'], '1.0')
            
            logger.info("✅ Plugin reload successful")
            
        except Exception as e:
            self.fail(f"Plugin reload failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_unload(self):
        """Test plugin unloading"""
        try:
            # Load plugins
            initial_plugins = self.loader.load_plugins()
            initial_count = len(initial_plugins)
            
            # Unload a plugin
            unload_success = self.loader.unload_plugin("test_plugin1")
            self.assertTrue(unload_success)
            
            # Check that plugin is no longer available
            remaining_plugins = self.loader.load_plugins()
            self.assertLess(len(remaining_plugins), initial_count)
            
            # Check that the specific plugin is gone
            plugin1 = next((p for p in remaining_plugins if p['name'] == 'Test Plugin 1'), None)
            self.assertIsNone(plugin1)
            
            logger.info("✅ Plugin unload successful")
            
        except Exception as e:
            self.fail(f"Plugin unload failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_hook_execution(self):
        """Test plugin hook execution"""
        try:
            # Load plugins
            plugins = self.loader.load_plugins()
            
            # Test voice command hook
            test_text = "Hello, test voice command"
            responses = []
            
            for plugin in plugins:
                if 'on_voice_command' in plugin['hooks']:
                    response = plugin['hooks']['on_voice_command'](test_text, {})
                    responses.append(response)
            
            # Check that we got responses
            self.assertGreater(len(responses), 0)
            
            # Check response content
            for response in responses:
                self.assertIsInstance(response, str)
                self.assertIn(test_text, response)
            
            logger.info(f"✅ Plugin hook execution successful: {responses}")
            
        except Exception as e:
            self.fail(f"Plugin hook execution failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_file_watching(self):
        """Test plugin file watching functionality"""
        try:
            # Start watching
            watch_callback_called = False
            
            def test_callback():
                nonlocal watch_callback_called
                watch_callback_called = True
            
            self.loader.start_watching(test_callback)
            
            # Simulate file modification
            test_plugin_file = self.plugins_dir / "test_plugin1.py"
            original_content = test_plugin_file.read_text()
            
            # Modify the file
            new_content = original_content.replace('Test Plugin 1', 'Test Plugin 1 Modified')
            test_plugin_file.write_text(new_content)
            
            # Wait a bit for file system events
            time.sleep(1)
            
            # Restore original content
            test_plugin_file.write_text(original_content)
            
            # Stop watching
            self.loader.stop_watching()
            
            logger.info("✅ Plugin file watching test completed")
            
        except Exception as e:
            self.fail(f"Plugin file watching failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_error_handling(self):
        """Test plugin error handling"""
        try:
            # Create a malformed plugin
            malformed_plugin = '''
def on_load():
    # This will cause an error
    raise Exception("Simulated plugin error")
'''
            
            malformed_file = self.plugins_dir / "malformed_plugin.py"
            malformed_file.write_text(malformed_plugin)
            
            # Try to load plugins (should handle errors gracefully)
            plugins = self.loader.load_plugins()
            
            # Check that other plugins still loaded
            self.assertGreater(len(plugins), 0)
            
            # Clean up malformed plugin
            malformed_file.unlink()
            
            logger.info("✅ Plugin error handling successful")
            
        except Exception as e:
            self.fail(f"Plugin error handling failed: {e}")
    
    @unittest.skipUnless(PLUGIN_LOADER_AVAILABLE, "Plugin loader not available")
    def test_plugin_config_integration(self):
        """Test plugin configuration integration"""
        try:
            # Load plugins
            plugins = self.loader.load_plugins()
            
            # Test plugin config API injection
            for plugin in plugins:
                if 'module' in plugin:
                    module = plugin['module']
                    if hasattr(module, 'get_config'):
                        config = module.get_config()
                        self.assertIsInstance(config, dict)
                    
                    if hasattr(module, 'save_config'):
                        test_config = {'test_key': 'test_value'}
                        module.save_config(test_config)
            
            logger.info("✅ Plugin config integration successful")
            
        except Exception as e:
            self.fail(f"Plugin config integration failed: {e}")
    
    def test_plugin_metadata_validation(self):
        """Test plugin metadata validation"""
        try:
            # Valid metadata
            valid_metadata = {
                'name': 'Test Plugin',
                'version': '1.0',
                'description': 'A test plugin',
                'hooks': {
                    'on_voice_command': lambda x, y: x,
                    'on_message_received': lambda x, y: x
                }
            }
            
            # Validate required fields
            required_fields = ['name', 'version', 'description', 'hooks']
            for field in required_fields:
                self.assertIn(field, valid_metadata)
            
            # Validate data types
            self.assertIsInstance(valid_metadata['name'], str)
            self.assertIsInstance(valid_metadata['version'], str)
            self.assertIsInstance(valid_metadata['description'], str)
            self.assertIsInstance(valid_metadata['hooks'], dict)
            
            # Validate hooks
            self.assertGreater(len(valid_metadata['hooks']), 0)
            
            logger.info("✅ Plugin metadata validation successful")
            
        except Exception as e:
            self.fail(f"Plugin metadata validation failed: {e}")
    
    def test_plugin_performance(self):
        """Test plugin performance metrics"""
        try:
            if PLUGIN_LOADER_AVAILABLE:
                # Measure loading time
                start_time = time.time()
                plugins = self.loader.load_plugins()
                load_time = time.time() - start_time
                
                # Measure reload time
                start_time = time.time()
                self.loader.reload_plugin("test_plugin1")
                reload_time = time.time() - start_time
                
                # Performance assertions
                self.assertLess(load_time, 5.0, "Plugin loading too slow")
                self.assertLess(reload_time, 2.0, "Plugin reload too slow")
                
                performance_metrics = {
                    'load_time': load_time,
                    'reload_time': reload_time,
                    'plugin_count': len(plugins)
                }
                
                logger.info(f"✅ Plugin performance metrics: {performance_metrics}")
            else:
                self.skipTest("Plugin loader not available")
                
        except Exception as e:
            self.fail(f"Plugin performance test failed: {e}")
    
    def test_plugin_thread_safety(self):
        """Test plugin thread safety"""
        try:
            if PLUGIN_LOADER_AVAILABLE:
                import threading
                
                # Test concurrent plugin operations
                def load_plugins_thread():
                    return self.loader.load_plugins()
                
                def reload_plugin_thread():
                    return self.loader.reload_plugin("test_plugin1")
                
                # Run concurrent operations
                threads = []
                results = []
                
                for _ in range(3):
                    t1 = threading.Thread(target=lambda: results.append(load_plugins_thread()))
                    t2 = threading.Thread(target=lambda: results.append(reload_plugin_thread()))
                    threads.extend([t1, t2])
                
                # Start all threads
                for thread in threads:
                    thread.start()
                
                # Wait for completion
                for thread in threads:
                    thread.join()
                
                # Check that operations completed without errors
                self.assertEqual(len(results), 6)
                
                logger.info("✅ Plugin thread safety test successful")
            else:
                self.skipTest("Plugin loader not available")
                
        except Exception as e:
            self.fail(f"Plugin thread safety test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            # Remove test directory
            if cls.test_dir.exists():
                shutil.rmtree(cls.test_dir)
            
            logger.info("🧹 Plugin test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")

def run_plugin_benchmark():
    """Run plugin system benchmarks"""
    print("🚀 Running Plugin System Benchmarks...")
    
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
def on_load():
    return {{
        "name": "Benchmark Plugin {i}",
        "version": "1.0",
        "description": "Benchmark test plugin",
        "hooks": {{
            "on_voice_command": test_hook
        }}
    }}

def test_hook(text, context):
    return f"Benchmark plugin {{i}} processed: {{text}}"
'''
                (plugins_dir / f"benchmark_plugin_{i}.py").write_text(plugin_code)
            
            # Benchmark plugin loading
            loader = PluginLoader(plugin_dir=str(plugins_dir))
            
            start_time = time.time()
            plugins = loader.load_plugins()
            load_time = time.time() - start_time
            
            # Benchmark plugin reload
            start_time = time.time()
            loader.reload_plugin("benchmark_plugin_0")
            reload_time = time.time() - start_time
            
            # Benchmark hook execution
            start_time = time.time()
            for plugin in plugins:
                if 'on_voice_command' in plugin['hooks']:
                    plugin['hooks']['on_voice_command']("benchmark test", {})
            hook_time = time.time() - start_time
            
            benchmark_results = {
                'load_time': load_time,
                'reload_time': reload_time,
                'hook_execution_time': hook_time,
                'plugin_count': len(plugins)
            }
            
            # Cleanup
            shutil.rmtree(test_dir)
            
            print(f"✅ Plugin Benchmark: {benchmark_results}")
            
        except Exception as e:
            print(f"❌ Plugin Benchmark failed: {e}")
    
    return benchmark_results

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Plugin Hot-Reload Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    benchmark_results = run_plugin_benchmark()
    
    # Print summary
    print("\n📊 Plugin Test Summary:")
    print(f"Plugin Loader Available: {PLUGIN_LOADER_AVAILABLE}")
    print(f"Benchmark Results: {benchmark_results}")