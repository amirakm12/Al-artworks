"""
Plugin Loader - No Sandbox Version
Unrestricted plugin loading with full system access
WARNING: This version provides NO security isolation
"""

import importlib
import sys
import os
import traceback
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Hot-reload imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("Watchdog not available - hot-reload disabled")

from config_manager import ConfigManager

class PluginLoader:
    """Unrestricted plugin loader with full system access"""
    
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        self.plugins = {}
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager()
        
        # Threading for hot-reload
        self.lock = threading.Lock()
        self.observer = None
        self.watching = False
        self.reload_callback = None
        
        self.logger.warning("⚠️  NO-SANDBOX MODE: Plugins have full system access!")
    
    def load_plugins(self) -> List[Any]:
        """Load all plugins with full system access"""
        plugins = []
        
        with self.lock:
            self.plugins.clear()
            
            # Add plugins directory to Python path
            sys.path.insert(0, str(self.plugin_dir))
            
            try:
                for filename in os.listdir(self.plugin_dir):
                    if filename.endswith(".py") and not filename.startswith("_"):
                        plugin_name = filename[:-3]
                        
                        try:
                            # Import module with full access
                            module = importlib.import_module(plugin_name)
                            
                            # Inject config API if available
                            if hasattr(module, "set_config_api"):
                                module.set_config_api(
                                    get_config=lambda: self.config.get_plugin_config(plugin_name),
                                    save_config=lambda cfg: self.config.set_plugin_config(plugin_name, cfg)
                                )
                            
                            # Call setup function if available
                            if hasattr(module, "setup"):
                                module.setup()
                                self.logger.info(f"✅ Setup called for plugin: {plugin_name}")
                            
                            # Check for Plugin class
                            if hasattr(module, 'Plugin'):
                                plugin_instance = module.Plugin()
                                plugin_data = {
                                    'name': plugin_name,
                                    'module': module,
                                    'instance': plugin_instance,
                                    'path': str(self.plugin_dir / filename),
                                    'type': 'class'
                                }
                                plugins.append(plugin_data)
                                self.plugins[plugin_name] = plugin_data
                                self.logger.info(f"✅ Loaded plugin class: {plugin_name}")
                            
                            # Check for on_load function (legacy)
                            elif hasattr(module, 'on_load'):
                                metadata = module.on_load()
                                metadata['module'] = module
                                metadata['path'] = str(self.plugin_dir / filename)
                                metadata['name'] = plugin_name
                                metadata['type'] = 'function'
                                plugins.append(metadata)
                                self.plugins[plugin_name] = metadata
                                self.logger.info(f"✅ Loaded plugin function: {plugin_name}")
                            
                            else:
                                # Basic module loading
                                plugin_data = {
                                    'name': plugin_name,
                                    'module': module,
                                    'path': str(self.plugin_dir / filename),
                                    'type': 'module'
                                }
                                plugins.append(plugin_data)
                                self.plugins[plugin_name] = plugin_data
                                self.logger.info(f"✅ Loaded plugin module: {plugin_name}")
                            
                        except Exception as e:
                            self.logger.error(f"❌ Failed to load plugin {plugin_name}: {e}")
                            traceback.print_exc()
                
                # Remove plugins directory from path
                if str(self.plugin_dir) in sys.path:
                    sys.path.remove(str(self.plugin_dir))
                
                self.logger.info(f"🎯 Loaded {len(plugins)} plugins with full system access")
                return plugins
                
            except Exception as e:
                self.logger.error(f"❌ Error loading plugins: {e}")
                traceback.print_exc()
                return []
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        try:
            if plugin_name in self.plugins:
                plugin_data = self.plugins[plugin_name]
                
                # Stop plugin if it has stop method
                if plugin_data['type'] == 'class' and hasattr(plugin_data['instance'], 'stop'):
                    plugin_data['instance'].stop()
                
                # Reload the module
                importlib.reload(plugin_data['module'])
                
                # Recreate instance if it's a class
                if plugin_data['type'] == 'class' and hasattr(plugin_data['module'], 'Plugin'):
                    plugin_data['instance'] = plugin_data['module'].Plugin()
                
                # Call setup again
                if hasattr(plugin_data['module'], 'setup'):
                    plugin_data['module'].setup()
                
                # Start plugin if it has start method
                if plugin_data['type'] == 'class' and hasattr(plugin_data['instance'], 'start'):
                    plugin_data['instance'].start()
                
                self.logger.info(f"🔄 Reloaded plugin: {plugin_name}")
                return True
            else:
                self.logger.warning(f"⚠️  Plugin not found for reload: {plugin_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to reload plugin {plugin_name}: {e}")
            traceback.print_exc()
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        try:
            if plugin_name in self.plugins:
                plugin_data = self.plugins[plugin_name]
                
                # Stop plugin if it has stop method
                if plugin_data['type'] == 'class' and hasattr(plugin_data['instance'], 'stop'):
                    plugin_data['instance'].stop()
                
                # Remove from modules
                if plugin_name in sys.modules:
                    del sys.modules[plugin_name]
                
                # Remove from plugins dict
                del self.plugins[plugin_name]
                
                self.logger.info(f"🗑️  Unloaded plugin: {plugin_name}")
                return True
            else:
                self.logger.warning(f"⚠️  Plugin not found for unload: {plugin_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to unload plugin {plugin_name}: {e}")
            traceback.print_exc()
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific plugin"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugins"""
        return list(self.plugins.keys())
    
    def call_plugin_function(self, plugin_name: str, function_name: str, *args, **kwargs) -> Any:
        """Call a function in a plugin with full access"""
        try:
            if plugin_name in self.plugins:
                plugin_data = self.plugins[plugin_name]
                
                if plugin_data['type'] == 'class':
                    # Call method on plugin instance
                    if hasattr(plugin_data['instance'], function_name):
                        return getattr(plugin_data['instance'], function_name)(*args, **kwargs)
                else:
                    # Call function in module
                    if hasattr(plugin_data['module'], function_name):
                        return getattr(plugin_data['module'], function_name)(*args, **kwargs)
                
                self.logger.warning(f"⚠️  Function {function_name} not found in plugin {plugin_name}")
                return None
            else:
                self.logger.warning(f"⚠️  Plugin {plugin_name} not found")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error calling {function_name} in plugin {plugin_name}: {e}")
            traceback.print_exc()
            return None
    
    def start_watching(self, reload_callback: Optional[callable] = None):
        """Start watching for plugin file changes"""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("⚠️  Watchdog not available - hot-reload disabled")
            return
        
        try:
            self.reload_callback = reload_callback
            event_handler = PluginChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, str(self.plugin_dir), recursive=False)
            self.observer.start()
            self.watching = True
            
            self.logger.info(f"🔄 Started watching {self.plugin_dir} for changes")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start plugin watcher: {e}")
    
    def stop_watching(self):
        """Stop watching for plugin changes"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watching = False
            self.logger.info("🛑 Stopped plugin watcher")
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get detailed information about a plugin"""
        if plugin_name not in self.plugins:
            return {}
        
        plugin_data = self.plugins[plugin_name]
        info = {
            'name': plugin_name,
            'type': plugin_data['type'],
            'path': plugin_data['path'],
            'module': str(plugin_data['module']),
        }
        
        # Add available methods/functions
        if plugin_data['type'] == 'class':
            methods = [attr for attr in dir(plugin_data['instance']) 
                      if not attr.startswith('_') and callable(getattr(plugin_data['instance'], attr))]
            info['methods'] = methods
        else:
            functions = [attr for attr in dir(plugin_data['module']) 
                       if not attr.startswith('_') and callable(getattr(plugin_data['module'], attr))]
            info['functions'] = functions
        
        return info

class PluginChangeHandler(FileSystemEventHandler):
    """Handler for plugin file system events"""
    
    def __init__(self, plugin_loader: PluginLoader):
        super().__init__()
        self.plugin_loader = plugin_loader
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.src_path.endswith(".py"):
            plugin_name = Path(event.src_path).stem
            self.logger.info(f"🔄 Plugin file changed: {plugin_name}")
            
            # Reload the plugin
            if self.plugin_loader.reload_plugin(plugin_name):
                # Call reload callback if provided
                if self.plugin_loader.reload_callback:
                    self.plugin_loader.reload_callback()
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.src_path.endswith(".py"):
            plugin_name = Path(event.src_path).stem
            self.logger.info(f"➕ New plugin file created: {plugin_name}")
            
            # Reload all plugins to include the new one
            if self.plugin_loader.reload_callback:
                self.plugin_loader.reload_callback()
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.src_path.endswith(".py"):
            plugin_name = Path(event.src_path).stem
            self.logger.info(f"🗑️  Plugin file deleted: {plugin_name}")
            
            # Unload the plugin
            self.plugin_loader.unload_plugin(plugin_name)
            
            # Call reload callback if provided
            if self.plugin_loader.reload_callback:
                self.plugin_loader.reload_callback()

# Convenience functions
def load_plugins(plugin_dir="plugins") -> List[Any]:
    """Load all plugins with full system access"""
    loader = PluginLoader(plugin_dir)
    return loader.load_plugins()

def reload_plugin(plugin_name: str, plugin_dir="plugins") -> bool:
    """Reload a specific plugin"""
    loader = PluginLoader(plugin_dir)
    return loader.reload_plugin(plugin_name)

def unload_plugin(plugin_name: str, plugin_dir="plugins") -> bool:
    """Unload a specific plugin"""
    loader = PluginLoader(plugin_dir)
    return loader.unload_plugin(plugin_name)

if __name__ == "__main__":
    # Test the no-sandbox plugin loader
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing No-Sandbox Plugin Loader...")
    print("⚠️  WARNING: This version provides NO security isolation!")
    
    loader = PluginLoader()
    plugins = loader.load_plugins()
    
    print(f"✅ Loaded {len(plugins)} plugins")
    
    for plugin in plugins:
        print(f"  - {plugin['name']} ({plugin['type']})")
    
    print("\n🔍 Plugin information:")
    for plugin_name in loader.list_plugins():
        info = loader.get_plugin_info(plugin_name)
        print(f"  {plugin_name}: {info}")