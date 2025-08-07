"""
Plugin Loader - Advanced Plugin Management with Hot-Reload
Loads plugins from plugins/ directory with live reloading capabilities
"""

import os
import importlib.util
import importlib
import logging
import threading
import time
import sys
import traceback
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# Config manager import
from config_manager import ConfigManager

# Hot-reload imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("Watchdog not available - hot-reload disabled")

class PluginReloadHandler(FileSystemEventHandler):
    """File system event handler for plugin hot-reload"""
    
    def __init__(self, reload_callback: Callable):
        super().__init__()
        self.reload_callback = reload_callback
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(".py"):
            # Debounce rapid file changes
            current_time = time.time()
            if event.src_path in self.last_modified:
                if current_time - self.last_modified[event.src_path] < 1.0:
                    return  # Ignore changes within 1 second
            
            self.last_modified[event.src_path] = current_time
            
            print(f"🔄 [Plugin Hot-Reload] Detected change in {event.src_path}")
            self.reload_callback()
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(".py"):
            print(f"➕ [Plugin Hot-Reload] New plugin detected: {event.src_path}")
            self.reload_callback()
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(".py"):
            print(f"🗑️ [Plugin Hot-Reload] Plugin deleted: {event.src_path}")
            self.reload_callback()

class PluginLoader:
    """Advanced plugin loader with hot-reload capabilities"""
    
    def __init__(self):
        self.plugins = []
        self.hooks = {}
        self.logger = logging.getLogger(__name__)
        self.plugins_dir = Path("plugins")
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Config manager
        self.config = ConfigManager()
        
        # Threading for hot-reload
        self.lock = threading.Lock()
        self.observer = None
        self.watching = False
        
        # Callback for reload events
        self.reload_callback = None
    
    def load_plugins(self) -> List[Dict[str, Any]]:
        """Load all plugins from plugins/ directory"""
        with self.lock:
            self.plugins.clear()
            self.hooks.clear()
            
            if not self.plugins_dir.exists():
                self.logger.warning("Plugins directory not found")
                return []
            
            for file in self.plugins_dir.iterdir():
                if file.suffix == ".py" and file.name != "__init__.py":
                    try:
                        plugin_data = self.load_plugin(file)
                        if plugin_data:
                            self.plugins.append(plugin_data)
                            self.register_hooks(plugin_data)
                            self.logger.info(f"✅ Loaded plugin: {plugin_data['name']}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed loading {file.name}: {e}")
            
            return self.plugins.copy()
    
    def load_plugin(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single plugin"""
        try:
            # Load plugin module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Inject config API to plugin if it has set_config_api function
            plugin_name = plugin_path.stem
            if hasattr(module, "set_config_api"):
                module.set_config_api(
                    get_config=lambda: self.config.get_plugin_config(plugin_name),
                    save_config=lambda cfg: self.config.set_plugin_config(plugin_name, cfg)
                )
            
            # Check for Plugin class (new pattern)
            if hasattr(module, 'Plugin'):
                plugin_instance = module.Plugin()
                metadata = {
                    'name': plugin_name,
                    'module': module,
                    'instance': plugin_instance,
                    'path': str(plugin_path),
                    'type': 'class'
                }
                return metadata
            
            # Check for on_load function (legacy pattern)
            elif hasattr(module, 'on_load'):
                metadata = module.on_load()
                metadata['module'] = module
                metadata['path'] = str(plugin_path)
                metadata['name'] = plugin_name
                metadata['type'] = 'function'
                return metadata
            else:
                self.logger.warning(f"Plugin {plugin_path.name} missing Plugin class or on_load function")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_path}: {e}")
            return None
    
    def register_hooks(self, plugin_data: Dict[str, Any]):
        """Register plugin hooks"""
        if 'hooks' in plugin_data:
            for hook_name, hook_func in plugin_data['hooks'].items():
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                self.hooks[hook_name].append({
                    'plugin': plugin_data['name'],
                    'function': hook_func
                })
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all registered hooks for a specific event"""
        results = []
        
        with self.lock:
            if hook_name in self.hooks:
                for hook in self.hooks[hook_name]:
                    try:
                        result = hook['function'](*args, **kwargs)
                        if result is not None:
                            results.append({
                                'plugin': hook['plugin'],
                                'result': result
                            })
                    except Exception as e:
                        self.logger.error(f"Error in hook {hook_name} from {hook['plugin']}: {e}")
        
        return results
    
    def start_watching(self, reload_callback: Callable):
        """Start watching plugins directory for changes"""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available - hot-reload disabled")
            return
        
        if self.watching:
            self.stop_watching()
        
        try:
            self.reload_callback = reload_callback
            event_handler = PluginReloadHandler(self.reload_plugins)
            
            self.observer = Observer()
            self.observer.schedule(event_handler, str(self.plugins_dir), recursive=False)
            self.observer.start()
            
            self.watching = True
            self.logger.info(f"🔄 Started watching {self.plugins_dir} for changes")
            
            # Start watching thread
            def watch_thread():
                try:
                    while self.watching:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.stop_watching()
                finally:
                    if self.observer:
                        self.observer.join()
            
            thread = threading.Thread(target=watch_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting plugin watcher: {e}")
    
    def stop_watching(self):
        """Stop watching plugins directory"""
        if self.observer:
            self.observer.stop()
            self.observer = None
        
        self.watching = False
        self.logger.info("🔄 Stopped watching plugins directory")
    
    def reload_plugins(self):
        """Reload all plugins (called by hot-reload handler)"""
        self.logger.info("🔄 Reloading plugins...")
        
        # Reload plugins
        self.load_plugins()
        
        # Call callback if set
        if self.reload_callback:
            self.reload_callback()
        
        self.logger.info("✅ Plugins reloaded successfully")
    
    def get_plugins(self) -> List[Dict[str, Any]]:
        """Get all loaded plugins"""
        with self.lock:
            return self.plugins.copy()
    
    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific plugin by name"""
        with self.lock:
            for plugin in self.plugins:
                if plugin['name'] == name:
                    return plugin
            return None
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a specific plugin"""
        with self.lock:
            for i, plugin in enumerate(self.plugins):
                if plugin['name'] == name:
                    # Call unload function if available
                    if hasattr(plugin['module'], 'on_unload'):
                        try:
                            plugin['module'].on_unload()
                        except Exception as e:
                            self.logger.error(f"Error unloading plugin {name}: {e}")
                    
                    # Remove from plugins list
                    self.plugins.pop(i)
                    
                    # Remove hooks
                    for hook_name in list(self.hooks.keys()):
                        self.hooks[hook_name] = [
                            hook for hook in self.hooks[hook_name] 
                            if hook['plugin'] != name
                        ]
                    
                    self.logger.info(f"✅ Unloaded plugin: {name}")
                    return True
            
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin loader status"""
        return {
            'plugins_loaded': len(self.plugins),
            'hooks_registered': len(self.hooks),
            'watching': self.watching,
            'hot_reload_available': WATCHDOG_AVAILABLE,
            'plugins_dir': str(self.plugins_dir)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_watching()

# Global plugin loader instance
plugin_loader = PluginLoader()

def load_plugins():
    """Load all plugins"""
    return plugin_loader.load_plugins()

def call_hook(hook_name: str, *args, **kwargs):
    """Call a specific hook"""
    return plugin_loader.call_hook(hook_name, *args, **kwargs)

def get_plugins():
    """Get all loaded plugins"""
    return plugin_loader.get_plugins()

def start_watching(reload_callback: Callable):
    """Start watching plugins directory"""
    return plugin_loader.start_watching(reload_callback)

def stop_watching():
    """Stop watching plugins directory"""
    return plugin_loader.stop_watching()

def get_status():
    """Get plugin loader status"""
    return plugin_loader.get_status()

if __name__ == "__main__":
    # Test plugin loading with hot-reload
    print("🔌 Testing Plugin Loader with Hot-Reload...")
    
    def on_reload():
        print("🔄 Plugins reloaded!")
        plugins = get_plugins()
        print(f"Active plugins: {[p['name'] for p in plugins]}")
    
    # Load plugins
    plugins = load_plugins()
    print(f"Loaded {len(plugins)} plugins:")
    
    for plugin in plugins:
        print(f"  - {plugin['name']} v{plugin['version']}")
        print(f"    Description: {plugin['description']}")
        print(f"    Hooks: {list(plugin['hooks'].keys())}")
        print()
    
    # Start watching
    start_watching(on_reload)
    print("🔄 Hot-reload enabled! Edit plugins in the plugins/ directory...")
    
    try:
        # Keep running to test hot-reload
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping plugin loader...")
        stop_watching()