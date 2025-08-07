"""
Plugin Loader - Simplified Plugin Management System
Loads plugins from plugins/ directory and manages hooks
"""

import os
import importlib.util
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

class PluginLoader:
    """Simplified plugin loader for ChatGPT+ Clone"""
    
    def __init__(self):
        self.plugins = []
        self.hooks = {}
        self.logger = logging.getLogger(__name__)
        self.plugins_dir = Path("plugins")
        self.plugins_dir.mkdir(exist_ok=True)
    
    def load_plugins(self) -> List[Dict[str, Any]]:
        """Load all plugins from plugins/ directory"""
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
        
        return self.plugins
    
    def load_plugin(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single plugin"""
        try:
            # Load plugin module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get plugin metadata
            if hasattr(module, 'on_load'):
                metadata = module.on_load()
                metadata['module'] = module
                metadata['path'] = str(plugin_path)
                return metadata
            else:
                self.logger.warning(f"Plugin {plugin_path.name} missing on_load function")
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
    
    def get_plugins(self) -> List[Dict[str, Any]]:
        """Get all loaded plugins"""
        return self.plugins.copy()
    
    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific plugin by name"""
        for plugin in self.plugins:
            if plugin['name'] == name:
                return plugin
        return None
    
    def reload_plugins(self) -> List[Dict[str, Any]]:
        """Reload all plugins"""
        self.logger.info("🔄 Reloading plugins...")
        return self.load_plugins()
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a specific plugin"""
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

def reload_plugins():
    """Reload all plugins"""
    return plugin_loader.reload_plugins()

if __name__ == "__main__":
    # Test plugin loading
    print("🔌 Testing Plugin Loader...")
    
    plugins = load_plugins()
    print(f"Loaded {len(plugins)} plugins:")
    
    for plugin in plugins:
        print(f"  - {plugin['name']} v{plugin['version']}")
        print(f"    Description: {plugin['description']}")
        print(f"    Hooks: {list(plugin['hooks'].keys())}")
        print()
    
    # Test hook calling
    print("Testing hooks...")
    results = call_hook("on_message_received", "Hello plugins!", {})
    for result in results:
        print(f"  {result['plugin']}: {result['result']}")