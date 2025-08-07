"""
Plugin Loader System - Dynamic Plugin Management with Sandbox Isolation
Supports secure plugin loading, sandbox execution, and manifest-based discovery
"""

import os
import sys
import json
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import tempfile
import shutil

# Sandbox and security imports
try:
    from restrictedpython import compile_restricted, safe_builtins
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logging.warning("RestrictedPython not available - using basic sandbox")

try:
    import pluggy
    PLUGGY_AVAILABLE = True
except ImportError:
    PLUGGY_AVAILABLE = False
    logging.warning("Pluggy not available - using basic plugin system")

class PluginSandbox:
    """Secure sandbox for plugin execution"""
    
    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        self.sandbox_dir = Path(f"workspace/sandbox/{plugin_name}")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Restricted builtins for security
        self.safe_builtins = {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'bool': bool,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'dir': dir,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'chr': chr,
            'ord': ord,
            'hex': hex,
            'oct': oct,
            'bin': bin,
        }
    
    def execute_safely(self, code: str, globals_dict: Dict[str, Any] = None) -> Any:
        """Execute code in a restricted environment"""
        try:
            if RESTRICTED_PYTHON_AVAILABLE:
                # Use RestrictedPython for maximum security
                compiled = compile_restricted(code, '<plugin>', 'exec')
                restricted_globals = {
                    '__builtins__': safe_builtins,
                    '__name__': '__main__',
                    '__file__': str(self.sandbox_dir / 'plugin.py')
                }
                if globals_dict:
                    restricted_globals.update(globals_dict)
                
                exec(compiled, restricted_globals)
                return restricted_globals
            else:
                # Basic sandbox using exec with restricted globals
                safe_globals = {
                    '__builtins__': self.safe_builtins,
                    '__name__': '__main__',
                    '__file__': str(self.sandbox_dir / 'plugin.py')
                }
                if globals_dict:
                    safe_globals.update(globals_dict)
                
                exec(code, safe_globals)
                return safe_globals
                
        except Exception as e:
            logging.error(f"Sandbox execution error in {self.plugin_name}: {e}")
            raise
    
    def cleanup(self):
        """Clean up sandbox directory"""
        try:
            if self.sandbox_dir.exists():
                shutil.rmtree(self.sandbox_dir)
        except Exception as e:
            logging.error(f"Error cleaning up sandbox for {self.plugin_name}: {e}")

class PluginManager:
    """Main plugin manager with discovery and loading capabilities"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        self.loaded_plugins = {}
        self.plugin_hooks = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize plugin registry
        if PLUGGY_AVAILABLE:
            self.manager = pluggy.PluginManager("chatgpt_plus")
        else:
            self.manager = None
    
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Discover available plugins"""
        plugins = []
        
        if not self.plugins_dir.exists():
            return plugins
        
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                manifest_path = plugin_dir / "manifest.json"
                plugin_path = plugin_dir / "plugin.py"
                
                if manifest_path.exists() and plugin_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        
                        manifest['path'] = str(plugin_path)
                        manifest['directory'] = str(plugin_dir)
                        plugins.append(manifest)
                        
                    except Exception as e:
                        self.logger.error(f"Error reading manifest for {plugin_dir}: {e}")
        
        return plugins
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin"""
        try:
            plugin_dir = self.plugins_dir / plugin_name
            manifest_path = plugin_dir / "manifest.json"
            plugin_path = plugin_dir / "plugin.py"
            
            if not manifest_path.exists() or not plugin_path.exists():
                self.logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Create sandbox
            sandbox = PluginSandbox(plugin_name)
            
            # Load plugin code
            with open(plugin_path, 'r') as f:
                plugin_code = f.read()
            
            # Execute in sandbox
            plugin_globals = sandbox.execute_safely(plugin_code)
            
            # Check for required functions
            if 'register' not in plugin_globals:
                self.logger.error(f"Plugin {plugin_name} missing register function")
                return False
            
            # Register plugin
            plugin_info = {
                'manifest': manifest,
                'globals': plugin_globals,
                'sandbox': sandbox,
                'path': str(plugin_path)
            }
            
            self.loaded_plugins[plugin_name] = plugin_info
            
            # Call register function
            plugin_globals['register'](self)
            
            self.logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self) -> List[str]:
        """Load all available plugins"""
        loaded = []
        plugins = self.discover_plugins()
        
        for plugin in plugins:
            plugin_name = plugin['name']
            if self.load_plugin(plugin_name):
                loaded.append(plugin_name)
        
        return loaded
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        try:
            if plugin_name in self.loaded_plugins:
                plugin_info = self.loaded_plugins[plugin_name]
                
                # Call unregister if available
                if 'unregister' in plugin_info['globals']:
                    plugin_info['globals']['unregister'](self)
                
                # Cleanup sandbox
                plugin_info['sandbox'].cleanup()
                
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_name]
                
                self.logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get a loaded plugin"""
        return self.loaded_plugins.get(plugin_name)
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names"""
        return list(self.loaded_plugins.keys())
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback"""
        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []
        self.plugin_hooks[hook_name].append(callback)
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all registered hooks"""
        results = []
        if hook_name in self.plugin_hooks:
            for callback in self.plugin_hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in hook {hook_name}: {e}")
        return results
    
    def create_plugin_template(self, plugin_name: str, description: str = "") -> bool:
        """Create a new plugin template"""
        try:
            plugin_dir = self.plugins_dir / plugin_name
            plugin_dir.mkdir(exist_ok=True)
            
            # Create manifest
            manifest = {
                "name": plugin_name,
                "version": "1.0.0",
                "description": description,
                "author": "Plugin Developer",
                "dependencies": [],
                "hooks": [],
                "permissions": []
            }
            
            with open(plugin_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create plugin template
            plugin_template = f'''"""
{plugin_name} - Plugin Template
{description}
"""

def register(plugin_manager):
    """Register this plugin with the plugin manager"""
    print(f"Registering plugin: {plugin_name}")
    
    # Register hooks here
    # plugin_manager.register_hook("message_received", handle_message)
    # plugin_manager.register_hook("tool_executed", handle_tool)

def unregister(plugin_manager):
    """Unregister this plugin"""
    print(f"Unregistering plugin: {plugin_name}")

# Plugin-specific functions
def handle_message(message):
    """Handle incoming messages"""
    return f"Plugin {plugin_name} processed: {{message}}"

def handle_tool(tool_name, result):
    """Handle tool execution results"""
    return f"Plugin {plugin_name} handled tool: {{tool_name}}"
'''
            
            with open(plugin_dir / "plugin.py", 'w') as f:
                f.write(plugin_template)
            
            self.logger.info(f"Created plugin template: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating plugin template {plugin_name}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all plugins and sandboxes"""
        for plugin_name in list(self.loaded_plugins.keys()):
            self.unload_plugin(plugin_name)

# Global plugin manager instance
plugin_manager = PluginManager()

def load_plugins():
    """Load all available plugins"""
    return plugin_manager.load_all_plugins()

def get_plugin(name: str):
    """Get a specific plugin"""
    return plugin_manager.get_plugin(name)

def create_plugin(name: str, description: str = ""):
    """Create a new plugin template"""
    return plugin_manager.create_plugin_template(name, description)

if __name__ == "__main__":
    # Test plugin loading
    print("Discovering plugins...")
    plugins = plugin_manager.discover_plugins()
    print(f"Found {len(plugins)} plugins: {[p['name'] for p in plugins]}")
    
    print("Loading plugins...")
    loaded = plugin_manager.load_all_plugins()
    print(f"Loaded {len(loaded)} plugins: {loaded}")