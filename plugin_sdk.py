import os
import sys
import asyncio
import importlib.util
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass

log = logging.getLogger("PluginSDK")

PLUGIN_DIR = "plugins"

@dataclass
class PluginContext:
    """Context object passed to plugins with core system access"""
    ai_manager: Any = None
    gpu_device: Any = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None
    config: Dict[str, Any] = None
    state: Dict[str, Any] = None

class Plugin:
    def __init__(self, name: str, module, context: PluginContext = None):
        self.name = name
        self.module = module
        self.context = context or PluginContext()
        self.loaded = False
        self.error_count = 0
        self.last_error = None

    async def on_load(self):
        """Load the plugin and call its on_load method"""
        try:
            if hasattr(self.module, "on_load"):
                await maybe_async(self.module.on_load, self.context)
            self.loaded = True
            self.error_count = 0
            self.last_error = None
            log.info(f"Plugin {self.name} loaded successfully")
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            log.error(f"Plugin {self.name} failed to load: {e}")
            raise

    async def on_unload(self):
        """Unload the plugin and call its on_unload method"""
        try:
            if hasattr(self.module, "on_unload"):
                await maybe_async(self.module.on_unload, self.context)
            self.loaded = False
            log.info(f"Plugin {self.name} unloaded successfully")
        except Exception as e:
            log.error(f"Plugin {self.name} failed to unload: {e}")

    async def handle_event(self, event_name: str, *args, **kwargs):
        """Handle an event by calling the plugin's event handler"""
        if not self.loaded:
            return
        
        try:
            handler_name = f"on_{event_name}"
            handler = getattr(self.module, handler_name, None)
            if callable(handler):
                await maybe_async(handler, self.context, *args, **kwargs)
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            log.error(f"Plugin {self.name} failed handling event {event_name}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            "name": self.name,
            "loaded": self.loaded,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "has_on_load": hasattr(self.module, "on_load"),
            "has_on_unload": hasattr(self.module, "on_unload"),
            "event_handlers": [attr for attr in dir(self.module) if attr.startswith("on_")]
        }

async def maybe_async(func: Callable, *args, **kwargs):
    """Execute a function, handling both sync and async functions"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)

class PluginManager:
    def __init__(self, plugin_dir=PLUGIN_DIR, context: PluginContext = None):
        self.plugin_dir = plugin_dir
        self.context = context or PluginContext()
        self.plugins: Dict[str, Plugin] = {}
        self.event_listeners: Dict[str, List[Plugin]] = {}
        self.running = False
        
        # Ensure plugin directory exists
        os.makedirs(self.plugin_dir, exist_ok=True)

    def discover_plugins(self) -> List[str]:
        """Discover all available plugins in the plugin directory"""
        plugins = []
        if not os.path.exists(self.plugin_dir):
            log.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return plugins
        
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                plugin_name = filename[:-3]
                plugins.append(plugin_name)
        
        log.info(f"Discovered {len(plugins)} plugins: {plugins}")
        return plugins

    def load_plugin_module(self, name: str):
        """Load a plugin module from file"""
        path = os.path.join(self.plugin_dir, f"{name}.py")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Plugin file not found: {path}")
        
        # Create a unique module name to avoid conflicts
        module_name = f"plugin_{name}"
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return module

    async def load_all_plugins(self):
        """Load all discovered plugins"""
        log.info("Loading all plugins...")
        
        for name in self.discover_plugins():
            try:
                mod = self.load_plugin_module(name)
                plugin = Plugin(name, mod, self.context)
                await plugin.on_load()
                self.plugins[name] = plugin
                log.info(f"✓ Plugin loaded: {name}")
            except Exception as e:
                log.error(f"✗ Failed loading plugin {name}: {e}")

    async def unload_all_plugins(self):
        """Unload all loaded plugins"""
        log.info("Unloading all plugins...")
        
        for plugin in list(self.plugins.values()):
            try:
                await plugin.on_unload()
                log.info(f"✓ Plugin unloaded: {plugin.name}")
            except Exception as e:
                log.error(f"✗ Failed unloading plugin {plugin.name}: {e}")
        
        self.plugins.clear()

    async def emit_event(self, event_name: str, *args, **kwargs):
        """Emit an event to all loaded plugins"""
        if not self.running:
            return
        
        log.debug(f"Emitting event '{event_name}' to {len(self.plugins)} plugins")
        
        for plugin in self.plugins.values():
            try:
                await plugin.handle_event(event_name, *args, **kwargs)
            except Exception as e:
                log.error(f"Plugin {plugin.name} failed handling event {event_name}: {e}")

    async def reload_plugin(self, name: str):
        """Reload a specific plugin"""
        if name in self.plugins:
            await self.plugins[name].on_unload()
            del self.plugins[name]
        
        try:
            mod = self.load_plugin_module(name)
            plugin = Plugin(name, mod, self.context)
            await plugin.on_load()
            self.plugins[name] = plugin
            log.info(f"✓ Plugin reloaded: {name}")
            return True
        except Exception as e:
            log.error(f"✗ Failed reloading plugin {name}: {e}")
            return False

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about all plugins"""
        info = {
            "total_plugins": len(self.plugins),
            "loaded_plugins": len([p for p in self.plugins.values() if p.loaded]),
            "plugins": {}
        }
        
        for name, plugin in self.plugins.items():
            info["plugins"][name] = plugin.get_status()
        
        return info

    def set_context(self, context: PluginContext):
        """Update the plugin context"""
        self.context = context
        for plugin in self.plugins.values():
            plugin.context = context

    async def start(self):
        """Start the plugin manager"""
        self.running = True
        await self.load_all_plugins()
        log.info("Plugin manager started")

    async def stop(self):
        """Stop the plugin manager"""
        self.running = False
        await self.unload_all_plugins()
        log.info("Plugin manager stopped")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a specific plugin by name"""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all loaded plugin names"""
        return list(self.plugins.keys())

    def is_plugin_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded"""
        return name in self.plugins and self.plugins[name].loaded

# Example plugin template
EXAMPLE_PLUGIN_CODE = '''
# Example Plugin
# This plugin demonstrates the plugin SDK features

async def on_load(context):
    """Called when the plugin is loaded"""
    print(f"[ExamplePlugin] Loaded with context: {context}")
    
    # Access AI manager if available
    if context.ai_manager:
        print(f"[ExamplePlugin] AI Manager available: {context.ai_manager}")
    
    # Access GPU device if available
    if context.gpu_device:
        print(f"[ExamplePlugin] GPU Device: {context.gpu_device}")

async def on_unload(context):
    """Called when the plugin is unloaded"""
    print("[ExamplePlugin] Unloading...")

async def on_voice_recognized(context, text):
    """Called when voice is recognized"""
    print(f"[ExamplePlugin] Voice recognized: {text}")
    
    # Example: Generate AI response
    if context.ai_manager:
        response = await context.ai_manager.generate_text(f"Reply to: {text}")
        print(f"[ExamplePlugin] AI Response: {response}")

async def on_ai_response(context, prompt, response):
    """Called when AI generates a response"""
    print(f"[ExamplePlugin] AI Response: {response}")

async def on_tts_synthesize(context, text):
    """Called when TTS synthesis is requested"""
    print(f"[ExamplePlugin] TTS Request: {text}")

async def on_gpu_stats(context, stats):
    """Called when GPU stats are updated"""
    print(f"[ExamplePlugin] GPU Stats: {stats}")
'''

# Create example plugin if it doesn't exist
def create_example_plugin():
    """Create an example plugin for testing"""
    os.makedirs(PLUGIN_DIR, exist_ok=True)
    example_path = os.path.join(PLUGIN_DIR, "example_plugin.py")
    
    if not os.path.exists(example_path):
        with open(example_path, 'w') as f:
            f.write(EXAMPLE_PLUGIN_CODE)
        log.info(f"Created example plugin: {example_path}")

# Utility functions
def create_plugin_context(ai_manager=None, gpu_device=None, config=None):
    """Create a plugin context with the given components"""
    return PluginContext(
        ai_manager=ai_manager,
        gpu_device=gpu_device,
        event_loop=asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None,
        config=config or {},
        state={}
    )

async def test_plugin_system():
    """Test the plugin system"""
    logging.basicConfig(level=logging.INFO)
    
    # Create example plugin
    create_example_plugin()
    
    # Create plugin manager
    context = create_plugin_context()
    manager = PluginManager(context=context)
    
    try:
        # Start plugin manager
        await manager.start()
        
        # Emit some test events
        await manager.emit_event("voice_recognized", "Hello, this is a test")
        await manager.emit_event("ai_response", "Test prompt", "Test response")
        await manager.emit_event("gpu_stats", {"memory": "8GB", "usage": "50%"})
        
        # Get plugin info
        info = manager.get_plugin_info()
        print(f"Plugin info: {info}")
        
        # Stop plugin manager
        await manager.stop()
        
    except Exception as e:
        log.error(f"Plugin system test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_plugin_system())