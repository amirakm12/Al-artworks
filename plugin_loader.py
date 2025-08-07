import importlib.util
import asyncio
import os
import logging
from typing import List, Any

log = logging.getLogger("PluginLoader")

class PluginManager:
    """Async Plugin Manager with full AI integration"""
    
    def __init__(self, api):
        self.api = api
        self.plugins = []
        self.plugin_dir = "plugins"
        
        # Ensure plugin directory exists
        os.makedirs(self.plugin_dir, exist_ok=True)

    async def load_plugins(self, plugin_paths: List[str]):
        """Load plugins from specified paths"""
        log.info(f"Loading {len(plugin_paths)} plugins...")
        
        for path in plugin_paths:
            try:
                if os.path.exists(path):
                    mod = await self._import_plugin(path)
                    plugin = mod.Plugin(self.api)
                    await plugin.on_load()
                    self.plugins.append(plugin)
                    log.info(f"✓ Loaded plugin: {path}")
                else:
                    log.warning(f"Plugin file not found: {path}")
            except Exception as e:
                log.error(f"✗ Failed to load plugin {path}: {e}")

    async def load_all_plugins(self):
        """Dynamically discover and load all plugins from plugins directory"""
        if not os.path.exists(self.plugin_dir):
            log.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return
        
        plugin_files = []
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                plugin_files.append(os.path.join(self.plugin_dir, filename))
        
        log.info(f"Discovered {len(plugin_files)} plugin files")
        await self.load_plugins(plugin_files)

    async def _import_plugin(self, path: str):
        """Dynamically import a plugin module"""
        module_name = f"plugin_{os.path.basename(path)[:-3]}"
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin from {path}")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        return mod

    async def dispatch_voice_command(self, text: str) -> bool:
        """Dispatch voice command to all plugins, return True if handled"""
        log.debug(f"Dispatching voice command: {text}")
        
        for plugin in self.plugins:
            try:
                if await plugin.on_voice_command(text):
                    log.info(f"Voice command handled by plugin: {plugin.__class__.__name__}")
                    return True
            except Exception as e:
                log.error(f"Plugin {plugin.__class__.__name__} failed handling voice command: {e}")
        
        return False

    async def dispatch_ai_response(self, response: str):
        """Dispatch AI response to all plugins"""
        log.debug(f"Dispatching AI response: {response[:50]}...")
        
        for plugin in self.plugins:
            try:
                await plugin.on_ai_response(response)
            except Exception as e:
                log.error(f"Plugin {plugin.__class__.__name__} failed handling AI response: {e}")

    async def dispatch_event(self, event_name: str, *args, **kwargs):
        """Dispatch generic event to all plugins"""
        log.debug(f"Dispatching event: {event_name}")
        
        for plugin in self.plugins:
            try:
                handler_name = f"on_{event_name}"
                if hasattr(plugin, handler_name):
                    handler = getattr(plugin, handler_name)
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*args, **kwargs)
                    else:
                        handler(*args, **kwargs)
            except Exception as e:
                log.error(f"Plugin {plugin.__class__.__name__} failed handling event {event_name}: {e}")

    async def unload_all_plugins(self):
        """Unload all plugins gracefully"""
        log.info("Unloading all plugins...")
        
        for plugin in self.plugins:
            try:
                await plugin.on_unload()
                log.info(f"✓ Unloaded plugin: {plugin.__class__.__name__}")
            except Exception as e:
                log.error(f"✗ Failed to unload plugin {plugin.__class__.__name__}: {e}")
        
        self.plugins.clear()

    def get_plugin_info(self) -> List[dict]:
        """Get information about all loaded plugins"""
        info = []
        for plugin in self.plugins:
            info.append({
                "name": plugin.__class__.__name__,
                "module": plugin.__class__.__module__,
                "methods": [attr for attr in dir(plugin) if attr.startswith("on_")]
            })
        return info

    def get_plugin_count(self) -> int:
        """Get number of loaded plugins"""
        return len(self.plugins)

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a specific plugin is loaded"""
        return any(plugin.__class__.__name__ == plugin_name for plugin in self.plugins)