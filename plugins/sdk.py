import importlib
import json
import os
import logging
import asyncio
import torch
from typing import Optional, Dict, Any, Callable

log = logging.getLogger("PluginSDK")

class PluginContext:
    def __init__(self, config: Dict[str, Any], ai_manager, gpu_device: torch.device, event_loop: asyncio.AbstractEventLoop):
        self.config = config
        self.ai_manager = ai_manager  # Core AI manager for model calls, generation, etc.
        self.gpu_device = gpu_device  # torch.device('cuda') or 'cpu' or 'mps'
        self.loop = event_loop        # asyncio event loop
        self.state = {}
        self.event_handlers = {}      # Plugin event handlers

    def save_state(self, key: str, value: Any):
        """Save plugin state that persists across sessions"""
        self.state[key] = value
        log.info(f"State saved for key: {key}")

    def get_state(self, key: str, default=None):
        """Retrieve plugin state"""
        return self.state.get(key, default)

    async def run_async_task(self, coro):
        """Run async task in the main event loop"""
        return await coro

    def register_event_handler(self, event_name: str, handler: Callable):
        """Register an event handler for specific events"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
        log.info(f"Registered event handler for: {event_name}")

    async def trigger_event(self, event_name: str, *args, **kwargs):
        """Trigger an event to all registered handlers"""
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    await handler(*args, **kwargs)
                except Exception as e:
                    log.error(f"Event handler error for {event_name}: {e}")

class PluginBase:
    def __init__(self, context: PluginContext):
        self.context = context
        self.name = self.__class__.__name__
        log.info(f"Initializing plugin: {self.name}")

    def on_load(self):
        """Called when plugin is loaded - override for initialization"""
        log.info(f"Plugin {self.name} loaded")

    def on_unload(self):
        """Called when plugin is unloaded - override for cleanup"""
        log.info(f"Plugin {self.name} unloaded")

    async def on_event(self, event_name: str, *args, **kwargs):
        """Called when an event is broadcast - override for event handling"""
        log.info(f"Plugin {self.name} received event {event_name}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.context.config

    def save_config(self, key: str, value: Any):
        """Save configuration value"""
        self.context.config[key] = value

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Use AI manager to generate text (LLM).
        kwargs can include generation params like max_length, temperature, etc.
        """
        if not self.context.ai_manager:
            raise RuntimeError("AI Manager not set in PluginContext")

        log.info(f"Plugin {self.name} generating text with prompt: {prompt}")
        response = await self.context.ai_manager.generate(prompt, device=self.context.gpu_device, **kwargs)
        return response

    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Use AI manager to transcribe audio with Whisper or similar.
        """
        if not self.context.ai_manager:
            raise RuntimeError("AI Manager not set in PluginContext")

        log.info(f"Plugin {self.name} transcribing audio: {audio_path}")
        transcription = await self.context.ai_manager.transcribe(audio_path, device=self.context.gpu_device)
        return transcription

    async def synthesize_speech(self, text: str, output_path: str) -> str:
        """
        Use AI manager to synthesize speech from text.
        """
        if not self.context.ai_manager:
            raise RuntimeError("AI Manager not set in PluginContext")

        log.info(f"Plugin {self.name} synthesizing speech: {text}")
        audio_path = await self.context.ai_manager.synthesize_speech(text, output_path, device=self.context.gpu_device)
        return audio_path

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU device information"""
        return {
            "device": str(self.context.gpu_device),
            "cuda_available": torch.cuda.is_available(),
            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info(f"Plugin {self.name} cleared GPU cache")

# Plugin loader with sandbox toggle & AI manager injection
class PluginLoader:
    def __init__(self, plugin_dir: str = "plugins", sandbox: bool = True, context: Optional[PluginContext] = None):
        self.plugin_dir = plugin_dir
        self.plugins = []
        self.sandbox = sandbox
        self.context = context
        self.loaded_modules = {}

    def set_context(self, context: PluginContext):
        """Set the plugin context with AI manager and GPU device"""
        self.context = context

    def load_plugins(self):
        """Load all plugins from the plugin directory"""
        if not os.path.exists(self.plugin_dir):
            log.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                path = os.path.join(self.plugin_dir, filename)
                name = filename[:-3]
                
                try:
                    spec = importlib.util.spec_from_file_location(name, path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    
                    # Look for Plugin class in the module
                    plugin_class = getattr(mod, "Plugin", None)
                    if plugin_class and issubclass(plugin_class, PluginBase):
                        plugin_instance = plugin_class(self.context)
                        plugin_instance.on_load()
                        self.plugins.append(plugin_instance)
                        self.loaded_modules[name] = mod
                        log.info(f"Loaded plugin: {name}")
                    else:
                        log.warning(f"No Plugin class found in {filename}")
                        
                except Exception as e:
                    log.error(f"Failed to load plugin {name}: {e}")

    def unload_plugins(self):
        """Unload all plugins"""
        for plugin in self.plugins:
            try:
                plugin.on_unload()
            except Exception as e:
                log.error(f"Error unloading plugin {plugin.name}: {e}")
        
        self.plugins.clear()
        self.loaded_modules.clear()
        log.info("All plugins unloaded")

    async def broadcast_event(self, event_name: str, *args, **kwargs):
        """Broadcast an event to all loaded plugins"""
        log.info(f"Broadcasting event: {event_name}")
        for plugin in self.plugins:
            try:
                await plugin.on_event(event_name, *args, **kwargs)
            except Exception as e:
                log.error(f"Error in plugin {plugin.name} handling event {event_name}: {e}")

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about loaded plugins"""
        return {
            "total_plugins": len(self.plugins),
            "plugins": [plugin.name for plugin in self.plugins],
            "sandbox_enabled": self.sandbox,
            "plugin_directory": self.plugin_dir
        }

    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        # Find and unload the plugin
        plugin_to_remove = None
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                plugin_to_remove = plugin
                break
        
        if plugin_to_remove:
            try:
                plugin_to_remove.on_unload()
                self.plugins.remove(plugin_to_remove)
                log.info(f"Unloaded plugin: {plugin_name}")
            except Exception as e:
                log.error(f"Error unloading plugin {plugin_name}: {e}")
                return False
        
        # Reload the plugin
        try:
            filename = f"{plugin_name}.py"
            path = os.path.join(self.plugin_dir, filename)
            
            spec = importlib.util.spec_from_file_location(plugin_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            plugin_class = getattr(mod, "Plugin", None)
            if plugin_class and issubclass(plugin_class, PluginBase):
                plugin_instance = plugin_class(self.context)
                plugin_instance.on_load()
                self.plugins.append(plugin_instance)
                self.loaded_modules[plugin_name] = mod
                log.info(f"Reloaded plugin: {plugin_name}")
                return True
            else:
                log.error(f"No Plugin class found in {filename}")
                return False
                
        except Exception as e:
            log.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False