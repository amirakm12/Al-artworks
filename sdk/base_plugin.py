import asyncio
import logging
import inspect
from typing import Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("BasePlugin")

class PluginEventType(Enum):
    VOICE_COMMAND = "voice_command"
    AI_RESPONSE = "ai_response"
    SYSTEM_EVENT = "system_event"
    PLUGIN_EVENT = "plugin_event"
    TASK_COMPLETED = "task_completed"
    ERROR = "error"

@dataclass
class PluginEvent:
    event_type: PluginEventType
    data: Dict[str, Any]
    timestamp: float
    source: str
    context: Optional[Dict[str, Any]] = None

class BasePlugin:
    """Enhanced base class for all plugins with advanced async capabilities"""
    
    def __init__(self, api=None):
        self.api = api
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "Base plugin implementation"
        self.author = "Unknown"
        self.enabled = True
        self.priority = 5  # 1-10, lower is higher priority
        self.event_handlers: Dict[str, Callable] = {}
        self.async_handlers: Dict[str, Callable] = {}
        self.config = {}
        self.stats = {
            "commands_handled": 0,
            "events_processed": 0,
            "errors": 0,
            "start_time": None
        }
        
        log.info(f"Initializing plugin: {self.name}")

    def can_handle(self, command: str) -> bool:
        """Check if this plugin can handle the given command"""
        return False

    def handle(self, command: str) -> str:
        """Handle a command synchronously"""
        return "Not implemented"

    async def handle_async(self, command: str) -> str:
        """Handle a command asynchronously"""
        return self.handle(command)

    def register_event_handler(self, event_name: str, handler: Callable):
        """Register an event handler"""
        if asyncio.iscoroutinefunction(handler):
            self.async_handlers[event_name] = handler
        else:
            self.event_handlers[event_name] = handler
        log.debug(f"Registered {event_name} handler for {self.name}")

    async def emit_event(self, event: PluginEvent):
        """Emit an event to the plugin system"""
        if hasattr(self, 'api') and self.api:
            await self.api.emit_plugin_event(event)
        log.debug(f"Plugin {self.name} emitted event: {event.event_type.value}")

    async def on_load(self):
        """Called when plugin is loaded - override for initialization"""
        self.stats["start_time"] = asyncio.get_event_loop().time()
        log.info(f"Plugin {self.name} loaded successfully")

    async def on_unload(self):
        """Called when plugin is unloaded - override for cleanup"""
        log.info(f"Plugin {self.name} unloaded")

    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands"""
        if self.can_handle(text):
            try:
                result = await self.handle_async(text)
                self.stats["commands_handled"] += 1
                log.info(f"Plugin {self.name} handled voice command: {text[:50]}...")
                return True
            except Exception as e:
                self.stats["errors"] += 1
                log.error(f"Plugin {self.name} failed handling voice command: {e}")
        return False

    async def on_ai_response(self, response: str):
        """Handle AI responses"""
        self.stats["events_processed"] += 1
        pass

    async def on_system_event(self, event_type: str, data: Dict[str, Any]):
        """Handle system events"""
        self.stats["events_processed"] += 1
        pass

    async def on_plugin_event(self, event_name: str, *args, **kwargs):
        """Generic event handler"""
        self.stats["events_processed"] += 1
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration"""
        return self.config

    def set_config(self, config: Dict[str, Any]):
        """Set plugin configuration"""
        self.config.update(config)
        log.info(f"Plugin {self.name} configuration updated")

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        runtime = 0
        if self.stats["start_time"]:
            runtime = asyncio.get_event_loop().time() - self.stats["start_time"]
        
        return {
            **self.stats,
            "runtime_seconds": runtime,
            "enabled": self.enabled,
            "priority": self.priority
        }

    def enable(self):
        """Enable the plugin"""
        self.enabled = True
        log.info(f"Plugin {self.name} enabled")

    def disable(self):
        """Disable the plugin"""
        self.enabled = False
        log.info(f"Plugin {self.name} disabled")

    def is_enabled(self) -> bool:
        """Check if plugin is enabled"""
        return self.enabled

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "enabled": self.enabled,
            "priority": self.priority,
            "stats": self.get_stats()
        }

    async def health_check(self) -> bool:
        """Perform a health check on the plugin"""
        try:
            # Basic health check - can be overridden
            return self.enabled and self.api is not None
        except Exception as e:
            log.error(f"Plugin {self.name} health check failed: {e}")
            return False

    def __str__(self):
        return f"{self.name} v{self.version} ({'enabled' if self.enabled else 'disabled'})"

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}' enabled={self.enabled}>"