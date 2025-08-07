from sdk.base_plugin import BasePlugin, PluginEvent, PluginEventType
import asyncio
import logging

log = logging.getLogger("ExamplePlugin")

class ExamplePlugin(BasePlugin):
    """Enhanced example plugin demonstrating advanced features"""
    
    def __init__(self, api=None):
        super().__init__(api)
        self.name = "Example Plugin"
        self.version = "2.0.0"
        self.description = "Enhanced example plugin with async capabilities"
        self.author = "ChatGPT+ Team"
        self.priority = 3  # High priority
        
        # Register event handlers
        self.register_event_handler("custom_event", self.handle_custom_event)
        self.register_event_handler("async_event", self.handle_async_event)
        
        # Plugin-specific configuration
        self.config = {
            "greeting": "Hello from the enhanced plugin!",
            "max_responses": 10,
            "response_count": 0
        }

    def can_handle(self, command: str) -> bool:
        """Check if this plugin can handle the command"""
        keywords = ["hello", "hi", "greet", "example", "test"]
        return any(keyword in command.lower() for keyword in keywords)

    def handle(self, command: str) -> str:
        """Handle commands synchronously"""
        command_lower = command.lower()
        
        if "hello" in command_lower or "hi" in command_lower:
            return self.config["greeting"]
        elif "test" in command_lower:
            return "Plugin test successful!"
        elif "stats" in command_lower:
            return f"Plugin stats: {self.get_stats()}"
        else:
            return "I can handle hello, hi, test, and stats commands."

    async def handle_async(self, command: str) -> str:
        """Handle commands asynchronously with enhanced features"""
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        response = self.handle(command)
        self.config["response_count"] += 1
        
        # Emit an event when we handle a command
        event = PluginEvent(
            event_type=PluginEventType.PLUGIN_EVENT,
            data={"command": command, "response": response, "count": self.config["response_count"]},
            timestamp=asyncio.get_event_loop().time(),
            source=self.name
        )
        await self.emit_event(event)
        
        return response

    async def on_load(self):
        """Enhanced load method with initialization"""
        await super().on_load()
        log.info(f"Enhanced plugin {self.name} loaded with async capabilities")
        
        # Emit a system event
        event = PluginEvent(
            event_type=PluginEventType.SYSTEM_EVENT,
            data={"action": "plugin_loaded", "plugin": self.name},
            timestamp=asyncio.get_event_loop().time(),
            source=self.name
        )
        await self.emit_event(event)

    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands with enhanced logging"""
        if self.can_handle(text):
            try:
                response = await self.handle_async(text)
                log.info(f"Example plugin handled voice command: '{text}' -> '{response}'")
                return True
            except Exception as e:
                log.error(f"Example plugin failed handling voice command: {e}")
                return False
        return False

    async def on_ai_response(self, response: str):
        """Handle AI responses with analysis"""
        await super().on_ai_response(response)
        
        # Analyze AI responses for certain patterns
        if "thank" in response.lower():
            log.info("AI expressed gratitude - plugin noted")
        
        # Emit event for AI response analysis
        event = PluginEvent(
            event_type=PluginEventType.AI_RESPONSE,
            data={"response": response, "analysis": "processed"},
            timestamp=asyncio.get_event_loop().time(),
            source=self.name
        )
        await self.emit_event(event)

    def handle_custom_event(self, event_data: dict):
        """Handle custom events synchronously"""
        log.info(f"Example plugin received custom event: {event_data}")

    async def handle_async_event(self, event_data: dict):
        """Handle custom events asynchronously"""
        log.info(f"Example plugin received async event: {event_data}")
        # Simulate async processing
        await asyncio.sleep(0.05)

    async def health_check(self) -> bool:
        """Enhanced health check"""
        try:
            # Check if we can still handle commands
            test_command = "hello"
            if not self.can_handle(test_command):
                log.warning("Example plugin health check failed: cannot handle test command")
                return False
            
            # Check response count limit
            if self.config["response_count"] > self.config["max_responses"]:
                log.warning("Example plugin health check failed: exceeded max responses")
                return False
            
            return await super().health_check()
        except Exception as e:
            log.error(f"Example plugin health check failed: {e}")
            return False

    def get_config(self) -> dict:
        """Get enhanced configuration"""
        return {
            **super().get_config(),
            "custom_setting": "example_value",
            "response_count": self.config["response_count"]
        }

    def set_config(self, config: dict):
        """Set enhanced configuration"""
        super().set_config(config)
        # Update internal config
        if "greeting" in config:
            self.config["greeting"] = config["greeting"]
        if "max_responses" in config:
            self.config["max_responses"] = config["max_responses"]

def register():
    """Plugin registration function"""
    return ExamplePlugin()