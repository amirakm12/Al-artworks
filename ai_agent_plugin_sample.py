from plugin_routing import CommandEvent, SecurityContext, Role, Permission
from plugins.sdk import PluginBase
import asyncio
import logging
import time
from typing import Dict, Any, Optional

log = logging.getLogger("AdvancedPlugin")

class SecureEchoPlugin(PluginBase):
    """Advanced plugin with RBAC support and orchestrator integration"""
    
    def __init__(self, orchestrator=None):
        super().__init__(None)  # No API needed for this example
        self.orchestrator = orchestrator
        self.name = "SecureEchoPlugin"
        self.command_count = 0
        self.last_command_time = 0

    async def on_load(self):
        """Called when plugin is loaded"""
        log.info(f"[{self.name}] Advanced plugin loaded with RBAC support")
        if self.orchestrator:
            log.info(f"[{self.name}] Orchestrator integration available")

    async def on_unload(self):
        """Called when plugin is unloaded"""
        log.info(f"[{self.name}] Plugin unloaded. Processed {self.command_count} commands")

    async def handle_command(self, event: CommandEvent) -> bool:
        """Handle commands with role-based access control"""
        self.command_count += 1
        self.last_command_time = time.time()
        
        log.info(f"[{self.name}] Processing command: {event.command_name} "
                f"from user {event.context.user_id} with roles {[r.name for r in event.context.roles]}")
        
        if event.command_name == "echo_secure":
            # Anyone can use echo_secure
            message = event.args.get("message", "")
            event.result = f"Secure Echo: {message}"
            log.info(f"[{self.name}] Secure echo processed: {message}")
            return True
            
        elif event.command_name == "echo_admin":
            # Only admins can use echo_admin
            if Role.ADMIN in event.context.roles:
                message = event.args.get("message", "")
                event.result = f"Admin Echo: {message}"
                log.info(f"[{self.name}] Admin echo processed: {message}")
                return True
            else:
                event.error = "Access denied: Admin role required for echo_admin"
                log.warning(f"[{self.name}] Access denied for echo_admin by user {event.context.user_id}")
                return True
        
        elif event.command_name == "system_status":
            # System role required
            if Role.SYSTEM in event.context.roles or Role.ADMIN in event.context.roles:
                status = {
                    "plugin_name": self.name,
                    "commands_processed": self.command_count,
                    "last_command": self.last_command_time,
                    "uptime": time.time() - self.last_command_time if self.last_command_time > 0 else 0
                }
                event.result = status
                log.info(f"[{self.name}] System status provided")
                return True
            else:
                event.error = "Access denied: System role required for system_status"
                return True
        
        elif event.command_name == "submit_ai_task":
            # Only admins can submit AI tasks
            if Role.ADMIN not in event.context.roles:
                event.error = "Access denied: Admin role required to submit AI tasks"
                return True
            
            if not self.orchestrator:
                event.error = "Orchestrator not available"
                return True
            
            try:
                task_type = event.args.get("task_type", "ai")
                task_name = event.args.get("task_name", "plugin_task")
                context = event.args.get("context", {})
                
                task_id = await self.orchestrator.submit_ai_task(task_type, task_name, context)
                event.result = {
                    "task_id": task_id,
                    "status": "submitted",
                    "task_type": task_type,
                    "task_name": task_name
                }
                log.info(f"[{self.name}] AI task submitted: {task_name} ({task_type})")
                return True
            except Exception as e:
                event.error = f"Failed to submit AI task: {str(e)}"
                log.error(f"[{self.name}] Failed to submit AI task: {e}")
                return True
        
        elif event.command_name == "plugin_stats":
            # Anyone can get plugin stats
            stats = {
                "plugin_name": self.name,
                "commands_processed": self.command_count,
                "last_command_time": self.last_command_time,
                "uptime": time.time() - self.last_command_time if self.last_command_time > 0 else 0,
                "orchestrator_available": self.orchestrator is not None
            }
            event.result = stats
            log.info(f"[{self.name}] Plugin stats provided")
            return True
        
        return False

    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands with natural language processing"""
        text_lower = text.lower()
        
        if "echo" in text_lower and "secure" in text_lower:
            # Extract message from voice command
            message = text.replace("echo secure", "").strip()
            if message:
                log.info(f"[{self.name}] Voice command processed: echo_secure '{message}'")
                return True
        
        elif "admin echo" in text_lower:
            # This would require admin role check in a real implementation
            message = text.replace("admin echo", "").strip()
            if message:
                log.info(f"[{self.name}] Voice command processed: echo_admin '{message}' (would require admin role)")
                return True
        
        elif "plugin status" in text_lower or "plugin stats" in text_lower:
            log.info(f"[{self.name}] Voice command processed: plugin_stats")
            return True
        
        return False

    async def on_ai_response(self, response: str):
        """Handle AI responses"""
        log.info(f"[{self.name}] Received AI response: {response[:100]}...")

    async def on_system_event(self, event_type: str, data: Dict[str, Any]):
        """Handle system events"""
        log.info(f"[{self.name}] System event: {event_type} - {data}")

class AdvancedSystemPlugin(PluginBase):
    """Plugin with advanced system control capabilities"""
    
    def __init__(self, orchestrator=None):
        super().__init__(None)
        self.orchestrator = orchestrator
        self.name = "AdvancedSystemPlugin"
        self.system_commands = {
            "get_system_info": self._get_system_info,
            "get_gpu_stats": self._get_gpu_stats,
            "get_orchestrator_stats": self._get_orchestrator_stats,
            "clear_gpu_memory": self._clear_gpu_memory,
            "restart_services": self._restart_services
        }

    async def handle_command(self, event: CommandEvent) -> bool:
        """Handle system-level commands with strict permission checks"""
        
        # Check for system-level permissions
        if not (Role.SYSTEM in event.context.roles or Role.ADMIN in event.context.roles):
            event.error = "Access denied: System or Admin role required"
            return True
        
        command_name = event.command_name
        if command_name in self.system_commands:
            try:
                result = await self.system_commands[command_name](event.args)
                event.result = result
                log.info(f"[{self.name}] System command executed: {command_name}")
                return True
            except Exception as e:
                event.error = f"System command failed: {str(e)}"
                log.error(f"[{self.name}] System command failed: {command_name} - {e}")
                return True
        
        return False

    async def _get_system_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive system information"""
        import psutil
        import platform
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }

    async def _get_gpu_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get GPU statistics"""
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            gpu_stats = self.orchestrator.gpu_tuner.get_average_stats()
            device_info = self.orchestrator.gpu_tuner.get_device_info()
            return {
                "gpu_stats": gpu_stats,
                "device_info": device_info
            }
        except Exception as e:
            return {"error": f"Failed to get GPU stats: {str(e)}"}

    async def _get_orchestrator_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            return self.orchestrator.get_system_stats()
        except Exception as e:
            return {"error": f"Failed to get orchestrator stats: {str(e)}"}

    async def _clear_gpu_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clear GPU memory cache"""
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            await self.orchestrator.gpu_tuner.clear_gpu_memory()
            return {"status": "GPU memory cleared successfully"}
        except Exception as e:
            return {"error": f"Failed to clear GPU memory: {str(e)}"}

    async def _restart_services(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Restart system services (simulated)"""
        service_name = args.get("service", "unknown")
        
        # Simulate service restart
        await asyncio.sleep(1.0)
        
        return {
            "status": "success",
            "service": service_name,
            "message": f"Service {service_name} restarted successfully"
        }

class VoiceControlPlugin(PluginBase):
    """Plugin for advanced voice control capabilities"""
    
    def __init__(self, orchestrator=None):
        super().__init__(None)
        self.orchestrator = orchestrator
        self.name = "VoiceControlPlugin"
        self.voice_commands = {
            "start listening": "voice_start",
            "stop listening": "voice_stop", 
            "generate text": "ai_generate",
            "create image": "image_generate",
            "system status": "system_status",
            "gpu status": "gpu_status"
        }

    async def on_voice_command(self, text: str) -> bool:
        """Process voice commands with natural language understanding"""
        text_lower = text.lower()
        
        # Check for voice control commands
        for voice_cmd, action in self.voice_commands.items():
            if voice_cmd in text_lower:
                log.info(f"[{self.name}] Voice command detected: {voice_cmd}")
                
                if action == "ai_generate":
                    # Extract prompt from voice command
                    prompt = text.replace("generate text", "").strip()
                    if prompt:
                        await self._handle_ai_generation(prompt)
                        return True
                
                elif action == "image_generate":
                    # Extract prompt from voice command
                    prompt = text.replace("create image", "").strip()
                    if prompt:
                        await self._handle_image_generation(prompt)
                        return True
                
                elif action == "system_status":
                    await self._handle_system_status()
                    return True
                
                elif action == "gpu_status":
                    await self._handle_gpu_status()
                    return True
                
                return True
        
        return False

    async def _handle_ai_generation(self, prompt: str):
        """Handle AI text generation from voice command"""
        if self.orchestrator:
            try:
                task_id = await self.orchestrator.submit_ai_task(
                    "ai", "voice_text_generation", 
                    {"prompt": prompt, "model_type": "gpt"}
                )
                log.info(f"[{self.name}] AI generation task submitted: {task_id}")
            except Exception as e:
                log.error(f"[{self.name}] Failed to submit AI generation: {e}")

    async def _handle_image_generation(self, prompt: str):
        """Handle image generation from voice command"""
        if self.orchestrator:
            try:
                task_id = await self.orchestrator.submit_ai_task(
                    "gpu", "voice_image_generation",
                    {"prompt": prompt, "model_type": "stable_diffusion"}
                )
                log.info(f"[{self.name}] Image generation task submitted: {task_id}")
            except Exception as e:
                log.error(f"[{self.name}] Failed to submit image generation: {e}")

    async def _handle_system_status(self):
        """Handle system status voice command"""
        log.info(f"[{self.name}] System status requested via voice")

    async def _handle_gpu_status(self):
        """Handle GPU status voice command"""
        log.info(f"[{self.name}] GPU status requested via voice")

# Example usage
async def example_advanced_plugins():
    """Example of using advanced plugins with RBAC"""
    logging.basicConfig(level=logging.INFO)
    
    # Create plugins
    echo_plugin = SecureEchoPlugin()
    system_plugin = AdvancedSystemPlugin()
    voice_plugin = VoiceControlPlugin()
    
    # Load plugins
    await echo_plugin.on_load()
    await system_plugin.on_load()
    await voice_plugin.on_load()
    
    # Test voice commands
    voice_commands = [
        "echo secure hello world",
        "admin echo secret message",
        "plugin status",
        "generate text about artificial intelligence",
        "create image of a beautiful sunset",
        "system status"
    ]
    
    for voice_cmd in voice_commands:
        log.info(f"Testing voice command: {voice_cmd}")
        
        # Test with echo plugin
        handled = await echo_plugin.on_voice_command(voice_cmd)
        if handled:
            log.info(f"  Echo plugin handled: {voice_cmd}")
        
        # Test with voice plugin
        handled = await voice_plugin.on_voice_command(voice_cmd)
        if handled:
            log.info(f"  Voice plugin handled: {voice_cmd}")
    
    # Unload plugins
    await echo_plugin.on_unload()
    await system_plugin.on_unload()
    await voice_plugin.on_unload()
    
    log.info("Advanced plugin example completed")

if __name__ == "__main__":
    asyncio.run(example_advanced_plugins())