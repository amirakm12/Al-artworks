import asyncio
import logging
import subprocess
import os
import sys
from typing import Optional, Dict, Any

log = logging.getLogger("PluginSDK")

class PluginBase:
    """Base class for all plugins with async lifecycle hooks"""
    
    def __init__(self, api):
        self.api = api
        self.name = self.__class__.__name__
        log.info(f"Initializing plugin: {self.name}")

    async def on_load(self):
        """Called when plugin is loaded - override for initialization"""
        log.info(f"Plugin {self.name} loaded")

    async def on_unload(self):
        """Called when plugin is unloaded - override for cleanup"""
        log.info(f"Plugin {self.name} unloaded")

    async def on_voice_command(self, text: str) -> bool:
        """
        Called when voice command is recognized
        Return True if command was handled, False otherwise
        """
        return False

    async def on_ai_response(self, response: str):
        """Called when AI generates a response"""
        pass

    async def on_system_event(self, event_type: str, data: Dict[str, Any]):
        """Called for system events (startup, shutdown, etc.)"""
        pass

    async def on_plugin_event(self, event_name: str, *args, **kwargs):
        """Generic event handler - override for custom events"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration - override for custom config"""
        return {}

    def set_config(self, config: Dict[str, Any]):
        """Set plugin configuration - override for custom config"""
        pass

class AIManagerAPI:
    """API for plugins to interact with AI models and system control"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.command_history = []

    async def generate_response(self, prompt: str) -> str:
        """Generate AI response using the loaded model"""
        try:
            return await asyncio.to_thread(self.model_manager.generate, prompt)
        except Exception as e:
            log.error(f"Failed to generate AI response: {e}")
            return f"Error generating response: {str(e)}"

    async def execute_system_command(self, cmd: str) -> str:
        """Execute system command and return output"""
        log.info(f"Executing system command: {cmd}")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            output = stdout.decode('utf-8', errors='ignore')
            error = stderr.decode('utf-8', errors='ignore')
            
            # Log command history
            self.command_history.append({
                "command": cmd,
                "return_code": proc.returncode,
                "output": output,
                "error": error
            })
            
            if error:
                log.error(f"System command error: {error}")
            
            return output
            
        except Exception as e:
            log.error(f"Failed to execute system command: {e}")
            return f"Error executing command: {str(e)}"

    async def open_file(self, filepath: str) -> bool:
        """Open a file with default application"""
        try:
            if os.name == "nt":  # Windows
                os.startfile(filepath)
            else:  # Unix-like
                await self.execute_system_command(f"xdg-open '{filepath}'")
            
            log.info(f"Opened file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to open file {filepath}: {e}")
            return False

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import psutil
        
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        }

    async def get_process_list(self) -> list:
        """Get list of running processes"""
        import psutil
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return processes

    async def kill_process(self, pid: int) -> bool:
        """Kill a process by PID"""
        import psutil
        
        try:
            process = psutil.Process(pid)
            process.terminate()
            log.info(f"Terminated process {pid}")
            return True
        except psutil.NoSuchProcess:
            log.warning(f"Process {pid} not found")
            return False
        except psutil.AccessDenied:
            log.error(f"Access denied to process {pid}")
            return False

    def get_command_history(self) -> list:
        """Get command execution history"""
        return self.command_history

    async def create_file(self, filepath: str, content: str) -> bool:
        """Create a file with content"""
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            log.info(f"Created file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to create file {filepath}: {e}")
            return False

    async def read_file(self, filepath: str) -> Optional[str]:
        """Read file content"""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            log.error(f"Failed to read file {filepath}: {e}")
            return None

    async def delete_file(self, filepath: str) -> bool:
        """Delete a file"""
        try:
            os.remove(filepath)
            log.info(f"Deleted file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to delete file {filepath}: {e}")
            return False

    async def list_directory(self, path: str) -> list:
        """List directory contents"""
        try:
            return os.listdir(path)
        except Exception as e:
            log.error(f"Failed to list directory {path}: {e}")
            return []

    async def create_directory(self, path: str) -> bool:
        """Create a directory"""
        try:
            os.makedirs(path, exist_ok=True)
            log.info(f"Created directory: {path}")
            return True
        except Exception as e:
            log.error(f"Failed to create directory {path}: {e}")
            return False

    async def get_environment_variable(self, name: str) -> Optional[str]:
        """Get environment variable"""
        return os.environ.get(name)

    async def set_environment_variable(self, name: str, value: str) -> bool:
        """Set environment variable"""
        try:
            os.environ[name] = value
            log.info(f"Set environment variable: {name}={value}")
            return True
        except Exception as e:
            log.error(f"Failed to set environment variable {name}: {e}")
            return False

# Example plugin template
EXAMPLE_PLUGIN_TEMPLATE = '''
from plugins.sdk import PluginBase

class Plugin(PluginBase):
    async def on_load(self):
        """Called when plugin is loaded"""
        print(f"[{self.name}] Plugin loaded successfully")
        
        # Access AI manager API
        if self.api:
            print(f"[{self.name}] AI Manager API available")
    
    async def on_unload(self):
        """Called when plugin is unloaded"""
        print(f"[{self.name}] Plugin unloaded")
    
    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands"""
        if "hello" in text.lower():
            print(f"[{self.name}] Hello command received!")
            return True
        return False
    
    async def on_ai_response(self, response: str):
        """Handle AI responses"""
        print(f"[{self.name}] AI Response: {response}")
    
    async def on_system_event(self, event_type: str, data: dict):
        """Handle system events"""
        print(f"[{self.name}] System event: {event_type}")
'''