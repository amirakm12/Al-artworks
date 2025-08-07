#!/usr/bin/env python3
"""
Example Plugin - Demonstrates Plugin Class Pattern
Shows how to create plugins with start/stop lifecycle methods
"""

import time
import threading
from typing import Dict, Any, Optional

class Plugin:
    """Example plugin with lifecycle management"""
    
    def __init__(self):
        self.name = "Example Plugin"
        self.version = "1.0.0"
        self.description = "A simple example plugin with start/stop lifecycle"
        self.author = "ChatGPT+ Clone Team"
        
        # Plugin state
        self.is_running = False
        self.background_thread = None
        self.config = {}
        
        # Plugin capabilities
        self.hooks = {
            "on_voice_command": self.handle_voice_command,
            "on_message_received": self.handle_message,
            "on_tool_executed": self.handle_tool
        }
        
        # Commands this plugin responds to
        self.commands = {
            "hello": "Say hello world",
            "time": "Get current time",
            "status": "Get plugin status",
            "config": "Show plugin configuration"
        }
    
    def start(self):
        """Start the plugin"""
        if self.is_running:
            print(f"[{self.name}] Already running")
            return
        
        print(f"[{self.name}] Starting...")
        self.is_running = True
        
        # Start background thread for periodic tasks
        self.background_thread = threading.Thread(target=self._background_task, daemon=True)
        self.background_thread.start()
        
        # Load configuration
        self._load_config()
        
        print(f"[{self.name}] Started successfully")
    
    def stop(self):
        """Stop the plugin"""
        if not self.is_running:
            print(f"[{self.name}] Not running")
            return
        
        print(f"[{self.name}] Stopping...")
        self.is_running = False
        
        # Wait for background thread to finish
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=2.0)
        
        # Save configuration
        self._save_config()
        
        print(f"[{self.name}] Stopped successfully")
    
    def _background_task(self):
        """Background task that runs while plugin is active"""
        while self.is_running:
            try:
                # Do periodic work here
                time.sleep(5)  # Check every 5 seconds
                
                # Example: log status periodically
                if self.is_running:
                    print(f"[{self.name}] Background task running...")
                    
            except Exception as e:
                print(f"[{self.name}] Background task error: {e}")
                break
    
    def handle_voice_command(self, text: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Handle voice commands"""
        if not self.is_running:
            return None
        
        text_lower = text.lower()
        
        if "hello" in text_lower:
            return f"Hello from {self.name}! 👋"
        
        elif "time" in text_lower:
            current_time = time.strftime("%H:%M:%S")
            return f"Current time is {current_time}"
        
        elif "status" in text_lower:
            return f"{self.name} is {'running' if self.is_running else 'stopped'}"
        
        elif "config" in text_lower:
            config_str = ", ".join([f"{k}: {v}" for k, v in self.config.items()])
            return f"Plugin config: {config_str or 'No config'}"
        
        return None
    
    def handle_message(self, message: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Handle text messages"""
        if not self.is_running:
            return None
        
        # Example: respond to specific keywords
        if "plugin" in message.lower():
            return f"{self.name} is here to help!"
        
        return None
    
    def handle_tool(self, tool_name: str, result: Any, context: Dict[str, Any] = None) -> Optional[str]:
        """Handle tool execution results"""
        if not self.is_running:
            return None
        
        # Example: log tool usage
        print(f"[{self.name}] Tool '{tool_name}' was executed")
        return None
    
    def _load_config(self):
        """Load plugin configuration"""
        try:
            # This would load from the config manager
            self.config = {
                "enabled": True,
                "log_level": "INFO",
                "custom_setting": "default_value"
            }
            print(f"[{self.name}] Configuration loaded")
        except Exception as e:
            print(f"[{self.name}] Error loading config: {e}")
            self.config = {}
    
    def _save_config(self):
        """Save plugin configuration"""
        try:
            # This would save to the config manager
            print(f"[{self.name}] Configuration saved")
        except Exception as e:
            print(f"[{self.name}] Error saving config: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "running": self.is_running,
            "commands": self.commands,
            "config": self.config
        }
    
    def set_config(self, key: str, value: Any):
        """Set a configuration value"""
        self.config[key] = value
        print(f"[{self.name}] Config updated: {key} = {value}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)

# Legacy support - also support the old on_load pattern
def on_load():
    """Legacy plugin loading function"""
    return {
        "name": "Example Plugin (Legacy)",
        "version": "1.0.0",
        "description": "Legacy example plugin",
        "hooks": {
            "on_voice_command": lambda text, context: f"Legacy plugin says: {text}"
        }
    }

if __name__ == "__main__":
    # Test the plugin
    print("🧪 Testing Example Plugin...")
    
    plugin = Plugin()
    
    # Test start
    plugin.start()
    
    # Test voice command handling
    result = plugin.handle_voice_command("hello world")
    print(f"Voice command result: {result}")
    
    # Test message handling
    result = plugin.handle_message("This is a test message")
    print(f"Message result: {result}")
    
    # Test stop
    plugin.stop()
    
    print("✅ Example plugin test completed")