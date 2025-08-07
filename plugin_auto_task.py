import webbrowser
import logging
import os
import subprocess
import platform
import time
from typing import Dict, Any, Optional
from plugins.sdk import PluginBase

logger = logging.getLogger("AutoTaskPlugin")

class AutoTaskPlugin(PluginBase):
    """Plugin that automates tasks based on voice commands"""
    
    def __init__(self, api=None):
        super().__init__(api)
        self.name = "AutoTaskPlugin"
        self.commands_processed = 0
        self.last_command_time = 0
        
        # Define command mappings
        self.command_mappings = {
            "open browser": self._open_browser,
            "open chatgpt": self._open_chatgpt,
            "open gmail": self._open_gmail,
            "open calendar": self._open_calendar,
            "open drive": self._open_drive,
            "open youtube": self._open_youtube,
            "open github": self._open_github,
            "open settings": self._open_settings,
            "open file explorer": self._open_file_explorer,
            "open notepad": self._open_notepad,
            "open calculator": self._open_calculator,
            "say hello": self._say_hello,
            "get time": self._get_time,
            "get date": self._get_date,
            "system info": self._get_system_info,
            "clear screen": self._clear_screen,
            "restart computer": self._restart_computer,
            "shutdown computer": self._shutdown_computer,
            "sleep computer": self._sleep_computer,
            "lock computer": self._lock_computer
        }

    async def on_load(self):
        """Called when plugin is loaded"""
        logger.info(f"[{self.name}] AutoTask plugin loaded successfully")
        logger.info(f"[{self.name}] Available commands: {list(self.command_mappings.keys())}")

    async def on_unload(self):
        """Called when plugin is unloaded"""
        logger.info(f"[{self.name}] AutoTask plugin unloaded. Processed {self.commands_processed} commands")

    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands and execute corresponding tasks"""
        text_lower = text.lower().strip()
        self.commands_processed += 1
        self.last_command_time = time.time()
        
        logger.info(f"[{self.name}] Processing voice command: '{text}'")
        
        # Check for exact matches first
        for command, handler in self.command_mappings.items():
            if command in text_lower:
                try:
                    result = await handler(text)
                    logger.info(f"[{self.name}] Executed command '{command}': {result}")
                    return True
                except Exception as e:
                    logger.error(f"[{self.name}] Error executing command '{command}': {e}")
                    return True
        
        # Check for partial matches
        for command, handler in self.command_mappings.items():
            if any(word in text_lower for word in command.split()):
                try:
                    result = await handler(text)
                    logger.info(f"[{self.name}] Executed partial match '{command}': {result}")
                    return True
                except Exception as e:
                    logger.error(f"[{self.name}] Error executing partial match '{command}': {e}")
                    return True
        
        return False

    async def _open_browser(self, text: str) -> str:
        """Open default web browser"""
        webbrowser.open("https://www.google.com")
        return "Browser opened"

    async def _open_chatgpt(self, text: str) -> str:
        """Open ChatGPT"""
        webbrowser.open("https://chat.openai.com")
        return "ChatGPT opened"

    async def _open_gmail(self, text: str) -> str:
        """Open Gmail"""
        webbrowser.open("https://mail.google.com")
        return "Gmail opened"

    async def _open_calendar(self, text: str) -> str:
        """Open Google Calendar"""
        webbrowser.open("https://calendar.google.com")
        return "Calendar opened"

    async def _open_drive(self, text: str) -> str:
        """Open Google Drive"""
        webbrowser.open("https://drive.google.com")
        return "Google Drive opened"

    async def _open_youtube(self, text: str) -> str:
        """Open YouTube"""
        webbrowser.open("https://www.youtube.com")
        return "YouTube opened"

    async def _open_github(self, text: str) -> str:
        """Open GitHub"""
        webbrowser.open("https://github.com")
        return "GitHub opened"

    async def _open_settings(self, text: str) -> str:
        """Open system settings"""
        if platform.system() == "Windows":
            subprocess.run(["start", "ms-settings:"], shell=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "-a", "System Preferences"])
        else:  # Linux
            subprocess.run(["gnome-control-center"])
        return "Settings opened"

    async def _open_file_explorer(self, text: str) -> str:
        """Open file explorer"""
        if platform.system() == "Windows":
            subprocess.run(["explorer"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "."])
        else:  # Linux
            subprocess.run(["xdg-open", "."])
        return "File explorer opened"

    async def _open_notepad(self, text: str) -> str:
        """Open notepad/text editor"""
        if platform.system() == "Windows":
            subprocess.run(["notepad"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "-a", "TextEdit"])
        else:  # Linux
            subprocess.run(["gedit"])
        return "Notepad opened"

    async def _open_calculator(self, text: str) -> str:
        """Open calculator"""
        if platform.system() == "Windows":
            subprocess.run(["calc"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", "-a", "Calculator"])
        else:  # Linux
            subprocess.run(["gnome-calculator"])
        return "Calculator opened"

    async def _say_hello(self, text: str) -> str:
        """Say hello"""
        return "Hello! How can I assist you today?"

    async def _get_time(self, text: str) -> str:
        """Get current time"""
        current_time = time.strftime("%H:%M:%S")
        return f"Current time is {current_time}"

    async def _get_date(self, text: str) -> str:
        """Get current date"""
        current_date = time.strftime("%A, %B %d, %Y")
        return f"Today is {current_date}"

    async def _get_system_info(self, text: str) -> str:
        """Get system information"""
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        return f"System info: {system_info}"

    async def _clear_screen(self, text: str) -> str:
        """Clear console screen"""
        os.system('cls' if platform.system() == "Windows" else 'clear')
        return "Screen cleared"

    async def _restart_computer(self, text: str) -> str:
        """Restart computer (requires confirmation)"""
        logger.warning(f"[{self.name}] Restart computer command received - requires confirmation")
        return "Restart command received. Please confirm before proceeding."

    async def _shutdown_computer(self, text: str) -> str:
        """Shutdown computer (requires confirmation)"""
        logger.warning(f"[{self.name}] Shutdown computer command received - requires confirmation")
        return "Shutdown command received. Please confirm before proceeding."

    async def _sleep_computer(self, text: str) -> str:
        """Put computer to sleep"""
        if platform.system() == "Windows":
            subprocess.run(["powercfg", "/hibernate", "off"])
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["pmset", "sleepnow"])
        else:  # Linux
            subprocess.run(["systemctl", "suspend"])
        return "Computer going to sleep"

    async def _lock_computer(self, text: str) -> str:
        """Lock computer"""
        if platform.system() == "Windows":
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["pmset", "displaysleepnow"])
        else:  # Linux
            subprocess.run(["gnome-screensaver-command", "--lock"])
        return "Computer locked"

    async def on_ai_response(self, response: str):
        """Handle AI responses"""
        logger.info(f"[{self.name}] Received AI response: {response[:100]}...")

    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            "plugin_name": self.name,
            "commands_processed": self.commands_processed,
            "last_command_time": self.last_command_time,
            "uptime": time.time() - self.last_command_time if self.last_command_time > 0 else 0,
            "available_commands": list(self.command_mappings.keys())
        }

# Example usage
async def example_auto_task_plugin():
    """Example of using the AutoTask plugin"""
    logging.basicConfig(level=logging.INFO)
    
    # Create plugin
    plugin = AutoTaskPlugin()
    
    # Load plugin
    await plugin.on_load()
    
    # Test voice commands
    test_commands = [
        "open browser",
        "open chatgpt",
        "say hello",
        "get time",
        "get date",
        "system info"
    ]
    
    for command in test_commands:
        logger.info(f"Testing command: {command}")
        handled = await plugin.on_voice_command(command)
        if handled:
            logger.info(f"  ✓ Command handled: {command}")
        else:
            logger.info(f"  ✗ Command not handled: {command}")
    
    # Get stats
    stats = plugin.get_plugin_stats()
    logger.info(f"Plugin stats: {stats}")
    
    # Unload plugin
    await plugin.on_unload()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_auto_task_plugin())