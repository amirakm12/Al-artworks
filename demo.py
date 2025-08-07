#!/usr/bin/env python3
"""Demo script for ChatGPT+ Clone - shows basic functionality without full dependencies"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("Demo")

class MockAIAgent:
    """Mock AI agent for demo purposes"""
    
    def __init__(self):
        self.running = False
        self.command_history = []
        
    async def generate_response(self, prompt: str) -> str:
        """Generate a mock AI response"""
        responses = [
            f"I understand you said: '{prompt}'. This is a demo response.",
            f"Processing your request: '{prompt}'. Here's what I can do for you.",
            f"Demo AI response to: '{prompt}'. The system is working correctly.",
            f"Voice command received: '{prompt}'. This is a simulated response.",
        ]
        import random
        return random.choice(responses)
    
    async def execute_command(self, cmd: str) -> dict:
        """Mock command execution"""
        return {
            "command": cmd,
            "return_code": 0,
            "stdout": f"Demo: Executed '{cmd}' successfully",
            "stderr": "",
            "success": True
        }

class MockPluginManager:
    """Mock plugin manager for demo purposes"""
    
    def __init__(self):
        self.plugins = []
        self.command_handlers = {
            "hello": self._handle_hello,
            "time": self._handle_time,
            "weather": self._handle_weather,
            "system": self._handle_system,
        }
    
    async def dispatch_voice_command(self, text: str) -> bool:
        """Dispatch voice commands to handlers"""
        text_lower = text.lower()
        
        for keyword, handler in self.command_handlers.items():
            if keyword in text_lower:
                await handler(text)
                return True
        
        return False
    
    async def _handle_hello(self, text: str):
        """Handle hello commands"""
        log.info(f"Plugin: Hello command received: '{text}'")
        print("🤖 Plugin: Hello! How can I help you today?")
    
    async def _handle_time(self, text: str):
        """Handle time commands"""
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        log.info(f"Plugin: Time command received: '{text}'")
        print(f"🤖 Plugin: Current time is {current_time}")
    
    async def _handle_weather(self, text: str):
        """Handle weather commands"""
        log.info(f"Plugin: Weather command received: '{text}'")
        print("🤖 Plugin: Weather information would be retrieved here in a real implementation.")
    
    async def _handle_system(self, text: str):
        """Handle system commands"""
        log.info(f"Plugin: System command received: '{text}'")
        print("🤖 Plugin: System information would be displayed here.")

class MockVoiceAgent:
    """Mock voice agent for demo purposes"""
    
    def __init__(self, plugin_manager, ai_manager):
        self.plugin_manager = plugin_manager
        self.ai_manager = ai_manager
        self.running = False
        self.is_listening = False
    
    async def start(self):
        """Start the voice agent"""
        self.running = True
        log.info("Mock Voice Agent started")
    
    async def stop(self):
        """Stop the voice agent"""
        self.running = False
        log.info("Mock Voice Agent stopped")
    
    async def process_voice_command(self, text: str) -> str:
        """Process a voice command"""
        log.info(f"Processing voice command: {text}")
        
        # Try plugins first
        handled = await self.plugin_manager.dispatch_voice_command(text)
        
        if not handled and self.ai_manager:
            response = await self.ai_manager.generate_response(text)
            print(f"🤖 AI: {response}")
            return response
        
        return "Command processed by plugin"

class DemoSystem:
    """Demo system that shows the ChatGPT+ Clone functionality"""
    
    def __init__(self):
        self.config = self._load_config()
        self.ai_agent = MockAIAgent()
        self.plugin_manager = MockPluginManager()
        self.voice_agent = MockVoiceAgent(self.plugin_manager, self.ai_agent)
        self.running = False
    
    def _load_config(self) -> dict:
        """Load configuration"""
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning("config.json not found, using defaults")
            return {
                "voice_hotkey_enabled": True,
                "plugins_enabled": True,
                "demo_mode": True
            }
    
    async def start(self):
        """Start the demo system"""
        self.running = True
        log.info("=== ChatGPT+ Clone Demo Starting ===")
        
        # Start voice agent
        await self.voice_agent.start()
        
        print("\n🎤 ChatGPT+ Clone Demo System")
        print("=" * 50)
        print("This is a demo of the voice command system.")
        print("Type commands to simulate voice input:")
        print("  - hello")
        print("  - what time is it")
        print("  - weather")
        print("  - system info")
        print("  - exit (to quit)")
        print("=" * 50)
        
        # Main demo loop
        while self.running:
            try:
                # Simulate voice input
                command = input("\n🎤 Voice Command: ").strip()
                
                if command.lower() == "exit":
                    break
                
                if command:
                    # Process the command
                    await self.voice_agent.process_voice_command(command)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Error in demo loop: {e}")
    
    async def stop(self):
        """Stop the demo system"""
        self.running = False
        await self.voice_agent.stop()
        log.info("=== Demo System Stopped ===")

async def main():
    """Main demo function"""
    demo = DemoSystem()
    
    try:
        await demo.start()
    except KeyboardInterrupt:
        log.info("Demo interrupted by user")
    finally:
        await demo.stop()

def show_system_info():
    """Show system information"""
    print("\n📊 System Information")
    print("=" * 30)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check if key files exist
    files_to_check = [
        "config.json",
        "main.py",
        "ai_agent.py",
        "voice_agent.py",
        "plugin_loader.py",
        "requirements.txt",
        "setup.py"
    ]
    
    print("\n📁 File Structure:")
    for file in files_to_check:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Check directories
    dirs_to_check = [
        "plugins",
        "voice",
        "gpu",
        "profiling",
        "remote_control",
        "build",
        "docs"
    ]
    
    print("\n📂 Directories:")
    for dir in dirs_to_check:
        exists = Path(dir).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir}/")
    
    # Show config info
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print(f"\n⚙️  Configuration:")
        print(f"  Voice Hotkey: {config.get('voice_hotkey_enabled', False)}")
        print(f"  AR Overlay: {config.get('ar_overlay_enabled', False)}")
        print(f"  Plugins: {config.get('plugins_enabled', False)}")
    except Exception as e:
        print(f"  Config Error: {e}")

if __name__ == "__main__":
    print("🚀 ChatGPT+ Clone Demo")
    print("=" * 50)
    
    # Show system info
    show_system_info()
    
    # Ask user if they want to run the demo
    response = input("\nWould you like to run the voice command demo? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        asyncio.run(main())
    else:
        print("Demo skipped. Run 'python3 demo.py' again to try the demo.")
        print("\nTo install dependencies, run:")
        print("  python3 setup.py")
        print("\nTo test imports, run:")
        print("  python3 test_imports.py")