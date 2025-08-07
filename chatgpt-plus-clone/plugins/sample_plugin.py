# plugins/sample_plugin.py
"""
Sample Plugin - Example plugin for ChatGPT+ Clone
Demonstrates plugin system functionality with voice commands
"""

def on_load():
    """Plugin initialization - called when plugin is loaded"""
    return {
        "name": "Sample Plugin",
        "version": "1.0.0",
        "description": "A sample plugin that responds to voice commands",
        "author": "ChatGPT+ Clone Team",
        "hooks": {
            "on_voice_command": hello_world,
            "on_message_received": handle_message,
            "on_tool_executed": handle_tool
        },
        "permissions": ["voice_access", "read_files"],
        "commands": {
            "hello": "Say hello world",
            "time": "Get current time",
            "weather": "Get weather information (mock)"
        }
    }

def hello_world(text, context):
    """Handle voice command 'hello'"""
    print("🧩 Sample Plugin Activated - Text:", text)
    
    if "hello" in text.lower():
        return "Hello! I'm the sample plugin. Nice to meet you! 👋"
    elif "time" in text.lower():
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time} ⏰"
    elif "weather" in text.lower():
        return "The weather is sunny with a chance of AI! ☀️"
    else:
        return f"I heard you say: '{text}'. This is the sample plugin responding! 🎤"

def handle_message(message, context):
    """Handle incoming chat messages"""
    print(f"📝 Sample Plugin - Message received: {message}")
    
    if "plugin" in message.lower():
        return "Sample plugin is active and listening! 🔌"
    
    return None  # Don't respond to other messages

def handle_tool(tool_name, result, context):
    """Handle tool execution results"""
    print(f"🛠️ Sample Plugin - Tool executed: {tool_name}")
    
    if tool_name == "voice":
        return f"Voice tool was used. Sample plugin noticed! 🎤"
    elif tool_name == "code_interpreter":
        return f"Code was executed. Sample plugin is monitoring! 💻"
    
    return None

def on_unload():
    """Plugin cleanup - called when plugin is unloaded"""
    print("🧩 Sample Plugin - Unloading...")
    return "Sample plugin unloaded successfully"