# ChatGPT+ Clone - Automation Package

A comprehensive automation package for your ChatGPT+ clone with voice-driven control, scheduled tasks, system tray integration, and autonomous operation.

## 🚀 Features

### 🎤 Voice-Driven Automation
- **Continuous Voice Listening**: Always-on voice input with real-time processing
- **Voice Activity Detection**: Smart detection of speech vs. background noise
- **Voice Commands**: Control applications, open websites, execute system commands
- **Natural Language Processing**: Convert voice to text and process commands

### ⏰ Scheduled Tasks
- **Cron-like Scheduling**: Daily, hourly, weekly automated tasks
- **System Maintenance**: Automatic cleanup, log rotation, resource monitoring
- **Update Management**: Automatic version checking and updates
- **Custom Tasks**: Add your own scheduled operations

### 🖥️ System Integration
- **System Tray**: Always-accessible control panel
- **Auto-Launch**: Start on system boot with highest privileges
- **Remote Control**: WebSocket-based remote management
- **Resource Monitoring**: Real-time CPU, memory, GPU tracking

### 🔧 Plugin System
- **Extensible Architecture**: Add custom voice commands and automation
- **No Sandbox**: Full system access for powerful automation
- **Hot Reloading**: Update plugins without restarting
- **Event System**: Respond to voice commands and AI responses

## 📦 Installation

### Prerequisites
```bash
# Python 3.11+ required
python --version

# Install automation dependencies
pip install -r requirements-automation.txt
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/chatgpt-plus-clone.git
cd chatgpt-plus-clone

# Install dependencies
pip install -r requirements-automation.txt

# Run the application
python main.py
```

## 🎯 Usage

### Voice Commands

The system responds to natural voice commands:

```bash
# Application Control
"open browser"          # Opens default web browser
"open chatgpt"          # Opens ChatGPT in browser
"open gmail"            # Opens Gmail
"open calendar"         # Opens Google Calendar
"open settings"         # Opens system settings

# System Control
"get time"              # Announces current time
"get date"              # Announces current date
"system info"           # Shows system information
"lock computer"         # Locks the computer
"sleep computer"        # Puts computer to sleep

# Information
"say hello"             # Friendly greeting
"clear screen"          # Clears console
```

### System Tray

Right-click the system tray icon to access:

- **Start/Stop Service**: Control the main application
- **Show Status**: View system resources and uptime
- **Settings**: Configure application options
- **View Logs**: Open log directory
- **Check Updates**: Manual update check
- **Restart**: Restart the application
- **Quit**: Exit completely

### Auto-Launch Setup

#### Windows
```powershell
# Run as Administrator
.\auto_launcher.ps1

# Or manually set up scheduled task
schtasks /create /tn "ChatGPTPlusAutoRun" /tr "python main.py" /sc onlogon /ru SYSTEM
```

#### Linux/macOS
```bash
# Add to startup applications
# Linux: Add to ~/.config/autostart/
# macOS: Add to System Preferences > Users & Groups > Login Items
```

## 🔧 Configuration

### Main Configuration (`config.json`)

```json
{
  "voice_hotkey_enabled": true,
  "ar_overlay_enabled": true,
  "plugins_enabled": true,
  "remote_control_enabled": true,
  "profiling_enabled": true,
  "auto_update_enabled": true,
  "tray_app_enabled": true,
  "continuous_voice_enabled": true,
  "task_scheduler_enabled": true,
  
  "voice_settings": {
    "sample_rate": 16000,
    "channels": 1,
    "blocksize": 1024,
    "device": null
  },
  
  "ai_models": {
    "llm_model": "gpt2",
    "whisper_model": "base",
    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC"
  },
  
  "performance": {
    "gpu_acceleration": true,
    "max_concurrent_tasks": 5,
    "memory_limit_mb": 2048
  },
  
  "security": {
    "plugin_sandbox": false,
    "system_control": true,
    "auto_launch": true
  },
  
  "ui": {
    "theme": "dark",
    "language": "en",
    "notifications": true
  }
}
```

### Scheduled Tasks

The system automatically sets up these scheduled tasks:

- **Daily Cleanup** (2:00 AM): Clean old logs and temporary files
- **Hourly Status Check**: Monitor system resources
- **Weekly Update Check** (Sunday 10:00 AM): Check for updates

### Custom Scheduled Tasks

```python
# Add custom scheduled tasks
scheduler = TaskScheduler()

# Daily task at 8 AM
scheduler.add_cron_job(
    your_function,
    job_id="daily_task",
    hour=8,
    minute=0
)

# Every 30 minutes
scheduler.add_interval_job(
    your_function,
    job_id="interval_task",
    minutes=30
)

# One-time task
scheduler.add_date_job(
    your_function,
    job_id="one_time_task",
    run_date="2024-01-01 12:00:00"
)
```

## 🔌 Plugin Development

### Creating a Plugin

```python
from plugins.sdk import PluginBase

class MyPlugin(PluginBase):
    async def on_load(self):
        """Called when plugin is loaded"""
        print("My plugin loaded!")
    
    async def on_voice_command(self, text: str) -> bool:
        """Handle voice commands"""
        if "my command" in text.lower():
            # Execute your automation
            await self.api.execute_system_command("your_command")
            return True
        return False
    
    async def on_ai_response(self, response: str):
        """Handle AI responses"""
        print(f"AI said: {response}")
```

### Plugin API

Plugins have access to powerful system control:

```python
# System commands
await self.api.execute_system_command("ls -la")

# File operations
await self.api.open_file("/path/to/file")
await self.api.create_file("file.txt", "content")
await self.api.read_file("file.txt")

# System information
info = await self.api.get_system_info()
processes = await self.api.get_process_list()

# AI integration
response = await self.api.generate_response("Your prompt")
```

## 🎤 Voice Processing

### Continuous Voice Input

The system uses `sounddevice` for real-time audio processing:

```python
from continuous_voice import AsyncVoiceInput

# Create voice input
voice_input = AsyncVoiceInput(
    samplerate=16000,
    channels=1,
    blocksize=1024
)

# Process audio chunks
async def process_audio(chunk):
    # Your audio processing logic
    pass

# Start listening
await voice_input.start_listening(process_audio)
```

### Voice Activity Detection

```python
from continuous_voice import VoiceActivityDetector

vad = VoiceActivityDetector(
    threshold=0.01,      # Audio level threshold
    min_duration=0.5     # Minimum speech duration
)

# Detect speech in audio chunk
if vad.detect_activity(audio_chunk):
    speech_audio = vad.get_speech_audio()
```

## 🔄 Update System

### Automatic Updates

The system automatically checks for updates:

```python
from update_checker import UpdateChecker

checker = UpdateChecker(
    repo_owner="yourusername",
    repo_name="chatgpt-plus-clone",
    current_version="1.0.0"
)

# Check for updates
update_info = checker.get_update_info()
if update_info["update_available"]:
    checker.auto_update()
```

### Manual Update Check

```bash
# Check for updates manually
python -c "from update_checker import UpdateChecker; checker = UpdateChecker('yourusername', 'chatgpt-plus-clone'); print(checker.get_update_info())"
```

## 📊 Monitoring and Logging

### Log Files

The system creates comprehensive logs:

- `logs/chatgptplus.log` - Main application log
- `logs/chatgptplus_error.log` - Error messages only
- `logs/chatgptplus_debug.log` - Debug information
- `logs/voice/` - Voice processing logs
- `logs/ai/` - AI model logs
- `logs/plugin/` - Plugin activity logs
- `logs/system/` - System monitoring logs

### Performance Monitoring

```python
# Get system stats
import psutil

cpu_percent = psutil.cpu_percent()
memory_percent = psutil.virtual_memory().percent
disk_usage = psutil.disk_usage('/').percent

print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_usage}%")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Audio not working**
   ```bash
   # Check audio devices
   python -c "import sounddevice as sd; print(sd.query_devices())"
   
   # Test microphone
   python -c "import sounddevice as sd; sd.play(sd.rec(44100))"
   ```

2. **Plugins not loading**
   ```bash
   # Check plugin directory
   ls plugins/
   
   # Check plugin syntax
   python -m py_compile plugins/your_plugin.py
   ```

3. **System tray not showing**
   ```bash
   # Check if pystray is installed
   pip install pystray
   
   # Check system tray support
   python -c "import pystray; print('Tray supported')"
   ```

4. **Scheduled tasks not running**
   ```bash
   # Check scheduler status
   python -c "from task_scheduler import TaskScheduler; s = TaskScheduler(); print(s.get_stats())"
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

1. **High CPU usage**
   - Reduce voice processing frequency
   - Disable unused features
   - Check for memory leaks

2. **Memory issues**
   - Increase memory limit in config
   - Enable log rotation
   - Clean up old files

## 🔒 Security Considerations

### Plugin Security

⚠️ **Warning**: Plugins run with full system privileges (no sandbox)

- Only install plugins from trusted sources
- Review plugin code before installation
- Monitor plugin activity in logs
- Use separate user account for testing

### System Control

The application can:
- Execute system commands
- Access files and directories
- Control system power (sleep, shutdown)
- Launch applications
- Access network resources

### Best Practices

1. **Regular Updates**: Keep the application updated
2. **Log Monitoring**: Review logs regularly
3. **Backup Configuration**: Backup your config files
4. **Test Plugins**: Test plugins in isolated environment
5. **Monitor Resources**: Watch system resource usage

## 📈 Advanced Features

### Custom Voice Commands

Add custom commands to `plugin_auto_task.py`:

```python
async def _custom_command(self, text: str) -> str:
    """Your custom command"""
    # Your automation logic
    return "Command executed"
```

### Remote Control

Access via WebSocket at `ws://localhost:8765`:

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8765');

ws.send(JSON.stringify({
    type: "execute_command",
    command: "dir"
}));
```

### GPU Acceleration

Enable GPU acceleration for AI models:

```python
# In config.json
{
  "performance": {
    "gpu_acceleration": true,
    "gpu_memory_fraction": 0.8
  }
}
```

## 🤝 Contributing

### Adding New Features

1. Create feature branch
2. Implement functionality
3. Add tests
4. Update documentation
5. Submit pull request

### Plugin Development

1. Follow plugin template
2. Test thoroughly
3. Document commands
4. Add error handling
5. Submit to plugin repository

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: See docs/ folder
- **Discussions**: GitHub Discussions
- **Email**: your-email@example.com

---

**Happy Automating! 🤖✨**