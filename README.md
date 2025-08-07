# ChatGPT+ Clone

A comprehensive, GPU-optimized ChatGPT+ clone desktop application with advanced features including full voice interaction, autonomous task execution, and a robust plugin system.

## 🚀 Features

### Core AI Capabilities
- **GPU-Optimized AI Models**: Automatic GPU detection and utilization for AI models (Whisper, PyTorch LLMs, TTS, Image Generation)
- **Dynamic Precision Loading**: Support for fp16/bf16 with graceful CPU fallback
- **Dynamic GPU Load Balancing**: Automatic batch size tuning and GPU utilization optimization
- **Multiple AI Models**: Support for GPT-2, Whisper, Coqui TTS, Ollama, and more

### Voice System
- **Always-On Voice Listening**: Non-blocking voice capture with continuous processing
- **Real-Time Speech Recognition**: Whisper integration for accurate transcription
- **Text-to-Speech**: Coqui TTS for natural voice responses
- **Voice Activity Detection**: Smart detection of speech vs. silence
- **Global Hotkeys**: Ctrl+Shift+V to activate voice commands

### Advanced UI
- **AR/3D Overlay**: Real-time AI state visualization with PyQt6
- **System Tray Integration**: Background operation with system tray controls
- **Profiler Dashboard**: Real-time CPU/GPU/memory monitoring
- **Plugin Health Dashboard**: Plugin status and performance monitoring

### Plugin System
- **No Security Sandbox**: Full Python privileges for maximum flexibility
- **Async Plugin Hooks**: Event-driven architecture with async support
- **Hot Reloading**: Dynamic plugin loading and unloading
- **RBAC Security**: Role-Based Access Control for plugin permissions
- **Ultra-Complex Command Routing**: Advanced command routing with event bubbling

### System Control
- **Sovereign AI Agent**: Full system control capabilities
- **Process Management**: Start, stop, and monitor system processes
- **File Operations**: Create, read, delete, and manage files
- **System Commands**: Execute shell commands with full output capture
- **Remote Control**: WebSocket-based remote control interface

### Automation
- **Task Scheduler**: Cron-like job scheduling with APScheduler
- **Auto-Launcher**: Automatic startup on system boot
- **Update System**: Automatic updates with GitHub integration
- **Continuous Voice**: Always-on voice processing pipeline

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Dependencies
- **PyQt6**: Modern UI framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers
- **Whisper**: OpenAI's speech recognition
- **Coqui TTS**: Text-to-speech synthesis
- **SoundDevice**: Audio capture and playback
- **And more**: See `requirements.txt` for complete list

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd chatgpt-plus-clone

# Run the setup script
python3 setup.py

# Test the installation
python3 test_imports.py

# Start the application
python3 main.py
```

### Manual Installation
```bash
# Install system dependencies (Linux)
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev build-essential

# Install Python packages
pip install -r requirements.txt

# Create necessary directories
mkdir -p plugins logs downloads models voice gpu profiling remote_control build tests docs
```

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8 mypy isort

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 🎯 Usage

### Basic Usage
1. **Start the Application**: Run `python3 main.py`
2. **Voice Commands**: Press Ctrl+Shift+V to activate voice input
3. **Plugin Management**: Drop Python plugins in the `plugins/` directory
4. **System Control**: Use voice commands or the remote control interface

### Voice Commands
- "Hello" - Basic greeting and system test
- "What's the weather?" - Get weather information
- "Open file [filename]" - Open files with default applications
- "Show system info" - Display system statistics
- "Start [application]" - Launch applications

### Plugin Development
```python
from plugins.sdk import PluginBase

class Plugin(PluginBase):
    async def on_load(self):
        print("Plugin loaded!")
    
    async def on_voice_command(self, text: str) -> bool:
        if "hello" in text.lower():
            print("Hello command received!")
            return True
        return False
```

### Configuration
Edit `config.json` to customize:
- Voice settings (sample rate, hotkeys, etc.)
- AI model preferences
- GPU acceleration settings
- Plugin configurations
- Security settings

## 🏗️ Architecture

### Core Components
```
chatgpt-plus-clone/
├── main.py                 # Main application entry point
├── ai_agent.py            # Sovereign AI agent with system control
├── voice_agent.py         # Voice processing and TTS
├── plugin_loader.py       # Plugin management system
├── overlay_ar.py          # AR overlay and visualizations
├── voice_hotkey.py        # Global hotkey management
├── config.json            # Application configuration
├── requirements.txt       # Python dependencies
├── voice/                 # Voice processing modules
├── plugins/               # Plugin directory
├── gpu/                   # GPU optimization modules
├── profiling/             # System monitoring
├── remote_control/        # WebSocket remote control
└── build/                 # Build and packaging scripts
```

### Key Modules
- **SovereignAgent**: Full system control with AI model management
- **VoiceAgent**: Async voice processing with Whisper and TTS
- **PluginManager**: Dynamic plugin loading with async hooks
- **AROverlay**: Real-time system visualization
- **GPUTuner**: Dynamic GPU optimization and load balancing
- **TaskScheduler**: Automated task execution

## 🔧 Development

### Project Structure
```
├── main.py                    # Application entry point
├── ai_agent.py               # AI model management
├── voice_agent.py            # Voice processing
├── plugin_loader.py          # Plugin system
├── overlay_ar.py             # AR overlay
├── voice_hotkey.py           # Hotkey management
├── continuous_voice.py       # Always-on voice
├── task_scheduler.py         # Task automation
├── tray_app.py              # System tray
├── update_checker.py        # Auto-updates
├── logger.py                # Logging system
├── config.json              # Configuration
├── requirements.txt          # Dependencies
├── setup.py                 # Installation script
├── test_imports.py          # Import testing
├── voice/                   # Voice modules
│   ├── voice_agent_whisper.py
│   ├── tts_agent.py
│   └── async_voice_input.py
├── plugins/                 # Plugin system
│   ├── sdk.py              # Plugin SDK
│   └── sample_plugin.py    # Example plugin
├── gpu/                    # GPU optimization
│   └── gpu_tuning_loops.py
├── profiling/              # System monitoring
│   └── dashboard.py
├── remote_control/         # Remote control
│   └── server.py
├── build/                  # Build scripts
│   ├── build_installer.bat
│   └── installer.iss
└── docs/                   # Documentation
    ├── USER_GUIDE.md
    └── DEV_GUIDE.md
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_voice_agent.py

# Run with coverage
pytest --cov=. tests/
```

### Building
```bash
# Build executable (Windows)
python build/build_installer.bat

# Build executable (Linux/macOS)
pyinstaller --onefile main.py

# Create installer
python build/create_installer.py
```

## 🚨 Security

### Important Notes
- **No Sandbox**: Plugins run with full Python privileges
- **System Control**: The AI agent can execute system commands
- **File Access**: Plugins can read/write files on the system
- **Process Control**: Plugins can start/stop system processes

### Best Practices
- Only install plugins from trusted sources
- Review plugin code before installation
- Use RBAC to limit plugin permissions
- Monitor plugin activity through the dashboard

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Use Black for formatting, Flake8 for linting
2. **Type Hints**: Use type hints throughout the codebase
3. **Async/Await**: Prefer async/await over threading where possible
4. **Error Handling**: Comprehensive error handling and logging
5. **Documentation**: Document all public APIs and complex logic

### Plugin Development
1. Inherit from `PluginBase` in `plugins/sdk.py`
2. Implement required async hooks
3. Use the `AIManagerAPI` for system interactions
4. Follow the plugin template in `plugins/sample_plugin.py`

### Testing
1. Write tests for all new functionality
2. Use pytest for testing framework
3. Mock external dependencies
4. Test both success and failure scenarios

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: For Whisper speech recognition
- **Hugging Face**: For transformers and model hosting
- **Coqui AI**: For TTS synthesis
- **PyQt**: For the UI framework
- **PyTorch**: For deep learning capabilities

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check `docs/` for detailed guides
- **Discussions**: Use GitHub Discussions for questions
- **Wiki**: Community-maintained documentation

## 🔄 Updates

The system includes automatic updates via GitHub releases. Updates are checked daily and can be installed automatically or manually.

---

**Note**: This is a powerful AI system with full system control capabilities. Use responsibly and ensure you understand the security implications before deployment. 