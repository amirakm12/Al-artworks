# ChatGPT+ Clone

A fully-featured desktop AI assistant with GPU-optimized AI models, voice interaction, image generation, and extensible plugin system.

## 🚀 Features

### Core AI Capabilities
- **GPU-Aware AI Models**: Automatic device detection (CUDA > MPS > CPU)
- **Voice Recognition**: Real-time Whisper ASR with async processing
- **Text-to-Speech**: GPU-accelerated TTS synthesis
- **Local LLM**: Support for GPT-2, Dolphin-Mixtral, and other models
- **Image Generation**: Stable Diffusion with ControlNet editing
- **Code Execution**: Sandboxed Python code interpreter

### Advanced Features
- **Plugin System**: Hot-reloadable plugins with sandbox security
- **AR Overlay**: Holographic UI effects (experimental)
- **Voice Hotkeys**: Global hotkey activation (Ctrl+Shift+V)
- **Performance Monitoring**: Real-time GPU/CPU/RAM stats
- **Auto-Updater**: GitHub release-based updates
- **Cross-Platform**: Windows, Linux, macOS support

## 🏗️ Architecture

```
chatgpt-plus-clone/
├── main.py                 # Application entry point
├── device_utils.py         # GPU detection & optimization
├── voice_async.py          # Async voice processing
├── plugin_loader.py        # Plugin system & lifecycle
├── voice_hotkey.py         # Global hotkey management
├── voice_agent.py          # Speech recognition (Whisper)
├── voice_tts.py           # Text-to-speech synthesis
├── llm/model_loader.py     # Language model management
├── image_tools/            # Image generation & editing
│   ├── image_generator.py
│   └── controlnet_editor.py
├── overlay_ar.py           # AR/3D visual effects
├── settings_dialog.py      # Configuration UI
├── plugin_test_dialog.py   # Plugin debugging interface
├── build.py               # Cross-platform build system
├── performance_monitor.py  # System monitoring
├── config.json            # Application configuration
├── requirements.txt       # Dependencies
├── plugins/               # Plugin directory
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/chatgpt-plus-clone.git
   cd chatgpt-plus-clone
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

### Windows Installation (Recommended)
```powershell
# Run as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```

## 🎤 Voice Commands

- **Activation**: Press `Ctrl+Shift+V` to start voice input
- **Speaking**: Wait for confirmation beep, then speak clearly
- **Commands**: Try these voice commands:
  - "What's the weather today?"
  - "Write a Python function to calculate fibonacci"
  - "Generate an image of a sunset"
  - "Search for the latest news about AI"

## 🧩 Plugin System

### Using Plugins
Plugins automatically extend functionality:
- Add new voice commands
- Integrate external services
- Provide specialized tools
- Enhance the user interface

### Creating Plugins
```python
# plugins/my_plugin.py
class MyPlugin:
    def on_load(self):
        print("[Plugin] My plugin loaded")
        return True
    
    def on_unload(self):
        print("[Plugin] My plugin unloaded")
        return True
    
    def handle_voice_command(self, text, context):
        if "my command" in text.lower():
            return "Plugin response!"
        return None
```

## ⚙️ Configuration

### Settings Dialog
Access comprehensive settings through the UI:
- **Voice Settings**: Hotkey, sample rate, TTS options
- **AI Models**: Whisper, LLM, TTS, image generation models
- **Performance**: GPU acceleration, memory management
- **Plugins**: Enable/disable, test mode, sandboxing
- **Advanced**: AR overlay, security, debugging

### Configuration File
```json
{
  "voice_hotkey_enabled": true,
  "ar_overlay_enabled": true,
  "plugins_enabled": true,
  "gpu_acceleration": true,
  "voice_settings": {
    "hotkey": "ctrl+shift+v",
    "sample_rate": 16000,
    "silence_threshold": 0.01
  },
  "ai_models": {
    "whisper_model": "base",
    "llm_model": "gpt2",
    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
    "image_model": "runwayml/stable-diffusion-v1-5"
  }
}
```

## 🔧 Development

### Building from Source

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pyinstaller
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Build executable**:
   ```bash
   python build.py
   ```

### Plugin Development

1. **Create plugin file** in `plugins/` directory
2. **Implement required methods**:
   - `on_load()`: Plugin initialization
   - `on_unload()`: Plugin cleanup
   - `handle_voice_command()`: Voice command processing
3. **Test plugin** using the plugin test dialog
4. **Enable hot-reload** for development

### Testing Plugins
```bash
# Run plugin tests
python -m pytest tests/test_plugin_loader.py -v

# Use plugin test dialog
python plugin_test_dialog.py
```

## 🎨 AI Models

### GPU Optimization
The application automatically detects and uses the best available device:

- **CUDA GPU**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon GPU (macOS)
- **CPU**: Fallback for all platforms

### Supported Models

#### Speech Recognition (Whisper)
- Models: `tiny`, `base`, `small`, `medium`, `large`
- Automatic device selection
- Real-time transcription

#### Language Models
- **GPT-2**: Fast, lightweight
- **Dolphin-Mixtral**: High-quality responses
- **Custom models**: Hugging Face transformers

#### Text-to-Speech
- **Tacotron2-DDC**: High-quality synthesis
- **GPU acceleration**: Automatic optimization
- **Multiple voices**: Male, female, fast variants

#### Image Generation
- **Stable Diffusion**: High-quality image generation
- **ControlNet**: Advanced image editing
- **Multiple models**: SD 1.5, 2.1, custom models

## 📊 Performance Monitoring

### Real-time Stats
- **GPU Usage**: Memory, utilization, temperature
- **CPU Usage**: Overall and per-core
- **Memory**: RAM usage and allocation
- **Network**: Upload/download speeds

### Performance Optimization
- **Auto-cleanup**: Memory management
- **GPU cache clearing**: Manual and automatic
- **Model optimization**: Device-specific tuning
- **Resource limits**: Configurable thresholds

## 🔒 Security

### Plugin Sandboxing
- **Restricted execution**: Limited system access
- **Permission system**: Granular access control
- **Error isolation**: Plugin crashes don't affect main app
- **Resource limits**: Memory and CPU constraints

### Code Execution
- **Sandboxed Python**: Restricted environment
- **Timeout limits**: Configurable execution time
- **File access**: Controlled file operations
- **Network access**: Limited network permissions

## 🚀 Advanced Features

### AR Overlay
- **Holographic effects**: Neural network visualizations
- **Data flow animations**: Real-time processing display
- **3D effects**: Depth and perspective rendering
- **Performance**: Can be disabled on slower systems

### Auto-Updater
- **GitHub integration**: Release-based updates
- **Automatic checks**: Background update detection
- **Safe updates**: Verification and rollback
- **Changelog**: Release notes and features

### Cross-Platform Build
- **Windows**: PyInstaller executable
- **macOS**: App bundle creation
- **Linux**: AppImage and tar.gz packages
- **Dependencies**: Automatic bundling

## 🛠️ Troubleshooting

### Common Issues

#### Voice Not Working
1. **Check microphone**: Ensure device is connected and working
2. **Permissions**: Allow microphone access in system settings
3. **Hotkey conflict**: Verify `Ctrl+Shift+V` isn't used by other apps
4. **Audio drivers**: Update if needed

#### GPU Issues
1. **CUDA installation**: Ensure CUDA toolkit is installed
2. **Driver updates**: Update GPU drivers
3. **Memory**: Check GPU memory usage
4. **Fallback**: App will automatically use CPU if GPU fails

#### Plugin Errors
1. **Plugin test mode**: Enable for detailed error messages
2. **Dependencies**: Install required Python packages
3. **Permissions**: Check plugin security settings
4. **Logs**: Review plugin logs for specific errors

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CUDA-compatible for acceleration)

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Complete usage instructions
- **[Developer Guide](docs/DEV_GUIDE.md)**: Plugin development and contribution
- **[API Reference](docs/API.md)**: Technical documentation
- **[Examples](docs/EXAMPLES.md)**: Code examples and tutorials

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes**: Follow coding standards
4. **Add tests**: Ensure new features are tested
5. **Submit PR**: Include description and tests

### Coding Standards
- **Python**: Follow PEP 8 style guide
- **Documentation**: Use docstrings for all functions
- **Type Hints**: Include type annotations
- **Error Handling**: Comprehensive error handling
- **Testing**: Unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: Whisper speech recognition
- **Hugging Face**: Transformers and diffusers
- **PyQt6**: Modern UI framework
- **PyTorch**: GPU acceleration and AI models
- **Community**: Contributors and testers

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Community chat and support
- **Documentation**: Comprehensive guides and examples
- **Examples**: Sample plugins and configurations

---

**Happy AI Assisting! 🤖**

For the latest updates and community support, visit our GitHub repository.