# 🚀 ChatGPT+ Clone - Advanced AI Assistant

A comprehensive Windows desktop application that replicates ChatGPT Plus functionality with local AI capabilities, voice interaction, code execution, image generation, and more.

## ✨ Features

### 🤖 **AI Chat & Language Models**
- **Local LLM Support**: dolphin-mixtral:8x22b, LLaMA2, Mistral via Ollama
- **Conversation Memory**: Persistent chat history with vector storage
- **Context Awareness**: Intelligent conversation context management
- **Multi-Model Support**: Switch between different AI models

### 🎤 **Voice Interaction**
- **Speech-to-Text**: Whisper-powered voice recognition
- **Text-to-Speech**: Natural voice synthesis
- **Global Hotkey**: Ctrl+Shift+V for instant voice activation
- **Voice Activity Detection**: Automatic silence detection

### 💻 **Code Interpreter**
- **Python Execution**: Sandboxed code execution environment
- **File Processing**: Upload and analyze code files
- **Real-time Output**: Live code execution feedback
- **Virtual Environment**: Isolated Python execution

### 🌐 **Web Browser Agent**
- **Real-time Search**: DuckDuckGo and Brave Search integration
- **Web Scraping**: Intelligent content extraction
- **Browser Automation**: Playwright-powered web interaction
- **Search History**: Persistent search results

### 🎨 **Image Generation & Editing**
- **Stable Diffusion**: Local image generation
- **DALL-E Style**: Advanced prompt-based image creation
- **Image Editing**: ControlNet-powered image manipulation
- **Batch Processing**: Multiple image generation

### 📁 **File Management**
- **Drag & Drop**: Easy file upload interface
- **File Browser**: Integrated file system navigation
- **File Processing**: Automatic file type detection
- **Workspace Management**: Organized file storage

### 🔌 **Plugin System**
- **Dynamic Loading**: Hot-reloadable plugin architecture
- **Sandbox Security**: Restricted execution environment
- **Plugin Discovery**: Automatic plugin detection
- **Custom Tools**: Extensible functionality

### 🧠 **Memory System**
- **Vector Storage**: ChromaDB-powered semantic search
- **Conversation History**: Persistent chat storage
- **User Preferences**: Customizable settings
- **Context Memory**: Intelligent context management

### 🖥️ **VS Code Integration**
- **Monaco Editor**: Embedded code editor
- **File Linking**: Direct VS Code integration
- **Code Execution**: Inline code running
- **Workspace Sync**: Real-time file synchronization

### 🎭 **AR/3D Overlay**
- **Holographic UI**: Futuristic visual interface
- **Neural Visualization**: Animated neural network display
- **System Status**: Real-time system monitoring
- **Glass HUD**: Transparent overlay effects

## 🛠️ Installation

### Quick Start (Windows)

1. **Download and Extract**
   ```powershell
   # Clone or download the project
   git clone https://github.com/your-repo/chatgpt-plus-clone.git
   cd chatgpt-plus-clone
   ```

2. **Run Installation Script**
   ```powershell
   # Run as Administrator for best results
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\install.ps1
   ```

3. **Launch Application**
   ```powershell
   # Start the application
   .\start.ps1
   ```

### Manual Installation

1. **Prerequisites**
   - Python 3.10+ (automatically installed by script)
   - Windows 10/11 (64-bit)
   - 8GB RAM minimum (16GB recommended)
   - 10GB free disk space

2. **Install Dependencies**
   ```bash
   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install Ollama**
   ```bash
   # Download and install Ollama
   # https://ollama.ai/download
   
   # Pull AI models
   ollama pull dolphin-mixtral:8x22b
   ollama pull llama2:13b
   ollama pull mistral:7b
   ```

4. **Install Playwright Browsers**
   ```bash
   playwright install
   ```

## 🚀 Usage

### Basic Operation

1. **Launch Application**
   - Run `start.ps1` or double-click `ChatGPTPlusClone.exe`
   - The application will start with a modern Qt6 interface

2. **Voice Interaction**
   - Press `Ctrl+Shift+V` to activate voice input
   - Speak your message clearly
   - The AI will respond with voice synthesis

3. **Chat Interface**
   - Type messages in the chat area
   - Use the tools panel for different capabilities
   - Drag and drop files for processing

### Advanced Features

#### Code Interpreter
```python
# Example: Ask the AI to run Python code
"Can you create a function that calculates fibonacci numbers?"
```

#### Image Generation
```
"Generate an image of a futuristic city with flying cars"
```

#### Web Search
```
"Search for the latest news about artificial intelligence"
```

#### File Processing
- Drag and drop Python files for analysis
- Upload images for editing
- Process text files for summarization

### Plugin Development

1. **Create Plugin Directory**
   ```bash
   mkdir plugins/my_plugin
   ```

2. **Create Manifest**
   ```json
   {
     "name": "my_plugin",
     "version": "1.0.0",
     "description": "My custom plugin",
     "author": "Your Name",
     "dependencies": [],
     "hooks": ["message_received"],
     "permissions": ["read_files"]
   }
   ```

3. **Create Plugin Code**
   ```python
   def register(plugin_manager):
       plugin_manager.register_hook("message_received", handle_message)
   
   def handle_message(message):
       return f"Plugin processed: {message}"
   ```

## 🔧 Configuration

### Application Settings

Edit `config.json` to customize the application:

```json
{
  "app_name": "ChatGPT+ Clone",
  "version": "1.0.0",
  "default_model": "dolphin-mixtral:8x22b",
  "voice_hotkey": "ctrl+shift+v",
  "max_memory_size": "1GB",
  "enable_plugins": true,
  "enable_voice": true,
  "enable_image_generation": true,
  "enable_code_execution": true,
  "enable_web_search": true,
  "log_level": "INFO"
}
```

### Model Configuration

The application supports multiple AI models:

- **dolphin-mixtral:8x22b**: Best performance, requires more RAM
- **llama2:13b**: Good balance of performance and memory
- **mistral:7b**: Fastest, works on lower-end systems

### Voice Settings

Configure voice processing in the settings:

- **Sample Rate**: 16000 Hz (default)
- **Silence Threshold**: 0.01 (adjust for microphone sensitivity)
- **Max Recording Time**: 30 seconds
- **Voice Activity Detection**: Automatic silence detection

## 🏗️ Building from Source

### Development Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-repo/chatgpt-plus-clone.git
   cd chatgpt-plus-clone
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run in Development Mode**
   ```bash
   python main.py
   ```

### Building Executable

1. **Install PyInstaller**
   ```bash
   pip install pyinstaller
   ```

2. **Build Executable**
   ```bash
   .\build.bat
   ```

3. **Distribution Package**
   - Executable will be created in `dist/ChatGPTPlusClone/`
   - Run `install.bat` in the distribution folder
   - Launch `ChatGPTPlusClone.exe`

## 📁 Project Structure

```
chatgpt-plus-clone/
├── main.py                 # Main application entry point
├── requirements.txt        # Complete dependencies
├── requirements-*.txt      # Tiered dependency files
├── install.ps1            # Windows installation script
├── build.bat              # Build script for executable
├── config.json            # Application configuration
├── ui/                    # User interface components
│   ├── chat_interface.py  # Chat UI
│   ├── tools_panel.py     # Tools panel
│   ├── file_browser.py    # File browser
│   └── voice_panel.py     # Voice interface
├── llm/                   # Language model integration
│   └── agent_orchestrator.py
├── tools/                 # AI tools and capabilities
│   ├── code_executor.py   # Python code execution
│   ├── web_browser.py     # Web search and scraping
│   ├── image_editor.py    # Image generation/editing
│   └── voice_agent.py     # Voice processing
├── memory/                # Memory and storage system
│   └── memory_manager.py  # Persistent memory management
├── plugins/               # Plugin system
│   └── sample_plugin/     # Example plugin
├── workspace/             # User workspace
├── vs_code_link/          # VS Code integration
├── plugin_loader.py       # Plugin management system
├── voice_hotkey.py        # Global voice hotkey
└── overlay_ar.py          # AR/3D overlay system
```

## 🔌 Plugin API

### Plugin Structure

```
plugins/my_plugin/
├── manifest.json          # Plugin metadata
└── plugin.py             # Plugin code
```

### Available Hooks

- `message_received`: Called when user sends a message
- `tool_executed`: Called when a tool is executed
- `file_uploaded`: Called when a file is uploaded
- `voice_activated`: Called when voice input is activated

### Plugin Permissions

- `read_files`: Access to file system
- `execute_code`: Execute Python code
- `web_access`: Access to web resources
- `voice_access`: Access to voice features

## 🐛 Troubleshooting

### Common Issues

1. **Voice Not Working**
   - Check microphone permissions
   - Ensure Whisper is installed: `pip install openai-whisper`
   - Test microphone in Windows settings

2. **Slow Performance**
   - Close other applications
   - Use smaller models (mistral:7b)
   - Increase system RAM

3. **Model Errors**
   - Ensure Ollama is running: `ollama serve`
   - Check model installation: `ollama list`
   - Pull models: `ollama pull dolphin-mixtral:8x22b`

4. **Build Errors**
   - Ensure PyInstaller is installed
   - Check Python version (3.10+)
   - Run as administrator

### Logs and Debugging

- Application logs: `logs/` directory
- Memory database: `memory/memory.db`
- Configuration: `config.json`
- Plugin logs: Check individual plugin directories

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to functions
- Include docstrings for classes and methods
- Write tests for new features
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference
- **Whisper**: Speech recognition
- **Stable Diffusion**: Image generation
- **PyQt6**: Modern UI framework
- **ChromaDB**: Vector storage
- **Playwright**: Web automation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/chatgpt-plus-clone/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/chatgpt-plus-clone/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/chatgpt-plus-clone/wiki)

---

**Built with ❤️ using PyQt6, Ollama, and modern AI technologies**

*This project is not affiliated with OpenAI or ChatGPT. It's an independent implementation for educational and personal use.*