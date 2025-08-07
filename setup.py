#!/usr/bin/env python3
"""Setup script for ChatGPT+ Clone"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_pip_packages():
    """Install required pip packages"""
    packages = [
        "PyQt6>=6.4.0",
        "PyQt6-WebEngine>=6.4.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "sounddevice>=0.4.6",
        "numpy>=1.21.0",
        "psutil>=5.8.0",
        "keyboard>=0.13.5",
        "openai-whisper>=20231117",
        "TTS>=0.17.0",
        "requests>=2.31.0",
        "websockets>=11.0.3",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "aiofiles>=0.7.0",
        "watchdog>=3.0.0",
        "pyinstaller>=5.13.0",
        "pytest>=7.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "isort>=5.12.0",
    ]
    
    print("Installing Python packages...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True

def install_system_dependencies():
    """Install system-specific dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing Linux system dependencies...")
        packages = [
            "portaudio19-dev",
            "python3-dev",
            "build-essential",
            "libasound2-dev",
        ]
        
        # Try different package managers
        package_managers = [
            ("apt-get", "sudo apt-get update && sudo apt-get install -y"),
            ("yum", "sudo yum install -y"),
            ("dnf", "sudo dnf install -y"),
        ]
        
        for manager, install_cmd in package_managers:
            try:
                result = subprocess.run(f"which {manager}", shell=True, capture_output=True)
                if result.returncode == 0:
                    for package in packages:
                        if not run_command(f"{install_cmd} {package}", f"Installing {package} via {manager}"):
                            return False
                    return True
            except:
                continue
        
        print("Warning: Could not install system dependencies automatically")
        print("Please install the following packages manually:")
        for package in packages:
            print(f"  - {package}")
        return True
        
    elif system == "darwin":  # macOS
        print("Installing macOS system dependencies...")
        if not run_command("brew install portaudio", "Installing portaudio via Homebrew"):
            print("Warning: Could not install portaudio")
            print("Please install Homebrew and run: brew install portaudio")
            return False
        return True
        
    elif system == "windows":
        print("Windows detected - no additional system dependencies needed")
        return True
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "plugins",
        "logs",
        "downloads",
        "models",
        "voice",
        "gpu",
        "profiling",
        "remote_control",
        "build",
        "tests",
        "docs",
    ]
    
    print("Creating directories...")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    return True

def create_sample_plugin():
    """Create a sample plugin"""
    sample_plugin_content = '''from plugins.sdk import PluginBase

class Plugin(PluginBase):
    """Sample plugin for ChatGPT+ Clone"""
    
    async def on_load(self):
        """Called when plugin is loaded"""
        print(f"[{self.name}] Sample plugin loaded successfully")
        
        # Access AI manager API
        if self.api:
            print(f"[{self.name}] AI Manager API available")
    
    async def on_unload(self):
        """Called when plugin is unloaded"""
        print(f"[{self.name}] Sample plugin unloaded")
    
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
    
    plugin_path = "plugins/sample_plugin.py"
    try:
        with open(plugin_path, "w") as f:
            f.write(sample_plugin_content)
        print(f"✓ Created sample plugin: {plugin_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample plugin: {e}")
        return False

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Models and data
models/
*.bin
*.safetensors

# Audio files
*.wav
*.mp3
*.flac

# Temporary files
*.tmp
*.temp

# OS
.DS_Store
Thumbs.db

# Build artifacts
dist/
build/
*.exe
*.dmg
*.deb
*.rpm

# Configuration (optional)
# config.json
# .env
'''
    
    try:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore")
        return True
    except Exception as e:
        print(f"✗ Failed to create .gitignore: {e}")
        return False

def main():
    """Main setup function"""
    print("=== ChatGPT+ Clone Setup ===\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Install system dependencies
    if not install_system_dependencies():
        print("Warning: System dependencies installation failed")
    
    # Install Python packages
    if not install_pip_packages():
        print("✗ Failed to install Python packages")
        return False
    
    # Create directories
    if not create_directories():
        print("✗ Failed to create directories")
        return False
    
    # Create sample plugin
    if not create_sample_plugin():
        print("✗ Failed to create sample plugin")
        return False
    
    # Create .gitignore
    if not create_gitignore():
        print("✗ Failed to create .gitignore")
        return False
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Run: python3 test_imports.py")
    print("2. Run: python3 main.py")
    print("3. Check the documentation in docs/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)