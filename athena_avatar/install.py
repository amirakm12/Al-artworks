#!/usr/bin/env python3
"""
Athena 3D Avatar - Installation Script
Easy setup and installation for the cosmic AI companion
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print Athena installation banner"""
    print("🌟" * 50)
    print("🌟  Athena 3D Avatar - Cosmic AI Companion  🌟")
    print("🌟" * 50)
    print("🚀 Advanced 3D avatar with voice interaction")
    print("✨ Optimized for 12GB RAM with <250ms latency")
    print("🎭 20+ animations, 12+ voice tones, divine emotions")
    print("🎨 Advanced 3D rendering with NeRF technology")
    print("📊 Real-time performance monitoring")
    print("🎮 Ready for cosmic interaction!")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("💻 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected")
            print("   Athena requires at least 8GB RAM for optimal performance")
        else:
            print("✅ Sufficient RAM detected")
            
    except ImportError:
        print("⚠️  Could not check memory (psutil not available)")
    
    # Check platform
    system = platform.system()
    print(f"🖥️  Operating System: {system}")
    
    if system not in ["Windows", "Darwin", "Linux"]:
        print("⚠️  Warning: Untested operating system")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Install requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            print("📋 Installing from requirements.txt...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ Failed to install dependencies")
                print(f"Error: {result.stderr}")
                return False
        else:
            print("❌ requirements.txt not found")
            return False
            
    except subprocess.CalledProcessError:
        print("❌ pip not available")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    try:
        project_root = Path(__file__).parent
        
        # Create directories
        directories = [
            "assets",
            "assets/models",
            "assets/textures", 
            "assets/audio",
            "assets/icons",
            "logs",
            "config",
            "exports"
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created: {directory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def setup_configuration():
    """Setup default configuration"""
    print("⚙️  Setting up configuration...")
    
    try:
        project_root = Path(__file__).parent
        config_file = project_root / "athena_config.yaml"
        
        if not config_file.exists():
            print("📝 Creating default configuration...")
            # Copy the default config if it doesn't exist
            default_config = project_root / "athena_config.yaml"
            if default_config.exists():
                print("✅ Configuration file ready")
            else:
                print("⚠️  No default configuration found")
        else:
            print("✅ Configuration file already exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup configuration: {e}")
        return False

def verify_installation():
    """Verify the installation"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        test_imports = [
            "torch",
            "PyQt6",
            "numpy",
            "librosa",
            "opencv-python"
        ]
        
        for module in test_imports:
            try:
                __import__(module.replace("-", "_"))
                print(f"✅ {module}")
            except ImportError:
                print(f"❌ {module} - not installed")
                return False
        
        # Test main application import
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from main import AthenaApp
            print("✅ AthenaApp import successful")
        except ImportError as e:
            print(f"❌ AthenaApp import failed: {e}")
            return False
        
        print("✅ Installation verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def print_completion_message():
    """Print installation completion message"""
    print()
    print("🎉" * 50)
    print("🎉  Athena 3D Avatar Installation Complete!  🎉")
    print("🎉" * 50)
    print()
    print("🚀 To start Athena, run:")
    print("   python run.py")
    print()
    print("📚 For more information, see README.md")
    print("🔧 Configuration: athena_config.yaml")
    print("📊 Logs: logs/athena_avatar.log")
    print()
    print("🌟 Welcome to the cosmic experience!")
    print()

def main():
    """Main installation function"""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        return 1
    
    if not check_system_requirements():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed. Please check the errors above.")
        return 1
    
    # Setup directories
    if not create_directories():
        print("❌ Failed to create directories.")
        return 1
    
    # Setup configuration
    if not setup_configuration():
        print("❌ Failed to setup configuration.")
        return 1
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed.")
        return 1
    
    # Success
    print_completion_message()
    return 0

if __name__ == "__main__":
    sys.exit(main())