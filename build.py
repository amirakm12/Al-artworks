import subprocess
import os
import sys
import logging
from pathlib import Path

log = logging.getLogger("BuildScript")

def build_exe():
    """Build executable using PyInstaller"""
    
    # Define the PyInstaller command
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--name=ChatGPTPlus",
        "--clean",
        "--noconfirm",
        
        # Add data files
        "--add-data", "config.json;.",
        "--add-data", "version.json;.",
        "--add-data", "requirements.txt;.",
        "--add-data", "plugins;plugins",
        "--add-data", "ui;ui",
        "--add-data", "gpu;gpu",
        "--add-data", "voice;voice",
        "--add-data", "build;build",
        "--add-data", "docs;docs",
        "--add-data", "tests;tests",
        
        # Add hidden imports
        "--hidden-import", "PyQt6",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "PyQt6.QtGui",
        "--hidden-import", "PyQt6.QtWebEngineWidgets",
        "--hidden-import", "PyQt6.QtWebEngineCore",
        "--hidden-import", "sounddevice",
        "--hidden-import", "numpy",
        "--hidden-import", "torch",
        "--hidden-import", "transformers",
        "--hidden-import", "whisper",
        "--hidden-import", "TTS",
        "--hidden-import", "requests",
        "--hidden-import", "psutil",
        "--hidden-import", "asyncio",
        "--hidden-import", "logging",
        "--hidden-import", "json",
        "--hidden-import", "queue",
        "--hidden-import", "threading",
        "--hidden-import", "time",
        "--hidden-import", "pathlib",
        "--hidden-import", "fastapi",
        "--hidden-import", "uvicorn",
        "--hidden-import", "websockets",
        
        # Main script
        "main.py"
    ]
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        log.error("main.py not found in current directory")
        return False
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        log.info(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        log.error("PyInstaller not installed. Install with: pip install pyinstaller")
        return False
    
    log.info("Starting PyInstaller build...")
    log.info(f"Command: {' '.join(pyinstaller_cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(pyinstaller_cmd, check=True, capture_output=True, text=True)
        
        log.info("✓ Build completed successfully")
        log.info(f"Build output: {result.stdout}")
        
        # Check if executable was created
        exe_path = "dist/ChatGPTPlus.exe"
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024**2)  # MB
            log.info(f"✓ Executable created: {exe_path}")
            log.info(f"  File size: {file_size:.1f} MB")
            return True
        else:
            log.error(f"✗ Executable not found: {exe_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        log.error(f"✗ Build failed: {e}")
        log.error(f"Build error output: {e.stderr}")
        return False
    except Exception as e:
        log.error(f"✗ Build error: {e}")
        return False

def clean_build():
    """Clean build artifacts"""
    log.info("Cleaning build artifacts...")
    
    # Remove build directories
    for directory in ["build", "dist", "__pycache__"]:
        if os.path.exists(directory):
            try:
                import shutil
                shutil.rmtree(directory)
                log.info(f"Removed {directory}/")
            except Exception as e:
                log.warning(f"Failed to remove {directory}/: {e}")
    
    # Remove spec files
    for spec_file in Path(".").glob("*.spec"):
        try:
            spec_file.unlink()
            log.info(f"Removed {spec_file}")
        except Exception as e:
            log.warning(f"Failed to remove {spec_file}: {e}")

def create_installer_package():
    """Create installer package (zip for now)"""
    import zipfile
    
    exe_path = "dist/ChatGPTPlus.exe"
    if not os.path.exists(exe_path):
        log.error("Executable not found, cannot create installer")
        return False
    
    try:
        # Create zip installer
        installer_name = "ChatGPTPlus_installer.zip"
        with zipfile.ZipFile(installer_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add executable
            zipf.write(exe_path, "ChatGPTPlus.exe")
            
            # Add additional files
            additional_files = [
                "config.json",
                "version.json", 
                "README.md",
                "requirements.txt"
            ]
            
            for file in additional_files:
                if os.path.exists(file):
                    zipf.write(file, file)
                    log.info(f"Added {file} to installer")
        
        installer_size = os.path.getsize(installer_name) / (1024**2)  # MB
        log.info(f"✓ Installer created: {installer_name}")
        log.info(f"  Installer size: {installer_size:.1f} MB")
        return True
        
    except Exception as e:
        log.error(f"✗ Failed to create installer: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    log.info("Checking build dependencies...")
    
    required_packages = [
        "pyinstaller",
        "torch", 
        "transformers",
        "PyQt6",
        "sounddevice",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            log.info(f"✓ {package} is available")
        except ImportError:
            log.error(f"✗ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        log.error(f"Missing packages: {missing_packages}")
        log.error("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    log.info("✓ All dependencies are available")
    return True

def main():
    """Main build function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log.info("=== Starting Build Process ===")
    
    # Check dependencies
    if not check_dependencies():
        log.error("Build failed: Missing dependencies")
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build executable
    if not build_exe():
        log.error("Build failed: PyInstaller build failed")
        sys.exit(1)
    
    # Create installer
    create_installer_package()
    
    log.info("=== Build Process Completed Successfully! ===")
    log.info("Executable: dist/ChatGPTPlus.exe")
    log.info("Installer: ChatGPTPlus_installer.zip")

if __name__ == "__main__":
    main()