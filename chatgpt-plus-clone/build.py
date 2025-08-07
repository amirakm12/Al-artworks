"""
Cross-Platform Build System for ChatGPT+ Clone
Creates standalone executables and installers for Windows, Linux, and macOS
"""

import os
import sys
import shutil
import subprocess
import platform
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Build configuration
APP_NAME = "ChatGPTPlus"
ENTRY_SCRIPT = "main.py"
DIST_DIR = "dist"
BUILD_DIR = "build"
ASSETS_DIR = "assets"
ICON_FILE = "assets/icon.ico"  # Windows
ICON_FILE_MAC = "assets/icon.icns"  # macOS
VERSION_FILE = "version.json"

# Platform-specific settings
PLATFORM_CONFIGS = {
    "Windows": {
        "icon": ICON_FILE,
        "hidden_imports": [
            "PyQt6",
            "PyQt6.QtWebEngineWidgets",
            "PyQt6.QtOpenGLWidgets",
            "torch",
            "whisper",
            "TTS",
            "keyboard",
            "sounddevice",
            "psutil"
        ],
        "exclude_modules": [],
        "extra_files": [
            ("config.json", "."),
            ("requirements.txt", "."),
            ("plugins", "plugins"),
            ("ui", "ui"),
            ("memory", "memory"),
            ("docs", "docs"),
            ("assets", "assets"),
            ("security", "security"),
            ("tests", "tests")
        ]
    },
    "Linux": {
        "icon": None,  # Linux doesn't use .ico files
        "hidden_imports": [
            "PyQt6",
            "PyQt6.QtWebEngineWidgets",
            "PyQt6.QtOpenGLWidgets",
            "torch",
            "whisper",
            "TTS",
            "psutil"
        ],
        "exclude_modules": ["keyboard"],  # keyboard doesn't work on Linux
        "extra_files": [
            ("config.json", "."),
            ("requirements.txt", "."),
            ("plugins", "plugins"),
            ("ui", "ui"),
            ("memory", "memory"),
            ("docs", "docs"),
            ("assets", "assets"),
            ("security", "security"),
            ("tests", "tests")
        ]
    },
    "Darwin": {  # macOS
        "icon": ICON_FILE_MAC,
        "hidden_imports": [
            "PyQt6",
            "PyQt6.QtWebEngineWidgets",
            "PyQt6.QtOpenGLWidgets",
            "torch",
            "whisper",
            "TTS",
            "psutil"
        ],
        "exclude_modules": ["keyboard"],
        "extra_files": [
            ("config.json", "."),
            ("requirements.txt", "."),
            ("plugins", "plugins"),
            ("ui", "ui"),
            ("memory", "memory"),
            ("docs", "docs"),
            ("assets", "assets"),
            ("security", "security"),
            ("tests", "tests")
        ]
    }
}

class BuildSystem:
    """Cross-platform build system for ChatGPT+ Clone"""
    
    def __init__(self):
        self.platform = platform.system()
        self.config = PLATFORM_CONFIGS.get(self.platform, PLATFORM_CONFIGS["Windows"])
        self.build_dir = Path(BUILD_DIR)
        self.dist_dir = Path(DIST_DIR)
        self.version = self.get_version()
        
        print(f"[Build] Platform: {self.platform}")
        print(f"[Build] Version: {self.version}")
    
    def get_version(self) -> str:
        """Get current version from version.json"""
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f:
                    version_data = json.load(f)
                return version_data.get('version', '1.0.0')
        except Exception as e:
            print(f"[Build] Warning: Could not read version file: {e}")
        
        return '1.0.0'
    
    def check_dependencies(self) -> bool:
        """Check if required build dependencies are available"""
        print("[Build] Checking dependencies...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            print("[Build] Error: Python 3.10+ required")
            return False
        
        # Check PyInstaller
        try:
            import PyInstaller
            print(f"[Build] PyInstaller version: {PyInstaller.__version__}")
        except ImportError:
            print("[Build] Error: PyInstaller not installed")
            print("[Build] Install with: pip install pyinstaller")
            return False
        
        # Check if entry script exists
        if not os.path.exists(ENTRY_SCRIPT):
            print(f"[Build] Error: Entry script {ENTRY_SCRIPT} not found")
            return False
        
        # Check if icon exists (if specified)
        if self.config["icon"] and not os.path.exists(self.config["icon"]):
            print(f"[Build] Warning: Icon file {self.config['icon']} not found")
        
        print("[Build] Dependencies check passed")
        return True
    
    def clean(self):
        """Clean previous builds"""
        print("[Build] Cleaning previous builds...")
        
        # Remove build and dist directories
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"[Build] Removed {dir_path}")
        
        # Remove .spec files
        spec_file = Path(f"{APP_NAME}.spec")
        if spec_file.exists():
            spec_file.unlink()
            print(f"[Build] Removed {spec_file}")
    
    def create_spec_file(self) -> str:
        """Create PyInstaller spec file"""
        print("[Build] Creating spec file...")
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{ENTRY_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas={self.config["extra_files"]},
    hiddenimports={self.config["hidden_imports"]},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={self.config["exclude_modules"]},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon={repr(self.config["icon"]) if self.config["icon"] else None},
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{APP_NAME}',
)
'''
        
        spec_file = f"{APP_NAME}.spec"
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        print(f"[Build] Created spec file: {spec_file}")
        return spec_file
    
    def build(self) -> bool:
        """Build the application"""
        print(f"[Build] Building {APP_NAME} for {self.platform}...")
        
        try:
            # Create spec file
            spec_file = self.create_spec_file()
            
            # Run PyInstaller
            cmd = [
                "pyinstaller",
                "--clean",
                "--noconfirm",
                spec_file
            ]
            
            print(f"[Build] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("[Build] Build completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[Build] Build failed: {e}")
            print(f"[Build] Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"[Build] Build failed: {e}")
            return False
    
    def create_installer(self) -> Optional[str]:
        """Create installer package"""
        print("[Build] Creating installer package...")
        
        try:
            # Check if dist directory exists
            dist_path = self.dist_dir / APP_NAME
            if not dist_path.exists():
                print(f"[Build] Error: Build output not found at {dist_path}")
                return None
            
            # Create installer based on platform
            if self.platform == "Windows":
                return self.create_windows_installer()
            elif self.platform == "Darwin":
                return self.create_macos_installer()
            else:
                return self.create_linux_installer()
                
        except Exception as e:
            print(f"[Build] Installer creation failed: {e}")
            return None
    
    def create_windows_installer(self) -> str:
        """Create Windows installer"""
        print("[Build] Creating Windows installer...")
        
        # Create installer directory
        installer_dir = Path("installer")
        installer_dir.mkdir(exist_ok=True)
        
        # Copy built application
        app_source = self.dist_dir / APP_NAME
        app_dest = installer_dir / APP_NAME
        
        if app_dest.exists():
            shutil.rmtree(app_dest)
        shutil.copytree(app_source, app_dest)
        
        # Create batch file for easy launch
        batch_content = f'''@echo off
echo Starting {APP_NAME}...
cd /d "%~dp0"
cd {APP_NAME}
start {APP_NAME}.exe
'''
        
        batch_file = installer_dir / f"start_{APP_NAME}.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        # Create README
        readme_content = f'''# {APP_NAME} Installer

## Installation
1. Extract this folder to your desired location
2. Run `start_{APP_NAME}.bat` to launch the application
3. Or navigate to the {APP_NAME} folder and run {APP_NAME}.exe

## System Requirements
- Windows 10/11
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- Internet connection for first-time setup

## Troubleshooting
- If the application doesn't start, check Windows Defender settings
- Ensure you have the latest Visual C++ Redistributables
- Run as Administrator if you encounter permission issues

Version: {self.version}
Build Date: {platform.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
        
        readme_file = installer_dir / "README.txt"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create zip installer
        installer_zip = f"{APP_NAME}_v{self.version}_Windows.zip"
        shutil.make_archive(installer_zip[:-4], 'zip', installer_dir)
        
        print(f"[Build] Windows installer created: {installer_zip}")
        return installer_zip
    
    def create_macos_installer(self) -> str:
        """Create macOS installer"""
        print("[Build] Creating macOS installer...")
        
        # Create .app bundle
        app_bundle = f"{APP_NAME}.app"
        app_source = self.dist_dir / APP_NAME
        app_dest = Path(app_bundle)
        
        if app_dest.exists():
            shutil.rmtree(app_dest)
        shutil.copytree(app_source, app_dest)
        
        # Create DMG installer
        dmg_name = f"{APP_NAME}_v{self.version}_macOS.dmg"
        
        # Note: This is a simplified DMG creation
        # For production, you'd want to use create-dmg or similar tools
        print(f"[Build] macOS app bundle created: {app_bundle}")
        print(f"[Build] Note: DMG creation requires additional tools")
        
        return app_bundle
    
    def create_linux_installer(self) -> str:
        """Create Linux installer"""
        print("[Build] Creating Linux installer...")
        
        # Create AppImage or tar.gz
        app_source = self.dist_dir / APP_NAME
        
        # Create tar.gz package
        tar_name = f"{APP_NAME}_v{self.version}_Linux.tar.gz"
        
        with tarfile.open(tar_name, "w:gz") as tar:
            tar.add(app_source, arcname=APP_NAME)
        
        print(f"[Build] Linux package created: {tar_name}")
        return tar_name
    
    def run_tests(self) -> bool:
        """Run tests before building"""
        print("[Build] Running tests...")
        
        try:
            # Run basic tests
            test_cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("[Build] Tests passed")
                return True
            else:
                print(f"[Build] Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[Build] Test execution failed: {e}")
            return False
    
    def validate_build(self) -> bool:
        """Validate the built application"""
        print("[Build] Validating build...")
        
        try:
            # Check if executable exists
            if self.platform == "Windows":
                exe_path = self.dist_dir / APP_NAME / f"{APP_NAME}.exe"
            else:
                exe_path = self.dist_dir / APP_NAME / APP_NAME
            
            if not exe_path.exists():
                print(f"[Build] Error: Executable not found at {exe_path}")
                return False
            
            # Check file size (should be reasonable)
            file_size = exe_path.stat().st_size
            if file_size < 1000000:  # Less than 1MB
                print(f"[Build] Warning: Executable seems too small ({file_size} bytes)")
            
            print(f"[Build] Build validation passed (size: {file_size} bytes)")
            return True
            
        except Exception as e:
            print(f"[Build] Build validation failed: {e}")
            return False
    
    def build_all(self) -> bool:
        """Complete build process"""
        print("=" * 50)
        print(f"[Build] Starting build process for {APP_NAME}")
        print("=" * 50)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Clean previous builds
        self.clean()
        
        # Step 3: Run tests (optional)
        if not self.run_tests():
            print("[Build] Warning: Tests failed, continuing anyway...")
        
        # Step 4: Build application
        if not self.build():
            return False
        
        # Step 5: Validate build
        if not self.validate_build():
            return False
        
        # Step 6: Create installer
        installer_path = self.create_installer()
        if installer_path:
            print(f"[Build] Installer created: {installer_path}")
        
        print("=" * 50)
        print("[Build] Build process completed successfully!")
        print("=" * 50)
        
        return True

def main():
    """Main build function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ChatGPT+ Clone")
    parser.add_argument("--clean", action="store_true", help="Clean only")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--validate", action="store_true", help="Validate existing build")
    parser.add_argument("--installer", action="store_true", help="Create installer only")
    
    args = parser.parse_args()
    
    build_system = BuildSystem()
    
    if args.clean:
        build_system.clean()
        print("[Build] Clean completed")
        return
    
    if args.test:
        success = build_system.run_tests()
        sys.exit(0 if success else 1)
    
    if args.validate:
        success = build_system.validate_build()
        sys.exit(0 if success else 1)
    
    if args.installer:
        installer_path = build_system.create_installer()
        if installer_path:
            print(f"[Build] Installer created: {installer_path}")
            sys.exit(0)
        else:
            print("[Build] Installer creation failed")
            sys.exit(1)
    
    # Full build process
    success = build_system.build_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()