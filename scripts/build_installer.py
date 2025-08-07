#!/usr/bin/env python3
"""
Enhanced Build Script for ChatGPT+ Clone
Integrates PyInstaller with Inno Setup for comprehensive installer creation
"""

import os
import sys
import subprocess
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("BuildInstaller")

class BuildConfig:
    """Configuration for the build process"""
    
    def __init__(self):
        self.project_name = "ChatGPTPlusClone"
        self.version = "1.0.0"
        self.description = "Advanced AI Assistant with Voice Control and Plugin System"
        self.author = "ChatGPT+ Team"
        self.website = "https://github.com/your-repo/chatgpt-plus-clone"
        
        # Build paths
        self.project_root = Path(__file__).parent.parent
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.installer_dir = self.project_root / "installer"
        
        # PyInstaller settings
        self.pyinstaller_args = [
            "--onefile",
            "--windowed",  # No console window
            "--name", self.project_name,
            "--icon", str(self.project_root / "assets" / "icon.ico"),
            "--add-data", f"{self.project_root / 'config.json'};.",
            "--add-data", f"{self.project_root / 'plugins'};plugins",
            "--add-data", f"{self.project_root / 'sdk'};sdk",
            "--hidden-import", "torch",
            "--hidden-import", "transformers",
            "--hidden-import", "sounddevice",
            "--hidden-import", "whisper",
            "--hidden-import", "TTS",
            "--hidden-import", "PyQt6",
            "--hidden-import", "websockets",
            "--hidden-import", "fastapi",
            "--hidden-import", "uvicorn",
        ]
        
        # Inno Setup settings
        self.innosetup_script = self.installer_dir / "setup.iss"
        self.innosetup_output = self.installer_dir / "output"
        
        # Platform-specific settings
        self.platform = platform.system().lower()
        self.is_windows = self.platform == "windows"
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"

class BuildManager:
    """Manages the complete build process"""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        self.build_success = False
        
    def clean_build_dirs(self):
        """Clean previous build artifacts"""
        log.info("Cleaning build directories...")
        
        dirs_to_clean = [
            self.config.dist_dir,
            self.config.build_dir,
            self.config.innosetup_output
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                log.info(f"Cleaned {dir_path}")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        log.info("Checking build dependencies...")
        
        required_tools = []
        
        # Check Python
        if sys.version_info < (3, 8):
            log.error("Python 3.8+ required")
            return False
        
        # Check PyInstaller
        try:
            import PyInstaller
            log.info(f"PyInstaller {PyInstaller.__version__} found")
        except ImportError:
            log.error("PyInstaller not found. Install with: pip install pyinstaller")
            return False
        
        # Check Inno Setup (Windows only)
        if self.config.is_windows:
            innosetup_paths = [
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                r"C:\Program Files\Inno Setup 6\ISCC.exe"
            ]
            
            innosetup_found = False
            for path in innosetup_paths:
                if os.path.exists(path):
                    self.config.innosetup_exe = path
                    innosetup_found = True
                    log.info(f"Inno Setup found at {path}")
                    break
            
            if not innosetup_found:
                log.warning("Inno Setup not found. Install from: https://jrsoftware.org/isinfo.php")
                log.info("Continuing without installer creation...")
        
        return True
    
    def create_version_info(self):
        """Create version information for the executable"""
        version_info = {
            "version": self.config.version,
            "description": self.config.description,
            "author": self.config.author,
            "website": self.config.website,
            "build_date": subprocess.check_output(["date"]).decode().strip()
        }
        
        version_file = self.config.project_root / "version_info.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        log.info(f"Created version info: {version_file}")
    
    def run_pyinstaller(self) -> bool:
        """Run PyInstaller to create the executable"""
        log.info("Running PyInstaller...")
        
        try:
            # Prepare PyInstaller command
            cmd = ["pyinstaller"] + self.config.pyinstaller_args + [
                str(self.config.project_root / "main.py")
            ]
            
            log.info(f"PyInstaller command: {' '.join(cmd)}")
            
            # Run PyInstaller
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                log.info("PyInstaller build successful")
                return True
            else:
                log.error(f"PyInstaller failed: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Error running PyInstaller: {e}")
            return False
    
    def create_innosetup_script(self):
        """Create Inno Setup script for Windows installer"""
        if not self.config.is_windows:
            log.info("Skipping Inno Setup script creation (not Windows)")
            return
        
        log.info("Creating Inno Setup script...")
        
        # Ensure installer directory exists
        self.config.installer_dir.mkdir(exist_ok=True)
        
        # Create Inno Setup script
        script_content = f"""[Setup]
AppName={self.config.project_name}
AppVersion={self.config.version}
AppDescription={self.config.description}
AppPublisher={self.config.author}
AppPublisherURL={self.config.website}
DefaultDirName={{autopf}}\\{self.config.project_name}
DefaultGroupName={self.config.project_name}
OutputDir={self.config.innosetup_output}
OutputBaseFilename={self.config.project_name}Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
PrivilegesRequired=lowest
AllowNoIcons=yes
UninstallDisplayIcon={{app}}\\{self.config.project_name}.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Languages\\English.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{{cm:CreateQuickLaunchIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "{self.config.dist_dir}\\{self.config.project_name}.exe"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "{self.config.project_root}\\config.json"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "{self.config.project_root}\\plugins\\*"; DestDir: "{{app}}\\plugins"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{self.config.project_root}\\sdk\\*"; DestDir: "{{app}}\\sdk"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{self.config.project_root}\\docs\\*"; DestDir: "{{app}}\\docs"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{self.config.project_name}"; Filename: "{{app}}\\{self.config.project_name}.exe"
Name: "{{group}}\\Uninstall {self.config.project_name}"; Filename: "{{uninstallexe}}"
Name: "{{commondesktop}}\\{self.config.project_name}"; Filename: "{{app}}\\{self.config.project_name}.exe"; Tasks: desktopicon
Name: "{{userappdata}}\\Microsoft\\Internet Explorer\\Quick Launch\\{self.config.project_name}"; Filename: "{{app}}\\{self.config.project_name}.exe"; Tasks: quicklaunchicon

[Run]
Filename: "{{app}}\\{self.config.project_name}.exe"; Description: "Launch {self.config.project_name}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
end;
"""
        
        with open(self.config.innosetup_script, 'w') as f:
            f.write(script_content)
        
        log.info(f"Created Inno Setup script: {self.config.innosetup_script}")
    
    def run_innosetup(self) -> bool:
        """Run Inno Setup to create the installer"""
        if not self.config.is_windows or not hasattr(self.config, 'innosetup_exe'):
            log.info("Skipping Inno Setup (not available)")
            return True
        
        log.info("Running Inno Setup...")
        
        try:
            cmd = [self.config.innosetup_exe, str(self.config.innosetup_script)]
            log.info(f"Inno Setup command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                log.info("Inno Setup build successful")
                return True
            else:
                log.error(f"Inno Setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Error running Inno Setup: {e}")
            return False
    
    def create_linux_package(self):
        """Create Linux package (AppImage or DEB)"""
        if not self.config.is_linux:
            return
        
        log.info("Creating Linux package...")
        
        # Create AppImage structure
        appimage_dir = self.config.dist_dir / "AppDir"
        appimage_dir.mkdir(exist_ok=True)
        
        # Copy executable and dependencies
        shutil.copy2(
            self.config.dist_dir / f"{self.config.project_name}",
            appimage_dir / "AppRun"
        )
        
        # Create desktop file
        desktop_content = f"""[Desktop Entry]
Name={self.config.project_name}
Comment={self.config.description}
Exec=AppRun
Icon={self.config.project_name}
Type=Application
Categories=Utility;
"""
        
        with open(appimage_dir / f"{self.config.project_name}.desktop", 'w') as f:
            f.write(desktop_content)
        
        log.info("Linux package structure created")
    
    def create_macos_package(self):
        """Create macOS package (DMG)"""
        if not self.config.is_macos:
            return
        
        log.info("Creating macOS package...")
        
        # Create .app bundle structure
        app_bundle = self.config.dist_dir / f"{self.config.project_name}.app"
        contents_dir = app_bundle / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        for dir_path in [contents_dir, macos_dir, resources_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        shutil.copy2(
            self.config.dist_dir / f"{self.config.project_name}",
            macos_dir / f"{self.config.project_name}"
        )
        
        # Create Info.plist
        info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{self.config.project_name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.chatgptplus.{self.config.project_name.lower()}</string>
    <key>CFBundleName</key>
    <string>{self.config.project_name}</string>
    <key>CFBundleVersion</key>
    <string>{self.config.version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{self.config.version}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
"""
        
        with open(contents_dir / "Info.plist", 'w') as f:
            f.write(info_plist)
        
        log.info("macOS package structure created")
    
    def build(self) -> bool:
        """Run the complete build process"""
        log.info("Starting build process...")
        
        try:
            # Step 1: Clean previous builds
            self.clean_build_dirs()
            
            # Step 2: Check dependencies
            if not self.check_dependencies():
                return False
            
            # Step 3: Create version info
            self.create_version_info()
            
            # Step 4: Run PyInstaller
            if not self.run_pyinstaller():
                return False
            
            # Step 5: Create platform-specific packages
            if self.config.is_windows:
                self.create_innosetup_script()
                if not self.run_innosetup():
                    log.warning("Inno Setup failed, but PyInstaller build succeeded")
            elif self.config.is_linux:
                self.create_linux_package()
            elif self.config.is_macos:
                self.create_macos_package()
            
            # Step 6: Verify build
            exe_path = self.config.dist_dir / f"{self.config.project_name}.exe"
            if not exe_path.exists():
                exe_path = self.config.dist_dir / f"{self.config.project_name}"
            
            if exe_path.exists():
                log.info(f"Build successful! Executable: {exe_path}")
                self.build_success = True
                return True
            else:
                log.error("Build failed: executable not found")
                return False
                
        except Exception as e:
            log.error(f"Build process failed: {e}")
            return False
    
    def get_build_info(self) -> Dict[str, Any]:
        """Get information about the build"""
        return {
            "success": self.build_success,
            "platform": self.config.platform,
            "version": self.config.version,
            "project_name": self.config.project_name,
            "dist_dir": str(self.config.dist_dir),
            "build_dir": str(self.config.build_dir),
            "installer_dir": str(self.config.installer_dir)
        }

def main():
    """Main build function"""
    log.info("=== ChatGPT+ Clone Build System ===")
    
    # Create build configuration
    config = BuildConfig()
    
    # Create build manager
    builder = BuildManager(config)
    
    # Run build
    success = builder.build()
    
    # Report results
    build_info = builder.get_build_info()
    log.info(f"Build completed: {success}")
    log.info(f"Build info: {build_info}")
    
    if success:
        log.info("🎉 Build successful! Check the dist/ directory for the executable.")
        if config.is_windows and hasattr(config, 'innosetup_exe'):
            log.info("📦 Installer created in installer/output/ directory.")
    else:
        log.error("❌ Build failed! Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()