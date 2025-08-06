#!/usr/bin/env python3
"""
Build Script for Athena 3D Avatar
Package and distribute for Windows, macOS, Android, iOS, XR
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Any

class AthenaBuilder:
    """Builder for Athena 3D Avatar application"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.assets_dir = self.project_root / "assets"
        
        # Platform detection
        self.platform = platform.system().lower()
        self.architecture = platform.machine()
        
        # Build configurations
        self.build_configs = {
            'windows': {
                'target': 'win',
                'ext': '.exe',
                'icon': 'assets/athena_icon.ico',
                'additional_files': ['assets/*']
            },
            'macos': {
                'target': 'mac',
                'ext': '.app',
                'icon': 'assets/athena_icon.icns',
                'additional_files': ['assets/*']
            },
            'linux': {
                'target': 'linux',
                'ext': '',
                'icon': 'assets/athena_icon.png',
                'additional_files': ['assets/*']
            }
        }
    
    def clean_build(self):
        """Clean build directories"""
        try:
            print("🧹 Cleaning build directories...")
            
            # Remove dist directory
            if self.dist_dir.exists():
                shutil.rmtree(self.dist_dir)
                print(f"   Removed {self.dist_dir}")
            
            # Remove build directory
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
                print(f"   Removed {self.build_dir}")
            
            # Remove spec files
            for spec_file in self.project_root.glob("*.spec"):
                spec_file.unlink()
                print(f"   Removed {spec_file}")
            
            print("✅ Build directories cleaned")
            
        except Exception as e:
            print(f"❌ Failed to clean build: {e}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        try:
            print("📦 Installing dependencies...")
            
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.project_root / "requirements.txt")
            ], check=True)
            
            # Install PyInstaller
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pyinstaller"
            ], check=True)
            
            print("✅ Dependencies installed")
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            raise
    
    def create_assets_directory(self):
        """Create assets directory with placeholder files"""
        try:
            print("📁 Creating assets directory...")
            
            # Create assets directory
            self.assets_dir.mkdir(exist_ok=True)
            
            # Create placeholder files
            placeholder_files = [
                "athena_icon.ico",
                "athena_icon.icns", 
                "athena_icon.png",
                "athena_head.obj",
                "athena_body.obj",
                "athena_arms.obj",
                "athena_robes.obj",
                "athena_wreath.obj",
                "athena_veins.obj",
                "marble_robes_diffuse.png",
                "marble_robes_normal.png",
                "golden_wreath_diffuse.png",
                "metallic_arms_diffuse.png",
                "holographic_veins_diffuse.png",
                "marble_robes_pbr.mat",
                "golden_wreath_pbr.mat",
                "metallic_arms_pbr.mat",
                "holographic_veins_pbr.mat"
            ]
            
            for file_name in placeholder_files:
                file_path = self.assets_dir / file_name
                if not file_path.exists():
                    file_path.touch()
                    print(f"   Created {file_name}")
            
            print("✅ Assets directory created")
            
        except Exception as e:
            print(f"❌ Failed to create assets directory: {e}")
    
    def build_windows(self):
        """Build for Windows"""
        try:
            print("🪟 Building for Windows...")
            
            config = self.build_configs['windows']
            
            # PyInstaller command for Windows
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--icon={config['icon']}",
                "--add-data", f"assets{os.pathsep}assets",
                "--name", "Athena3DAvatar",
                "--distpath", str(self.dist_dir / "windows"),
                "--workpath", str(self.build_dir / "windows"),
                "main.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("✅ Windows build completed")
            
        except Exception as e:
            print(f"❌ Failed to build for Windows: {e}")
    
    def build_macos(self):
        """Build for macOS"""
        try:
            print("🍎 Building for macOS...")
            
            config = self.build_configs['macos']
            
            # PyInstaller command for macOS
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--icon={config['icon']}",
                "--add-data", f"assets{os.pathsep}assets",
                "--name", "Athena3DAvatar",
                "--distpath", str(self.dist_dir / "macos"),
                "--workpath", str(self.build_dir / "macos"),
                "main.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("✅ macOS build completed")
            
        except Exception as e:
            print(f"❌ Failed to build for macOS: {e}")
    
    def build_linux(self):
        """Build for Linux"""
        try:
            print("🐧 Building for Linux...")
            
            config = self.build_configs['linux']
            
            # PyInstaller command for Linux
            cmd = [
                "pyinstaller",
                "--onefile",
                f"--icon={config['icon']}",
                "--add-data", f"assets{os.pathsep}assets",
                "--name", "Athena3DAvatar",
                "--distpath", str(self.dist_dir / "linux"),
                "--workpath", str(self.build_dir / "linux"),
                "main.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("✅ Linux build completed")
            
        except Exception as e:
            print(f"❌ Failed to build for Linux: {e}")
    
    def build_android(self):
        """Build for Android (placeholder)"""
        try:
            print("🤖 Building for Android (placeholder)...")
            
            # Create Android build directory
            android_dir = self.dist_dir / "android"
            android_dir.mkdir(exist_ok=True)
            
            # Create placeholder APK
            apk_path = android_dir / "Athena3DAvatar.apk"
            apk_path.touch()
            
            print("✅ Android build placeholder created")
            
        except Exception as e:
            print(f"❌ Failed to build for Android: {e}")
    
    def build_ios(self):
        """Build for iOS (placeholder)"""
        try:
            print("📱 Building for iOS (placeholder)...")
            
            # Create iOS build directory
            ios_dir = self.dist_dir / "ios"
            ios_dir.mkdir(exist_ok=True)
            
            # Create placeholder IPA
            ipa_path = ios_dir / "Athena3DAvatar.ipa"
            ipa_path.touch()
            
            print("✅ iOS build placeholder created")
            
        except Exception as e:
            print(f"❌ Failed to build for iOS: {e}")
    
    def build_xr(self):
        """Build for XR platforms (placeholder)"""
        try:
            print("🥽 Building for XR platforms (placeholder)...")
            
            # Create XR build directory
            xr_dir = self.dist_dir / "xr"
            xr_dir.mkdir(exist_ok=True)
            
            # Create placeholder files for different XR platforms
            xr_platforms = ["oculus", "vive", "hololens", "magicleap"]
            
            for platform in xr_platforms:
                platform_dir = xr_dir / platform
                platform_dir.mkdir(exist_ok=True)
                
                # Create placeholder app
                app_path = platform_dir / f"Athena3DAvatar_{platform}.app"
                app_path.touch()
            
            print("✅ XR build placeholders created")
            
        except Exception as e:
            print(f"❌ Failed to build for XR: {e}")
    
    def build_all(self):
        """Build for all platforms"""
        try:
            print("🚀 Starting build for all platforms...")
            
            # Clean previous builds
            self.clean_build()
            
            # Install dependencies
            self.install_dependencies()
            
            # Create assets directory
            self.create_assets_directory()
            
            # Build for current platform
            if self.platform == "windows":
                self.build_windows()
            elif self.platform == "darwin":
                self.build_macos()
            elif self.platform == "linux":
                self.build_linux()
            
            # Build placeholders for other platforms
            self.build_android()
            self.build_ios()
            self.build_xr()
            
            print("✅ All builds completed!")
            
        except Exception as e:
            print(f"❌ Build failed: {e}")
            raise
    
    def create_installer(self):
        """Create installer packages"""
        try:
            print("📦 Creating installers...")
            
            # Create installers directory
            installers_dir = self.dist_dir / "installers"
            installers_dir.mkdir(exist_ok=True)
            
            # Create platform-specific installers
            if self.platform == "windows":
                self._create_windows_installer(installers_dir)
            elif self.platform == "darwin":
                self._create_macos_installer(installers_dir)
            elif self.platform == "linux":
                self._create_linux_installer(installers_dir)
            
            print("✅ Installers created")
            
        except Exception as e:
            print(f"❌ Failed to create installers: {e}")
    
    def _create_windows_installer(self, installers_dir: Path):
        """Create Windows installer"""
        try:
            # Create NSIS script
            nsis_script = installers_dir / "installer.nsi"
            
            with open(nsis_script, 'w') as f:
                f.write("""
!include "MUI2.nsh"

Name "Athena 3D Avatar"
OutFile "Athena3DAvatar_Setup.exe"
InstallDir "$PROGRAMFILES\\Athena3DAvatar"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath "$INSTDIR"
    File "windows\\Athena3DAvatar.exe"
    File /r "assets"
    
    CreateDirectory "$SMPROGRAMS\\Athena3DAvatar"
    CreateShortCut "$SMPROGRAMS\\Athena3DAvatar\\Athena3DAvatar.lnk" "$INSTDIR\\Athena3DAvatar.exe"
    CreateShortCut "$DESKTOP\\Athena3DAvatar.lnk" "$INSTDIR\\Athena3DAvatar.exe"
    
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\Athena3DAvatar.exe"
    RMDir /r "$INSTDIR\\assets"
    Delete "$INSTDIR\\Uninstall.exe"
    RMDir "$INSTDIR"
    
    Delete "$SMPROGRAMS\\Athena3DAvatar\\Athena3DAvatar.lnk"
    RMDir "$SMPROGRAMS\\Athena3DAvatar"
    Delete "$DESKTOP\\Athena3DAvatar.lnk"
SectionEnd
                """)
            
            print("   Created Windows installer script")
            
        except Exception as e:
            print(f"   Failed to create Windows installer: {e}")
    
    def _create_macos_installer(self, installers_dir: Path):
        """Create macOS installer"""
        try:
            # Create DMG script
            dmg_script = installers_dir / "create_dmg.sh"
            
            with open(dmg_script, 'w') as f:
                f.write("""
#!/bin/bash
# Create macOS DMG installer

DMG_NAME="Athena3DAvatar.dmg"
APP_NAME="Athena3DAvatar.app"
VOLUME_NAME="Athena 3D Avatar"

# Create DMG
hdiutil create -volname "$VOLUME_NAME" -srcfolder "macos/$APP_NAME" -ov -format UDZO "$DMG_NAME"

echo "Created $DMG_NAME"
                """)
            
            # Make script executable
            os.chmod(dmg_script, 0o755)
            
            print("   Created macOS installer script")
            
        except Exception as e:
            print(f"   Failed to create macOS installer: {e}")
    
    def _create_linux_installer(self, installers_dir: Path):
        """Create Linux installer"""
        try:
            # Create AppImage script
            appimage_script = installers_dir / "create_appimage.sh"
            
            with open(appimage_script, 'w') as f:
                f.write("""
#!/bin/bash
# Create Linux AppImage

APP_NAME="Athena3DAvatar"
APP_DIR="$APP_NAME.AppDir"

# Create AppDir structure
mkdir -p "$APP_DIR/usr/bin"
mkdir -p "$APP_DIR/usr/share/applications"
mkdir -p "$APP_DIR/usr/share/icons/hicolor/256x256/apps"

# Copy application
cp "linux/Athena3DAvatar" "$APP_DIR/usr/bin/"
cp -r "assets" "$APP_DIR/usr/bin/"

# Create desktop file
cat > "$APP_DIR/usr/share/applications/$APP_NAME.desktop" << EOF
[Desktop Entry]
Name=Athena 3D Avatar
Comment=Cosmic AI Companion
Exec=Athena3DAvatar
Icon=athena_icon
Type=Application
Categories=Graphics;3DGraphics;
EOF

# Create AppImage
appimagetool "$APP_DIR" "$APP_NAME.AppImage"

echo "Created $APP_NAME.AppImage"
                """)
            
            # Make script executable
            os.chmod(appimage_script, 0o755)
            
            print("   Created Linux installer script")
            
        except Exception as e:
            print(f"   Failed to create Linux installer: {e}")
    
    def create_release_package(self):
        """Create release package"""
        try:
            print("📦 Creating release package...")
            
            # Create release directory
            release_dir = self.dist_dir / "release"
            release_dir.mkdir(exist_ok=True)
            
            # Copy all builds to release
            for platform_dir in self.dist_dir.iterdir():
                if platform_dir.is_dir() and platform_dir.name != "release":
                    shutil.copytree(platform_dir, release_dir / platform_dir.name)
            
            # Create README for release
            readme_path = release_dir / "README.txt"
            with open(readme_path, 'w') as f:
                f.write("""
Athena 3D Avatar - Release Package

This package contains builds for multiple platforms:

Windows:
- Athena3DAvatar.exe (Standalone executable)
- Athena3DAvatar_Setup.exe (Installer)

macOS:
- Athena3DAvatar.app (Application bundle)
- Athena3DAvatar.dmg (Disk image)

Linux:
- Athena3DAvatar (Standalone executable)
- Athena3DAvatar.AppImage (Portable AppImage)

System Requirements:
- 12GB RAM minimum
- OpenGL 4.0+ compatible graphics card
- 2GB free disk space

Installation:
1. Extract the appropriate package for your platform
2. Run the executable or installer
3. Follow the on-screen instructions

For more information, visit: https://github.com/athena-avatar
                """)
            
            print("✅ Release package created")
            
        except Exception as e:
            print(f"❌ Failed to create release package: {e}")

def main():
    """Main build function"""
    try:
        print("🌟 Athena 3D Avatar - Build Script")
        print("=" * 50)
        
        builder = AthenaBuilder()
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "clean":
                builder.clean_build()
            elif command == "deps":
                builder.install_dependencies()
            elif command == "assets":
                builder.create_assets_directory()
            elif command == "windows":
                builder.build_windows()
            elif command == "macos":
                builder.build_macos()
            elif command == "linux":
                builder.build_linux()
            elif command == "android":
                builder.build_android()
            elif command == "ios":
                builder.build_ios()
            elif command == "xr":
                builder.build_xr()
            elif command == "installer":
                builder.create_installer()
            elif command == "release":
                builder.create_release_package()
            else:
                print(f"Unknown command: {command}")
                print("Available commands: clean, deps, assets, windows, macos, linux, android, ios, xr, installer, release")
        else:
            # Build all by default
            builder.build_all()
            builder.create_installer()
            builder.create_release_package()
        
        print("\n🎉 Build completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()