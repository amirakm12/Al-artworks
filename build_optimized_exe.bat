@echo off
title Al-artworks AI Suite - Build Optimized Executable
color 0A

echo 🚀 Building Optimized Al-artworks AI Suite Executable...
echo 📦 Installing PyInstaller...

python -m pip install pyinstaller --quiet

echo 🔨 Building optimized executable...
pyinstaller --onefile --windowed --name "Al-artworks" --exclude-module matplotlib.tests --exclude-module numpy.random.tests --exclude-module PIL.tests "Al-artworks_Standalone_App.py"

if %errorLevel% equ 0 (
    copy "dist\Al-artworks.exe" "Al-artworks.exe" >nul 2>&1
    echo ✅ Optimized executable created: Al-artworks.exe
    rmdir /s /q "build" >nul 2>&1
    del "Al-artworks.spec" >nul 2>&1
) else (
    echo ❌ Build failed!
)

pause 