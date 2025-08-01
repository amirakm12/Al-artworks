@echo off
title Al-artworks AI Suite - Build Standalone Executable
color 0A

echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                AL-ARTWORKS AI SUITE                          ║
echo    ║                    BUILD EXECUTABLE                           ║
echo    ║                                                              ║
echo    ║  🎨 Creating Standalone .exe File                           ║
echo    ║  🚀 Single File Distribution                                ║
echo    ║  ⚡ No Python Installation Required                          ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 Building Al-artworks AI Suite Executable...
echo 📦 Installing PyInstaller...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python not found!
    echo    Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python detected
echo 📦 Installing PyInstaller...
python -m pip install pyinstaller --quiet

if %errorLevel% neq 0 (
    echo ❌ Failed to install PyInstaller
    pause
    exit /b 1
)

echo ✅ PyInstaller installed
echo 🔨 Building executable...
echo.

REM Create the executable
pyinstaller --onefile --windowed --name "Al-artworks_AI_Suite" --icon=NONE --add-data "Al-artworks_Standalone_App.py;." "Al-artworks_Standalone_App.py"

if %errorLevel% neq 0 (
    echo ❌ Build failed!
    echo 💡 Check the error messages above
    pause
    exit /b 1
)

echo.
echo ✅ Build successful!
echo 📁 Executable created: dist\Al-artworks_AI_Suite.exe
echo.

REM Copy to current directory for easy access
copy "dist\Al-artworks_AI_Suite.exe" "Al-artworks_AI_Suite.exe" >nul 2>&1

echo 🎉 Al-artworks AI Suite executable created successfully!
echo 📁 Location: Al-artworks_AI_Suite.exe
echo.
echo 🚀 Users can now run the app by double-clicking the .exe file!
echo 💡 No Python installation required on target machines
echo.

REM Clean up build files
echo 🧹 Cleaning up build files...
rmdir /s /q "build" >nul 2>&1
rmdir /s /q "__pycache__" >nul 2>&1
del "Al-artworks_AI_Suite.spec" >nul 2>&1

echo ✅ Build complete!
echo.
echo 🎨 Your standalone executable is ready for distribution!
pause 