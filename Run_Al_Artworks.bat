@echo off
title Al-artworks AI Suite - Standalone App
color 0A

echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                AL-ARTWORKS AI SUITE                          ║
echo    ║                    STANDALONE APP                             ║
echo    ║                                                              ║
echo    ║  🎨 AI-Powered Art Generation                               ║
echo    ║  🚀 No Installation Required                                 ║
echo    ║  ⚡ Ready to Create Amazing Artwork                         ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 Starting Al-artworks AI Suite...
echo 📦 Installing dependencies if needed...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python not found!
    echo    Please install Python 3.8+ from https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python detected
echo 🎨 Launching AI Art Generator...
echo.

REM Run the standalone application
python "Al-artworks_Standalone_App.py"

echo.
echo 👋 Thanks for using Al-artworks AI Suite!
echo 🎨 Keep creating amazing artwork!
pause 