@echo off
title Al-artworks Simple AI Suite - No CMake Required
color 0A

echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                AL-ARTWORKS SIMPLE AI SUITE                   ║
echo    ║                    NO CMAKE REQUIRED                          ║
echo    ║                                                              ║
echo    ║  🎨 AI-Powered Art Generation                               ║
echo    ║  🚀 No CMake, No Complex Dependencies                       ║
echo    ║  ⚡ Pure Python Art Creation                                 ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🚀 Starting Al-artworks Simple AI Suite...
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
echo 🎨 Launching Simple AI Art Generator...
echo.

REM Run the simple application
python "Al-artworks_Simple_App.py"

echo.
echo 👋 Thanks for using Al-artworks Simple AI Suite!
echo 🎨 Keep creating amazing artwork!
pause 