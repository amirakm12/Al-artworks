@echo off
echo ========================================
echo Portable C++ Development Setup
echo (No Administrator Rights Required)
echo ========================================
echo.

REM Create portable directory structure
if not exist "tools" mkdir tools
if not exist "tools\bin" mkdir tools\bin
if not exist "tools\include" mkdir tools\include
if not exist "tools\lib" mkdir tools\lib

echo Creating portable development environment...
echo.

REM Create a simple batch file to set up environment
echo @echo off > setup_env.bat
echo echo Setting up portable C++ environment... >> setup_env.bat
echo set PATH=%%~dp0tools\bin;%%PATH%% >> setup_env.bat
echo set CC=%%~dp0tools\bin\gcc.exe >> setup_env.bat
echo set CXX=%%~dp0tools\bin\g++.exe >> setup_env.bat
echo echo Environment ready! >> setup_env.bat
echo echo Run: setup_env.bat before building >> setup_env.bat

REM Create a simple build script that works without external tools
echo @echo off > build_simple.bat
echo echo Building with available tools... >> build_simple.bat
echo if exist "tools\bin\g++.exe" ( >> build_simple.bat
echo   echo Using portable GCC... >> build_simple.bat
echo   tools\bin\g++.exe -g -Wall -std=c++20 src\main.cpp -o build\debug\aisis.exe >> build_simple.bat
echo ) else ( >> build_simple.bat
echo   echo No compiler found. Please install MSYS2 manually. >> build_simple.bat
echo   echo Download from: https://www.msys2.org/ >> build_simple.bat
echo ) >> build_simple.bat

echo ========================================
echo Portable Setup Complete!
echo ========================================
echo.
echo This setup doesn't require administrator rights.
echo.
echo To install tools manually (recommended):
echo 1. Download MSYS2 from: https://www.msys2.org/
echo 2. Install to: tools\msys2\
echo 3. Copy gcc.exe and g++.exe to: tools\bin\
echo.
echo Or use the manual installation guide:
echo - Open MANUAL_INSTALL_GUIDE.md
echo - Follow the step-by-step instructions
echo.
echo After installing tools:
echo 1. Run: setup_env.bat
echo 2. Run: build_simple.bat
echo 3. Or press Ctrl+Shift+B in VS Code
echo.
pause 