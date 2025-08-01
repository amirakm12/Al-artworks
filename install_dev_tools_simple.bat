@echo off
echo ========================================
echo Simple C++ Development Tools Installer
echo ========================================
echo.

REM Set execution policy for PowerShell
echo Setting PowerShell execution policy...
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1

REM Create temporary directory
if not exist "%TEMP%\cpp_install" mkdir "%TEMP%\cpp_install"
cd /d "%TEMP%\cpp_install"

echo ========================================
echo 1. Installing Chocolatey Package Manager
echo ========================================
echo.

REM Check if Chocolatey is already installed
choco --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Chocolatey already installed
) else (
    echo Installing Chocolatey...
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if %errorLevel% neq 0 (
        echo WARNING: Chocolatey installation failed, trying alternative method...
        echo You may need to run this script as Administrator
        echo Right-click and select "Run as administrator"
        pause
    ) else (
        echo ✓ Chocolatey installed successfully
    )
)

echo.
echo ========================================
echo 2. Installing MSYS2 (MinGW-w64)
echo ========================================
echo.

REM Check if MSYS2 is already installed
if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✓ MSYS2 already installed
) else (
    echo Installing MSYS2...
    choco install msys2 -y
    if %errorLevel% neq 0 (
        echo WARNING: MSYS2 installation failed
        echo You can download it manually from: https://www.msys2.org/
    ) else (
        echo ✓ MSYS2 installed
        echo Adding MSYS2 to PATH...
        setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
    )
)

echo.
echo ========================================
echo 3. Installing CMake
echo ========================================
echo.

REM Check if CMake is installed
cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ CMake already installed
) else (
    echo Installing CMake...
    choco install cmake -y
    if %errorLevel% neq 0 (
        echo WARNING: CMake installation failed
        echo You can download it manually from: https://cmake.org/download/
    ) else (
        echo ✓ CMake installed
    )
)

echo.
echo ========================================
echo 4. Installing Git
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Git already installed
) else (
    echo Installing Git...
    choco install git -y
    if %errorLevel% neq 0 (
        echo WARNING: Git installation failed
        echo You can download it manually from: https://git-scm.com/
    ) else (
        echo ✓ Git installed
    )
)

echo.
echo ========================================
echo 5. Testing Installation
echo ========================================
echo.

REM Test GCC
echo Testing GCC...
gcc --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ GCC working
) else (
    echo ✗ GCC not found - you may need to restart your terminal
)

REM Test CMake
echo Testing CMake...
cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ CMake working
) else (
    echo ✗ CMake not found - you may need to restart your terminal
)

echo.
echo ========================================
echo Installation Summary
echo ========================================
echo.
echo If you see any warnings above, you can:
echo 1. Run this script as Administrator (right-click -> Run as administrator)
echo 2. Install tools manually from their websites
echo.
echo Manual installation links:
echo - MSYS2: https://www.msys2.org/
echo - CMake: https://cmake.org/download/
echo - Git: https://git-scm.com/
echo - Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
echo.
echo After installation:
echo 1. Restart your terminal/VS Code
echo 2. Try building with Ctrl+Shift+B
echo 3. Or run: .\build.bat
echo.
pause 