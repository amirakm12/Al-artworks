@echo off
echo ========================================
echo Installing C++ Development Tools
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo ✓ Running as Administrator
echo.

REM Create temporary directory
if not exist "%TEMP%\cpp_dev_install" mkdir "%TEMP%\cpp_dev_install"
cd /d "%TEMP%\cpp_dev_install"

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
        echo ERROR: Failed to install Chocolatey
        pause
        exit /b 1
    )
    echo ✓ Chocolatey installed successfully
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
        echo ERROR: Failed to install MSYS2
        pause
        exit /b 1
    )
    
    REM Add MSYS2 to PATH
    echo Adding MSYS2 to PATH...
    setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
    echo ✓ MSYS2 installed and added to PATH
)

echo.
echo ========================================
echo 3. Installing Visual Studio Build Tools
echo ========================================
echo.

REM Check if Visual Studio Build Tools are installed
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools" (
    echo ✓ Visual Studio Build Tools already installed
) else (
    echo Installing Visual Studio Build Tools...
    
    REM Download Visual Studio Build Tools installer
    echo Downloading Visual Studio Build Tools...
    powershell -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -OutFile 'vs_buildtools.exe'"
    
    REM Install with C++ workload
    echo Installing C++ build tools...
    vs_buildtools.exe --quiet --wait --norestart --nocache --installPath "C:\Program Files\Microsoft Visual Studio\2022\BuildTools" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows10SDK.19041
    
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Visual Studio Build Tools
        pause
        exit /b 1
    )
    echo ✓ Visual Studio Build Tools installed
)

echo.
echo ========================================
echo 4. Installing CMake
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
        echo ERROR: Failed to install CMake
        pause
        exit /b 1
    )
    echo ✓ CMake installed
)

echo.
echo ========================================
echo 5. Installing Git
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
        echo ERROR: Failed to install Git
        pause
        exit /b 1
    )
    echo ✓ Git installed
)

echo.
echo ========================================
echo 6. Installing vcpkg
echo ========================================
echo.

REM Check if vcpkg is installed
if exist "C:\vcpkg\vcpkg.exe" (
    echo ✓ vcpkg already installed
) else (
    echo Installing vcpkg...
    cd /d C:\
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install
    
    REM Add vcpkg to PATH
    setx PATH "%PATH%;C:\vcpkg" /M
    echo ✓ vcpkg installed and integrated
)

echo.
echo ========================================
echo 7. Installing Python (for build tools)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Python already installed
) else (
    echo Installing Python...
    choco install python -y
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Python
        pause
        exit /b 1
    )
    echo ✓ Python installed
)

echo.
echo ========================================
echo 8. Installing Ninja Build System
echo ========================================
echo.

REM Check if Ninja is installed
ninja --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Ninja already installed
) else (
    echo Installing Ninja...
    choco install ninja -y
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Ninja
        pause
        exit /b 1
    )
    echo ✓ Ninja installed
)

echo.
echo ========================================
echo 9. Installing Windows SDK
echo ========================================
echo.

REM Check if Windows SDK is installed
if exist "C:\Program Files (x86)\Windows Kits\10\Include" (
    echo ✓ Windows SDK already installed
) else (
    echo Installing Windows SDK...
    choco install windows-sdk-10-version-2004-all -y
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Windows SDK
        pause
        exit /b 1
    )
    echo ✓ Windows SDK installed
)

echo.
echo ========================================
echo 10. Testing Installation
echo ========================================
echo.

REM Test GCC
echo Testing GCC...
gcc --version
if %errorLevel% equ 0 (
    echo ✓ GCC working
) else (
    echo ✗ GCC not working - you may need to restart your terminal
)

REM Test MSVC
echo Testing MSVC...
where cl
if %errorLevel% equ 0 (
    echo ✓ MSVC found
) else (
    echo ✗ MSVC not found - you may need to restart your terminal
)

REM Test CMake
echo Testing CMake...
cmake --version
if %errorLevel% equ 0 (
    echo ✓ CMake working
) else (
    echo ✗ CMake not working - you may need to restart your terminal
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Installed tools:
echo - MSYS2 (MinGW-w64 GCC)
echo - Visual Studio Build Tools (MSVC)
echo - CMake
echo - Git
echo - vcpkg
echo - Python
echo - Ninja
echo - Windows SDK
echo.
echo Next steps:
echo 1. Restart your terminal/VS Code
echo 2. Try building your project with Ctrl+Shift+B
echo 3. Or run: .\build.bat
echo.
echo Your project is now ready for C++ development!
echo.
pause 