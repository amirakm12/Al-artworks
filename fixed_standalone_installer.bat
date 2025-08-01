@echo off
setlocal enabledelayedexpansion

REM ========================================
REM FIXED Enhanced Standalone Installer for Al-artworks
REM ========================================

REM Set console colors for modern UI
color 0A
cls

echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                    AL-ARTWORKS INSTALLER                     ║
echo    ║                    ======================                     ║
echo    ║                                                              ║
echo    ║  🎨 AI-Powered Art Creation Suite                           ║
echo    ║  🚀 Modern Development Environment                          ║
echo    ║  ⚡ Lightning-Fast Installation                             ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  This installer requires administrator privileges
    echo    Please run as administrator and try again
    pause
    exit /b 1
)

echo ✅ Administrator privileges confirmed
echo.

REM Create installation directory
set INSTALL_DIR=C:\Al-artworks
set TOOLS_DIR=%INSTALL_DIR%\tools
set PROJECTS_DIR=%INSTALL_DIR%\projects
set LOGS_DIR=%INSTALL_DIR%\logs

echo 📁 Creating installation directories...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
if not exist "%PROJECTS_DIR%" mkdir "%PROJECTS_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

echo ✅ Directories created successfully
echo.

REM ========================================
REM Step 1: Install MSYS2 (C++ Compiler) - FIXED
REM ========================================
echo 🔧 Step 1: Installing MSYS2 (C++ Compiler)
echo    =========================================

REM Check if MSYS2 is already installed and working
gcc --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ MSYS2 already installed and working
    goto :cmake_install
)

REM Check if MSYS2 is installed but not in PATH
if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo 🔧 MSYS2 found but not in PATH, adding...
    setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
    echo ✅ MSYS2 added to PATH
    goto :cmake_install
)

REM Kill any existing MSYS2 installer processes
taskkill /f /im msys2-installer.exe >nul 2>&1
timeout /t 2 >nul

echo 📥 Downloading MSYS2 installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe' -OutFile '%TOOLS_DIR%\msys2-installer.exe' -UseBasicParsing}"

if not exist "%TOOLS_DIR%\msys2-installer.exe" (
    echo ❌ Failed to download MSYS2
    echo    Trying alternative download method...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe' -OutFile '%TOOLS_DIR%\msys2-installer.exe'"
    
    if not exist "%TOOLS_DIR%\msys2-installer.exe" (
        echo ❌ MSYS2 download failed completely
        goto :cmake_install
    )
)

echo 🔄 Installing MSYS2...
echo    This may take several minutes...

REM Run MSYS2 installer with proper parameters
"%TOOLS_DIR%\msys2-installer.exe" --accept-messages --accept-licenses --root C:\msys64 --noconfirm

REM Wait for installation to complete
timeout /t 5 >nul

REM Check if installation was successful
if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✅ MSYS2 installed successfully
    echo 🔧 Adding MSYS2 to PATH...
    setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
    
    REM Update current session PATH
    set PATH=%PATH%;C:\msys64\mingw64\bin
) else (
    echo ❌ MSYS2 installation failed
    echo    Please install MSYS2 manually from https://www.msys2.org/
)

:cmake_install
echo.

REM ========================================
REM Step 2: Install CMake - FIXED
REM ========================================
echo 🔧 Step 2: Installing CMake (Build System)
echo    ========================================

REM Check if CMake is already installed
cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ CMake already installed
    goto :git_install
)

echo 📥 Downloading CMake installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi' -OutFile '%TOOLS_DIR%\cmake-installer.msi' -UseBasicParsing}"

if exist "%TOOLS_DIR%\cmake-installer.msi" (
    echo 🔄 Installing CMake...
    msiexec /i "%TOOLS_DIR%\cmake-installer.msi" /quiet ADD_TO_PATH=1 /l*v "%LOGS_DIR%\cmake-install.log"
    
    REM Wait for installation
    timeout /t 3 >nul
    
    REM Test installation
    cmake --version >nul 2>&1
    if !errorLevel! equ 0 (
        echo ✅ CMake installed successfully
    ) else (
        echo ❌ CMake installation failed
        echo    Please install CMake manually from https://cmake.org/download/
    )
) else (
    echo ❌ Failed to download CMake
)

:git_install
echo.

REM ========================================
REM Step 3: Install Git - FIXED
REM ========================================
echo 🔧 Step 3: Installing Git (Version Control)
echo    =========================================

REM Check if Git is already installed
git --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Git already installed
    goto :python_install
)

echo 📥 Downloading Git installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe' -OutFile '%TOOLS_DIR%\git-installer.exe' -UseBasicParsing}"

if exist "%TOOLS_DIR%\git-installer.exe" (
    echo 🔄 Installing Git...
    "%TOOLS_DIR%\git-installer.exe" /VERYSILENT /NORESTART /COMPONENTS="icons,ext\reg\shellhere,ext\reg\guihere"
    
    REM Wait for installation
    timeout /t 3 >nul
    
    REM Test installation
    git --version >nul 2>&1
    if !errorLevel! equ 0 (
        echo ✅ Git installed successfully
    ) else (
        echo ❌ Git installation failed
        echo    Please install Git manually from https://git-scm.com/
    )
) else (
    echo ❌ Failed to download Git
)

:python_install
echo.

REM ========================================
REM Step 4: Install Python Dependencies - FIXED
REM ========================================
echo 🔧 Step 4: Installing Python Dependencies
echo    ======================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  Python not found
    echo    Please install Python 3.8+ from https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation
    goto :project_setup
)

echo ✅ Python detected
echo 📦 Installing Python packages...

REM Upgrade pip first
python -m pip install --upgrade pip --quiet

REM Install packages with error handling
echo    Installing core packages...
python -m pip install numpy opencv-python pillow matplotlib seaborn --quiet
if !errorLevel! neq 0 (
    echo ⚠️  Some core packages failed to install
)

echo    Installing PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if !errorLevel! neq 0 (
    echo ⚠️  PyTorch installation failed
)

echo    Installing AI/ML packages...
python -m pip install transformers diffusers accelerate --quiet
if !errorLevel! neq 0 (
    echo ⚠️  Some AI/ML packages failed to install
)

echo    Installing web frameworks...
python -m pip install flask fastapi uvicorn --quiet
if !errorLevel! neq 0 (
    echo ⚠️  Some web packages failed to install
)

echo    Installing utility packages...
python -m pip install requests beautifulsoup4 lxml --quiet
if !errorLevel! neq 0 (
    echo ⚠️  Some utility packages failed to install
)

echo ✅ Python dependencies installation completed

:project_setup
echo.

REM ========================================
REM Step 5: Create Project Structure - FIXED
REM ========================================
echo 🔧 Step 5: Creating Project Structure
echo    ==================================

REM Copy current project to installation directory with error handling
echo 📁 Copying Al-artworks project...

if exist "%~dp0Al-artworks" (
    xcopy /E /I /Y "%~dp0Al-artworks" "%PROJECTS_DIR%\Al-artworks" >nul 2>&1
    if !errorLevel! equ 0 (
        echo ✅ Al-artworks project copied
    ) else (
        echo ⚠️  Al-artworks project copy failed
    )
) else (
    echo ⚠️  Al-artworks directory not found in current location
)

if exist "%~dp0aisis" (
    xcopy /E /I /Y "%~dp0aisis" "%PROJECTS_DIR%\aisis" >nul 2>&1
    if !errorLevel! equ 0 (
        echo ✅ Aisis project copied
    ) else (
        echo ⚠️  Aisis project copy failed
    )
) else (
    echo ⚠️  Aisis directory not found in current location
)

REM Create build scripts with proper error handling
echo 📝 Creating build scripts...

REM Build script
(
echo @echo off
echo echo Building Al-artworks project...
echo cd /d "%PROJECTS_DIR%\aisis"
echo if exist "build" rmdir /s /q "build"
echo cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
echo if !errorLevel! equ 0 ^(
echo   cmake --build build --config Release
echo   if !errorLevel! equ 0 ^(
echo     echo Build successful!
echo   ^) else ^(
echo     echo Build failed!
echo   ^)
echo ^) else ^(
echo   echo CMake configuration failed!
echo ^)
echo pause
) > "%PROJECTS_DIR%\build_all.bat"

REM Run script
(
echo @echo off
echo echo Starting Al-artworks AI Suite...
echo cd /d "%PROJECTS_DIR%\Al-artworks"
echo if exist "app.py" ^(
echo   python app.py
echo ^) else if exist "main.py" ^(
echo   python main.py
echo ^) else ^(
echo   python -m flask run --host=0.0.0.0 --port=5000
echo ^)
echo pause
) > "%PROJECTS_DIR%\run_ai_artworks.bat"

echo ✅ Build scripts created

echo.

REM ========================================
REM Step 6: Create Desktop Shortcuts - FIXED
REM ========================================
echo 🔧 Step 6: Creating Desktop Shortcuts
echo    ===================================

REM Create VBS script for shortcut creation
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%USERPROFILE%\Desktop\Al-artworks.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%PROJECTS_DIR%\run_ai_artworks.bat"
echo oLink.WorkingDirectory = "%PROJECTS_DIR%"
echo oLink.Description = "Al-artworks AI Suite"
echo oLink.IconLocation = "%SystemRoot%\System32\shell32.dll,0"
echo oLink.Save
) > "%TEMP%\CreateShortcut.vbs"

REM Execute VBS script
cscript //nologo "%TEMP%\CreateShortcut.vbs" >nul 2>&1
if !errorLevel! equ 0 (
    echo ✅ Desktop shortcut created
) else (
    echo ⚠️  Failed to create desktop shortcut
)

REM Clean up
del "%TEMP%\CreateShortcut.vbs" >nul 2>&1

echo.

REM ========================================
REM Step 7: Test Installation - FIXED
REM ========================================
echo 🔧 Step 7: Testing Installation
echo    =============================

set TEST_PASSED=0
set TEST_TOTAL=0

REM Test GCC
set /a TEST_TOTAL+=1
echo 🧪 Testing GCC...
gcc --version >nul 2>&1
if !errorLevel! equ 0 (
    echo ✅ GCC working
    set /a TEST_PASSED+=1
) else (
    echo ⚠️  GCC not working - restart terminal after installation
)

REM Test CMake
set /a TEST_TOTAL+=1
echo 🧪 Testing CMake...
cmake --version >nul 2>&1
if !errorLevel! equ 0 (
    echo ✅ CMake working
    set /a TEST_PASSED+=1
) else (
    echo ⚠️  CMake not working - restart terminal after installation
)

REM Test Git
set /a TEST_TOTAL+=1
echo 🧪 Testing Git...
git --version >nul 2>&1
if !errorLevel! equ 0 (
    echo ✅ Git working
    set /a TEST_PASSED+=1
) else (
    echo ⚠️  Git not working - restart terminal after installation
)

REM Test Python
set /a TEST_TOTAL+=1
echo 🧪 Testing Python...
python --version >nul 2>&1
if !errorLevel! equ 0 (
    echo ✅ Python working
    set /a TEST_PASSED+=1
) else (
    echo ⚠️  Python not working
)

echo.
echo 📊 Installation Test Results: !TEST_PASSED!/!TEST_TOTAL! tests passed

echo.

REM ========================================
REM Installation Complete
REM ========================================
echo.
echo    ╔══════════════════════════════════════════════════════════════╗
echo    ║                    INSTALLATION COMPLETE!                    ║
echo    ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🎉 Al-artworks has been successfully installed!
echo.
echo 📁 Installation Location: %INSTALL_DIR%
echo 📁 Projects Location: %PROJECTS_DIR%
echo 📁 Logs Location: %LOGS_DIR%
echo.
echo 🚀 Quick Start:
echo    1. Double-click "Al-artworks" on your desktop
echo    2. Or run: %PROJECTS_DIR%\run_ai_artworks.bat
echo.
echo 🔧 Development Tools:
echo    - MSYS2 (C++ Compiler)
echo    - CMake (Build System)
echo    - Git (Version Control)
echo    - Python Dependencies
echo.
echo ⚡ Next Steps:
echo    1. Restart your terminal/VS Code
echo    2. Open the project in VS Code
echo    3. Press Ctrl+Shift+B to build
echo    4. Start creating AI-powered artwork!
echo.
echo 🎨 Your AI Art Creation Suite is ready!
echo.
echo 📋 Troubleshooting:
echo    - If tools don't work, restart your terminal
echo    - Check logs in: %LOGS_DIR%
echo    - Manual installation guides available in project docs
echo.
pause 