@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Enhanced Standalone Installer for Al-artworks
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

echo 📁 Creating installation directories...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
if not exist "%PROJECTS_DIR%" mkdir "%PROJECTS_DIR%"

echo ✅ Directories created successfully
echo.

REM ========================================
REM Step 1: Install MSYS2 (C++ Compiler)
REM ========================================
echo 🔧 Step 1: Installing MSYS2 (C++ Compiler)
echo    =========================================

if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✅ MSYS2 already installed
) else (
    echo 📥 Downloading MSYS2 installer...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe' -OutFile '%TOOLS_DIR%\msys2-installer.exe' -UseBasicParsing}"
    
    if exist "%TOOLS_DIR%\msys2-installer.exe" (
        echo 🔄 Installing MSYS2...
        "%TOOLS_DIR%\msys2-installer.exe" --accept-messages --accept-licenses --root C:\msys64
        if !errorLevel! equ 0 (
            echo ✅ MSYS2 installed successfully
            echo 🔧 Adding MSYS2 to PATH...
            setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
        ) else (
            echo ❌ MSYS2 installation failed
        )
    ) else (
        echo ❌ Failed to download MSYS2
    )
)

echo.

REM ========================================
REM Step 2: Install CMake
REM ========================================
echo 🔧 Step 2: Installing CMake (Build System)
echo    ========================================

cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ CMake already installed
) else (
    echo 📥 Downloading CMake installer...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi' -OutFile '%TOOLS_DIR%\cmake-installer.msi' -UseBasicParsing}"
    
    if exist "%TOOLS_DIR%\cmake-installer.msi" (
        echo 🔄 Installing CMake...
        msiexec /i "%TOOLS_DIR%\cmake-installer.msi" /quiet ADD_TO_PATH=1
        if !errorLevel! equ 0 (
            echo ✅ CMake installed successfully
        ) else (
            echo ❌ CMake installation failed
        )
    ) else (
        echo ❌ Failed to download CMake
    )
)

echo.

REM ========================================
REM Step 3: Install Git
REM ========================================
echo 🔧 Step 3: Installing Git (Version Control)
echo    =========================================

git --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Git already installed
) else (
    echo 📥 Downloading Git installer...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe' -OutFile '%TOOLS_DIR%\git-installer.exe' -UseBasicParsing}"
    
    if exist "%TOOLS_DIR%\git-installer.exe" (
        echo 🔄 Installing Git...
        "%TOOLS_DIR%\git-installer.exe" /VERYSILENT /NORESTART
        if !errorLevel! equ 0 (
            echo ✅ Git installed successfully
        ) else (
            echo ❌ Git installation failed
        )
    ) else (
        echo ❌ Failed to download Git
    )
)

echo.

REM ========================================
REM Step 4: Install Python Dependencies
REM ========================================
echo 🔧 Step 4: Installing Python Dependencies
echo    ======================================

python --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Python detected
    echo 📦 Installing Python packages...
    pip install --upgrade pip
    pip install numpy opencv-python pillow matplotlib seaborn
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers diffusers accelerate
    pip install flask fastapi uvicorn
    pip install requests beautifulsoup4 lxml
    echo ✅ Python dependencies installed
) else (
    echo ⚠️  Python not found - please install Python 3.8+ manually
)

echo.

REM ========================================
REM Step 5: Create Project Structure
REM ========================================
echo 🔧 Step 5: Creating Project Structure
echo    ==================================

REM Copy current project to installation directory
echo 📁 Copying Al-artworks project...
xcopy /E /I /Y "%~dp0Al-artworks" "%PROJECTS_DIR%\Al-artworks"
xcopy /E /I /Y "%~dp0aisis" "%PROJECTS_DIR%\aisis"

REM Create build scripts
echo 📝 Creating build scripts...

echo @echo off > "%PROJECTS_DIR%\build_all.bat"
echo echo Building Al-artworks project... >> "%PROJECTS_DIR%\build_all.bat"
echo cd /d "%PROJECTS_DIR%\aisis" >> "%PROJECTS_DIR%\build_all.bat"
echo cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >> "%PROJECTS_DIR%\build_all.bat"
echo cmake --build build --config Release >> "%PROJECTS_DIR%\build_all.bat"
echo echo Build complete! >> "%PROJECTS_DIR%\build_all.bat"
echo pause >> "%PROJECTS_DIR%\build_all.bat"

echo @echo off > "%PROJECTS_DIR%\run_ai_artworks.bat"
echo echo Starting Al-artworks AI Suite... >> "%PROJECTS_DIR%\run_ai_artworks.bat"
echo cd /d "%PROJECTS_DIR%\Al-artworks" >> "%PROJECTS_DIR%\run_ai_artworks.bat"
echo python -m flask run --host=0.0.0.0 --port=5000 >> "%PROJECTS_DIR%\run_ai_artworks.bat"

echo ✅ Project structure created

echo.

REM ========================================
REM Step 6: Create Desktop Shortcuts
REM ========================================
echo 🔧 Step 6: Creating Desktop Shortcuts
echo    ===================================

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = "%USERPROFILE%\Desktop\Al-artworks.lnk" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "%PROJECTS_DIR%\run_ai_artworks.bat" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%PROJECTS_DIR%" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "Al-artworks AI Suite" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.IconLocation = "%SystemRoot%\System32\shell32.dll,0" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"
cscript //nologo "%TEMP%\CreateShortcut.vbs"
del "%TEMP%\CreateShortcut.vbs"

echo ✅ Desktop shortcuts created

echo.

REM ========================================
REM Step 7: Test Installation
REM ========================================
echo 🔧 Step 7: Testing Installation
echo    =============================

echo 🧪 Testing GCC...
gcc --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ GCC working
) else (
    echo ⚠️  GCC not working - restart terminal after installation
)

echo 🧪 Testing CMake...
cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ CMake working
) else (
    echo ⚠️  CMake not working - restart terminal after installation
)

echo 🧪 Testing Git...
git --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Git working
) else (
    echo ⚠️  Git not working - restart terminal after installation
)

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
pause 