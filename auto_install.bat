@echo off
echo ========================================
echo Automated C++ Development Tools Installer
echo ========================================
echo.

REM Create temporary directory
if not exist "%TEMP%\cpp_auto_install" mkdir "%TEMP%\cpp_auto_install"
cd /d "%TEMP%\cpp_auto_install"

echo ========================================
echo Step 1: Downloading MSYS2
echo ========================================
echo.

if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✓ MSYS2 already installed
) else (
    echo Downloading MSYS2 installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe' -OutFile 'msys2-installer.exe'"
    
    if exist "msys2-installer.exe" (
        echo Installing MSYS2...
        msys2-installer.exe --accept-messages --accept-licenses --noconfirm --root C:\msys64
        if %errorLevel% equ 0 (
            echo ✓ MSYS2 installed successfully
            echo Adding MSYS2 to PATH...
            setx PATH "%PATH%;C:\msys64\mingw64\bin" /M
        ) else (
            echo ✗ MSYS2 installation failed
        )
    ) else (
        echo ✗ Failed to download MSYS2
    )
)

echo.
echo ========================================
echo Step 2: Downloading CMake
echo ========================================
echo.

cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ CMake already installed
) else (
    echo Downloading CMake installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi' -OutFile 'cmake-installer.msi'"
    
    if exist "cmake-installer.msi" (
        echo Installing CMake...
        msiexec /i cmake-installer.msi /quiet ADD_TO_PATH=1
        if %errorLevel% equ 0 (
            echo ✓ CMake installed successfully
        ) else (
            echo ✗ CMake installation failed
        )
    ) else (
        echo ✗ Failed to download CMake
    )
)

echo.
echo ========================================
echo Step 3: Downloading Git
echo ========================================
echo.

git --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Git already installed
) else (
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe' -OutFile 'git-installer.exe'"
    
    if exist "git-installer.exe" (
        echo Installing Git...
        git-installer.exe /VERYSILENT /NORESTART
        if %errorLevel% equ 0 (
            echo ✓ Git installed successfully
        ) else (
            echo ✗ Git installation failed
        )
    ) else (
        echo ✗ Failed to download Git
    )
)

echo.
echo ========================================
echo Step 4: Testing Installations
echo ========================================
echo.

echo Testing GCC...
gcc --version
if %errorLevel% equ 0 (
    echo ✓ GCC working
) else (
    echo ✗ GCC not working - restart terminal after installation
)

echo.
echo Testing CMake...
cmake --version
if %errorLevel% equ 0 (
    echo ✓ CMake working
) else (
    echo ✗ CMake not working - restart terminal after installation
)

echo.
echo Testing Git...
git --version
if %errorLevel% equ 0 (
    echo ✓ Git working
) else (
    echo ✗ Git not working - restart terminal after installation
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Restart your terminal/VS Code
echo 2. Test with: gcc --version, cmake --version, git --version
echo 3. Try building with Ctrl+Shift+B or .\build.bat
echo.
pause 