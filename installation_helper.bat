@echo off
echo ========================================
echo C++ Development Tools Installation Helper
echo ========================================
echo.

:menu
echo Choose an installation step:
echo.
echo 1. Install MSYS2 (C++ Compiler)
echo 2. Install CMake (Build System)
echo 3. Install Git (Version Control)
echo 4. Test All Installations
echo 5. Build Your Project
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto step1
if "%choice%"=="2" goto step2
if "%choice%"=="3" goto step3
if "%choice%"=="4" goto step4
if "%choice%"=="5" goto step5
if "%choice%"=="6" goto exit
goto menu

:step1
echo.
echo ========================================
echo Step 1: Installing MSYS2 (C++ Compiler)
echo ========================================
echo.
echo Opening MSYS2 download page...
start https://www.msys2.org/
echo.
echo Instructions:
echo 1. Click the download link on the website
echo 2. Save the installer to your Downloads folder
echo 3. Right-click the installer and select "Run as administrator"
echo 4. Install to the default location: C:\msys64\
echo 5. Complete the installation
echo.
echo After installation, press any key to continue...
pause
goto menu

:step2
echo.
echo ========================================
echo Step 2: Installing CMake (Build System)
echo ========================================
echo.
echo Opening CMake download page...
start https://cmake.org/download/
echo.
echo Instructions:
echo 1. Click "Windows x64 Installer" to download
echo 2. Right-click the installer and select "Run as administrator"
echo 3. Choose "Add CMake to system PATH for all users"
echo 4. Complete the installation
echo.
echo After installation, press any key to continue...
pause
goto menu

:step3
echo.
echo ========================================
echo Step 3: Installing Git (Version Control)
echo ========================================
echo.
echo Opening Git download page...
start https://git-scm.com/
echo.
echo Instructions:
echo 1. Click "Download for Windows" to download
echo 2. Right-click the installer and select "Run as administrator"
echo 3. Use default settings (just click Next)
echo 4. Choose "Git from the command line and also from 3rd-party software"
echo 5. Complete the installation
echo.
echo After installation, press any key to continue...
pause
goto menu

:step4
echo.
echo ========================================
echo Step 4: Testing All Installations
echo ========================================
echo.
echo Testing GCC (C++ Compiler)...
gcc --version
if %errorLevel% equ 0 (
    echo ✓ GCC is working!
) else (
    echo ✗ GCC not found. Make sure MSYS2 is installed and restart your terminal.
)

echo.
echo Testing CMake...
cmake --version
if %errorLevel% equ 0 (
    echo ✓ CMake is working!
) else (
    echo ✗ CMake not found. Make sure CMake is installed and restart your terminal.
)

echo.
echo Testing Git...
git --version
if %errorLevel% equ 0 (
    echo ✓ Git is working!
) else (
    echo ✗ Git not found. Make sure Git is installed and restart your terminal.
)

echo.
echo If any tools show as "not found":
echo 1. Restart your terminal/VS Code
echo 2. Make sure you ran installers as Administrator
echo 3. Check that tools are in your PATH
echo.
pause
goto menu

:step5
echo.
echo ========================================
echo Step 5: Building Your Project
echo ========================================
echo.
echo Testing build...
if exist "build.bat" (
    echo Running build script...
    call build.bat
) else (
    echo Build script not found. Creating one...
    echo @echo off > build.bat
    echo g++ -g -Wall -std=c++20 src\main.cpp -o build\debug\aisis.exe >> build.bat
    echo if exist "build\debug\aisis.exe" ( >> build.bat
    echo   echo Build successful! >> build.bat
    echo ) else ( >> build.bat
    echo   echo Build failed. Check if GCC is installed. >> build.bat
    echo ) >> build.bat
    echo.
    echo Created build.bat. Running it now...
    call build.bat
)

echo.
echo If build succeeds, you can:
echo - Press Ctrl+Shift+B in VS Code to build
echo - Press F5 in VS Code to debug
echo - Run .\build.bat from command line
echo.
pause
goto menu

:exit
echo.
echo Installation helper completed!
echo.
echo Remember to:
echo 1. Restart your terminal/VS Code after installations
echo 2. Test with: gcc --version, cmake --version, git --version
echo 3. Try building with Ctrl+Shift+B or .\build.bat
echo.
pause 