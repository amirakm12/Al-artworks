@echo off
echo Installing MSYS2...
echo.

REM Check if MSYS2 is already installed
if exist "C:\msys64" (
    echo MSYS2 is already installed at C:\msys64
    goto :install_packages
)

REM Run the installer
echo Running MSYS2 installer...
msys2-installer.exe --accept-messages --accept-licenses

REM Wait a moment for installation
timeout /t 5 /nobreak >nul

REM Check if installation was successful
if not exist "C:\msys64" (
    echo ERROR: MSYS2 installation failed
    echo Please run the installer manually
    pause
    exit /b 1
)

:install_packages
echo.
echo Installing required packages...
C:\msys64\usr\bin\pacman.exe -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-gcc-libs mingw-w64-x86_64-gcc-objc mingw-w64-x86_64-gdb mingw-w64-x86_64-make

echo.
echo Installation complete!
echo Please restart your terminal to use gcc/g++
pause 