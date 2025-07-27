@echo off
REM ULTIMATE System - Windows Build Script
REM This script builds the ULTIMATE System on Windows platforms

echo 🚀 ULTIMATE System - Windows Build Script
echo ==========================================

REM Check for CMake
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake not found. Please install CMake and add it to PATH
    exit /b 1
)

REM Configuration
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

set BUILD_DIR=build_windows
set GENERATOR=%2
if "%GENERATOR%"=="" set GENERATOR=Visual Studio 17 2022

echo 📋 Build Configuration:
echo    Build Type: %BUILD_TYPE%
echo    Build Directory: %BUILD_DIR%
echo    Generator: %GENERATOR%
echo.

REM Clean previous build if requested
if "%3"=="clean" (
    echo 🧹 Cleaning previous build...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM Configure with CMake
echo ⚙️ Configuring with CMake...
if "%GENERATOR%"=="MinGW Makefiles" (
    cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_EXAMPLES=ON
    if %ERRORLEVEL% neq 0 (
        echo ❌ CMake configuration failed
        exit /b 1
    )
    
    echo 🔨 Building with MinGW...
    mingw32-make -j%NUMBER_OF_PROCESSORS%
) else (
    cmake .. -G "%GENERATOR%" -A x64 -DBUILD_EXAMPLES=ON
    if %ERRORLEVEL% neq 0 (
        echo ❌ CMake configuration failed
        exit /b 1
    )
    
    echo 🔨 Building with Visual Studio...
    cmake --build . --config %BUILD_TYPE% --parallel %NUMBER_OF_PROCESSORS%
)

if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed
    exit /b 1
)

REM Test the build
echo 🧪 Testing build artifacts...
if "%GENERATOR%"=="MinGW Makefiles" (
    if exist "lib\libultimate.a" (
        echo ✅ Core library built successfully
    ) else (
        echo ❌ Core library build failed
        exit /b 1
    )
    
    if exist "bin\ultimate_system.exe" (
        echo ✅ Main executable built successfully
    ) else (
        echo ❌ Main executable build failed
        exit /b 1
    )
) else (
    if exist "%BUILD_TYPE%\ultimate.lib" (
        echo ✅ Core library built successfully
    ) else (
        echo ❌ Core library build failed
        exit /b 1
    )
    
    if exist "%BUILD_TYPE%\ultimate_system.exe" (
        echo ✅ Main executable built successfully
    ) else (
        echo ❌ Main executable build failed
        exit /b 1
    )
)

echo.
echo 🎉 ULTIMATE System build completed successfully!
echo 📦 Build artifacts are in: %BUILD_DIR%
echo.
echo 🚀 Alternative build commands:
echo    Visual Studio: build_windows.bat Release "Visual Studio 17 2022"
echo    MinGW:         build_windows.bat Release "MinGW Makefiles"
echo    Clean build:   build_windows.bat Release "Visual Studio 17 2022" clean