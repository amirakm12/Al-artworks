@echo off
setlocal enabledelayedexpansion

echo ========================================
echo AI-ARTWORKS BUILD SCRIPT
echo ========================================
echo.

:: Check if we're in the correct directory
if not exist "CMakeLists.txt" (
    echo ERROR: CMakeLists.txt not found. Please run this script from the AI-ARTWORKS directory.
    pause
    exit /b 1
)

:: Set up Visual Studio environment
echo Setting up Visual Studio environment...
call "%~dp0setup_vs_environment.ps1" -Architecture x64 -PreferredVersion 2022
if errorlevel 1 (
    echo ERROR: Failed to set up Visual Studio environment
    pause
    exit /b 1
)

:: Create build directory
echo Creating build directory...
if not exist "build" mkdir build
cd build

:: Configure with CMake
echo Configuring project with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    cd ..
    pause
    exit /b 1
)

:: Build the project
echo Building project...
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    pause
    exit /b 1
)

:: Copy dependencies if needed
echo Copying dependencies...
if exist "Release\ai-artworks.exe" (
    echo Build successful! Executable created: build\Release\ai-artworks.exe
) else (
    echo WARNING: Executable not found in expected location
)

cd ..

echo.
echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo To run the application:
echo   cd build\Release
echo   ai-artworks.exe
echo.
echo To run with a specific model:
echo   ai-artworks.exe --model "path\to\model.gguf"
echo.
pause