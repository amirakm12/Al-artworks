@echo off
echo ========================================
echo Portable C++ Development Tools Installer
echo (No Administrator Rights Required)
echo ========================================
echo.

REM Create portable installation directory
set PORTABLE_DIR=%USERPROFILE%\CppDevTools
if not exist "%PORTABLE_DIR%" mkdir "%PORTABLE_DIR%"
cd /d "%PORTABLE_DIR%"

echo ========================================
echo Step 1: Creating Portable Environment
echo ========================================
echo.

REM Create portable directory structure
if not exist "bin" mkdir bin
if not exist "include" mkdir include
if not exist "lib" mkdir lib
if not exist "projects" mkdir projects

echo ✓ Portable environment created

echo.
echo ========================================
echo Step 2: Downloading Portable Tools
echo ========================================
echo.

REM Download portable CMake
echo Downloading portable CMake...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.zip' -OutFile 'cmake-portable.zip'"

if exist "cmake-portable.zip" (
    echo Extracting CMake...
    powershell -Command "Expand-Archive -Path 'cmake-portable.zip' -DestinationPath 'cmake-portable' -Force"
    copy "cmake-portable\cmake-3.28.1-windows-x86_64\bin\cmake.exe" "bin\"
    echo ✓ Portable CMake ready
) else (
    echo ✗ Failed to download CMake
)

echo.
echo ========================================
echo Step 3: Creating Project Template
echo ========================================
echo.

REM Create portable project template
if not exist "projects\template" mkdir "projects\template"
if not exist "projects\template\src" mkdir "projects\template\src"
if not exist "projects\template\include" mkdir "projects\template\include"

REM Create sample main.cpp
echo #include ^<iostream^> > "projects\template\src\main.cpp"
echo. >> "projects\template\src\main.cpp"
echo int main() { >> "projects\template\src\main.cpp"
echo     std::cout ^<^< "Hello, C++ World!" ^<^< std::endl; >> "projects\template\src\main.cpp"
echo     return 0; >> "projects\template\src\main.cpp"
echo } >> "projects\template\src\main.cpp"

REM Create CMakeLists.txt
echo cmake_minimum_required(VERSION 3.20) > "projects\template\CMakeLists.txt"
echo project(MyProject VERSION 1.0.0 LANGUAGES CXX) >> "projects\template\CMakeLists.txt"
echo set(CMAKE_CXX_STANDARD 20) >> "projects\template\CMakeLists.txt"
echo set(CMAKE_CXX_STANDARD_REQUIRED ON) >> "projects\template\CMakeLists.txt"
echo file(GLOB_RECURSE SOURCES src/*.cpp) >> "projects\template\CMakeLists.txt"
echo add_executable(${PROJECT_NAME} ${SOURCES}) >> "projects\template\CMakeLists.txt"

REM Create build script
echo @echo off > "projects\template\build.bat"
echo echo Building project... >> "projects\template\build.bat"
echo cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug >> "projects\template\build.bat"
echo cmake --build build --config Debug >> "projects\template\build.bat"
echo if exist "build\Debug\*.exe" ( >> "projects\template\build.bat"
echo   echo Build successful! >> "projects\template\build.bat"
echo   echo Running executable... >> "projects\template\build.bat"
echo   build\Debug\*.exe >> "projects\template\build.bat"
echo ) else ( >> "projects\template\build.bat"
echo   echo Build failed! >> "projects\template\build.bat"
echo ) >> "projects\template\build.bat"

echo ✓ Project template created

echo.
echo ========================================
echo Step 4: Creating Environment Setup
echo ========================================
echo.

REM Create environment setup script
echo @echo off > "setup_env.bat"
echo echo Setting up portable C++ environment... >> "setup_env.bat"
echo set PATH=%~dp0bin;%PATH%% >> "setup_env.bat"
echo set CC=%~dp0bin\gcc.exe >> "setup_env.bat"
echo set CXX=%~dp0bin\g++.exe >> "setup_env.bat"
echo echo Environment ready! >> "setup_env.bat"
echo echo Run: setup_env.bat before building >> "setup_env.bat"

echo ✓ Environment setup created

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Portable installation created at:
echo %PORTABLE_DIR%
echo.
echo Project template at:
echo %PORTABLE_DIR%\projects\template
echo.
echo To use:
echo 1. Run: setup_env.bat
echo 2. Copy template to your project folder
echo 3. Run: build.bat
echo.
echo No administrator rights required!
echo.
pause 