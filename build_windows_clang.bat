@echo off
REM Build script for Windows with Clang/LLVM toolchain
REM This script addresses the lld-link library linking issues

echo ========================================
echo Windows Clang Build Configuration
echo ========================================

REM Set local environment
setlocal enabledelayedexpansion

REM Check if Visual Studio is installed
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: Visual Studio not found. Please install Visual Studio 2019 or later.
    exit /b 1
)

REM Find Visual Studio installation path
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSINSTALLDIR=%%i"
)

if "%VSINSTALLDIR%"=="" (
    echo ERROR: Visual Studio installation not found.
    exit /b 1
)

echo Found Visual Studio at: %VSINSTALLDIR%

REM Find MSVC tools version
for /f %%i in ('dir "%VSINSTALLDIR%\VC\Tools\MSVC" /b /o-n') do (
    set "MSVC_VERSION=%%i"
    goto :found_msvc
)
:found_msvc

set "MSVC_TOOLS_PATH=%VSINSTALLDIR%\VC\Tools\MSVC\%MSVC_VERSION%"
echo Using MSVC Tools: %MSVC_VERSION%

REM Find Windows SDK
set "WINDOWS_KITS_ROOT=%ProgramFiles(x86)%\Windows Kits\10"
if not exist "%WINDOWS_KITS_ROOT%" (
    set "WINDOWS_KITS_ROOT=%ProgramFiles%\Windows Kits\10"
)

if not exist "%WINDOWS_KITS_ROOT%" (
    echo ERROR: Windows SDK not found.
    exit /b 1
)

REM Find latest Windows SDK version
for /f %%i in ('dir "%WINDOWS_KITS_ROOT%\Lib" /b /o-n') do (
    if exist "%WINDOWS_KITS_ROOT%\Lib\%%i\ucrt\x64" (
        set "WINSDK_VERSION=%%i"
        goto :found_winsdk
    )
)
:found_winsdk

echo Using Windows SDK: %WINSDK_VERSION%

REM Set up environment variables for Clang
set "INCLUDE=%MSVC_TOOLS_PATH%\include;%WINDOWS_KITS_ROOT%\Include\%WINSDK_VERSION%\ucrt;%WINDOWS_KITS_ROOT%\Include\%WINSDK_VERSION%\shared;%WINDOWS_KITS_ROOT%\Include\%WINSDK_VERSION%\um;%WINDOWS_KITS_ROOT%\Include\%WINSDK_VERSION%\winrt"

set "LIB=%MSVC_TOOLS_PATH%\lib\x64;%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\ucrt\x64;%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\um\x64"

set "LIBPATH=%MSVC_TOOLS_PATH%\lib\x64;%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\ucrt\x64"

REM Add MSVC tools to PATH
set "PATH=%MSVC_TOOLS_PATH%\bin\Hostx64\x64;%PATH%"

echo ========================================
echo Environment Setup Complete
echo ========================================
echo INCLUDE=%INCLUDE%
echo.
echo LIB=%LIB%
echo.
echo LIBPATH=%LIBPATH%
echo ========================================

REM Create build directory
if not exist build mkdir build
cd build

echo ========================================
echo Running CMake Configuration
echo ========================================

REM Solution 1: Use Microsoft linker with Clang (Recommended)
cmake .. -G "Ninja" ^
    -DCMAKE_C_COMPILER=clang-cl ^
    -DCMAKE_CXX_COMPILER=clang-cl ^
    -DCMAKE_LINKER=link.exe ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_C_FLAGS="/clang:-fms-compatibility-version=19.29" ^
    -DCMAKE_CXX_FLAGS="/clang:-fms-compatibility-version=19.29"

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed with Microsoft linker. Trying alternative...
    
    REM Solution 2: Use lld-link with explicit library paths
    cmake .. -G "Ninja" ^
        -DCMAKE_C_COMPILER=clang ^
        -DCMAKE_CXX_COMPILER=clang++ ^
        -DCMAKE_LINKER=lld-link ^
        -DCMAKE_BUILD_TYPE=Debug ^
        -DCMAKE_EXE_LINKER_FLAGS="/LIBPATH:\"%MSVC_TOOLS_PATH%\lib\x64\" /LIBPATH:\"%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\ucrt\x64\" /LIBPATH:\"%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\um\x64\"" ^
        -DCMAKE_SHARED_LINKER_FLAGS="/LIBPATH:\"%MSVC_TOOLS_PATH%\lib\x64\" /LIBPATH:\"%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\ucrt\x64\" /LIBPATH:\"%WINDOWS_KITS_ROOT%\Lib\%WINSDK_VERSION%\um\x64\""
)

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed with both approaches.
    echo Please check your Clang and Visual Studio installations.
    exit /b 1
)

echo ========================================
echo Building Project
echo ========================================

ninja

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================

cd ..
endlocal