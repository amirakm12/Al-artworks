# Windows Clang Build Guide

## Problem Description

When building C/C++ projects with Clang/LLVM on Windows, you may encounter linking errors like:

```
lld-link: error: could not open 'oldnames.lib': no such file or directory
lld-link: error: could not open 'msvcrtd.lib': no such file or directory
```

This occurs because lld-link (LLVM's linker) cannot find the Microsoft Visual C++ runtime libraries that are required for Windows applications.

## Root Cause

The issue stems from the fact that:

1. **lld-link** expects libraries to be in specific search paths
2. **MSVC runtime libraries** (`oldnames.lib`, `msvcrtd.lib`, etc.) are installed with Visual Studio
3. **Library search paths** are not automatically configured for lld-link when using Clang

## Solutions

### Solution 1: Use Microsoft Linker (Recommended)

The most reliable approach is to use Clang compiler with Microsoft's `link.exe` linker:

#### Manual CMake Configuration:
```bash
cmake .. -G "Ninja" \
    -DCMAKE_C_COMPILER=clang-cl \
    -DCMAKE_CXX_COMPILER=clang-cl \
    -DCMAKE_LINKER=link.exe \
    -DCMAKE_BUILD_TYPE=Debug
```

#### Using the Batch Script:
```cmd
build_windows_clang.bat
```

#### Using the PowerShell Script:
```powershell
.\build_windows_clang.ps1 -UseClangCL
```

### Solution 2: Configure lld-link with Library Paths

If you prefer to use lld-link, you need to explicitly specify library search paths:

#### Find Your Paths:
1. **Visual Studio Installation**: Usually `C:\Program Files\Microsoft Visual Studio\2022\Community`
2. **MSVC Tools**: `{VS_PATH}\VC\Tools\MSVC\{VERSION}\lib\x64`
3. **Windows SDK**: `C:\Program Files (x86)\Windows Kits\10\Lib\{VERSION}\ucrt\x64`

#### Manual CMake Configuration:
```bash
cmake .. -G "Ninja" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER=lld-link \
    -DCMAKE_EXE_LINKER_FLAGS="/LIBPATH:\"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\lib\x64\" /LIBPATH:\"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64\" /LIBPATH:\"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64\""
```

#### Using the PowerShell Script with LLD:
```powershell
.\build_windows_clang.ps1 -ForceLLD
```

### Solution 3: Environment Variables

Set up the environment variables that lld-link uses to find libraries:

```cmd
set "LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

set "INCLUDE=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um"
```

## Automated Solutions

### Option 1: Batch Script (`build_windows_clang.bat`)

Features:
- Automatically finds Visual Studio and Windows SDK installations
- Sets up proper environment variables
- Tries multiple build strategies
- Works with Command Prompt

Usage:
```cmd
build_windows_clang.bat
```

### Option 2: PowerShell Script (`build_windows_clang.ps1`)

Features:
- More robust error handling
- Multiple build strategies with fallbacks
- Colored output and detailed logging
- Flexible command-line options

Usage:
```powershell
# Basic build
.\build_windows_clang.ps1

# Clean build with Release configuration
.\build_windows_clang.ps1 -Clean -BuildType Release

# Force use of lld-link
.\build_windows_clang.ps1 -ForceLLD

# Use regular clang instead of clang-cl
.\build_windows_clang.ps1 -UseClangCL:$false
```

### Option 3: Modified CMakeLists.txt

The project's `CMakeLists.txt` has been updated with automatic detection and configuration for Clang on Windows. It will:

1. Detect if Clang is being used
2. Automatically find Visual Studio and Windows SDK paths
3. Configure the appropriate linker and library paths
4. Fall back to Microsoft linker if needed

## Build Strategies Tried (in order)

The automated scripts try multiple strategies:

1. **ClangCL + MSVC Linker**: `clang-cl` with `link.exe` (most compatible)
2. **Clang + MSVC Linker**: `clang` with `link.exe` 
3. **ClangCL + LLD**: `clang-cl` with `lld-link` and explicit paths
4. **Clang + LLD**: `clang` with `lld-link` and explicit paths

## Prerequisites

Ensure you have installed:

1. **Visual Studio 2019 or later** with:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10/11 SDK

2. **LLVM/Clang** (download from https://releases.llvm.org/)

3. **CMake** (3.20 or later)

4. **Ninja** (optional, but recommended for faster builds)

## Troubleshooting

### Error: "Visual Studio not found"
- Install Visual Studio with C++ development workload
- Ensure MSVC build tools are installed

### Error: "Windows SDK not found"
- Install Windows 10/11 SDK through Visual Studio installer
- Or download standalone SDK from Microsoft

### Error: "clang not found"
- Add LLVM bin directory to PATH
- Verify installation: `clang --version`

### Error: "ninja not found"
- Install Ninja: `winget install Ninja-build.Ninja`
- Or use Visual Studio generator: `-G "Visual Studio 17 2022"`

### Linking still fails
- Try using Microsoft linker instead of lld-link
- Check that all required libraries are present in LIB paths
- Verify MSVC tools version matches your installation

## Manual Verification

To verify your setup manually:

```cmd
# Check Clang
clang --version

# Check linker
lld-link --version
link.exe /?

# Check library paths
dir "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
dir "C:\Program Files (x86)\Windows Kits\10\Lib"
```

## Additional Resources

- [LLVM on Windows Documentation](https://llvm.org/docs/GettingStartedVS.html)
- [Clang MSVC Compatibility](https://clang.llvm.org/docs/MSVCCompatibility.html)
- [CMake Windows Support](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-windows)

## Support

If you continue to experience issues:

1. Check the exact error messages in the build output
2. Verify all prerequisites are installed
3. Try the different build strategies manually
4. Consider using Visual Studio's integrated Clang support as an alternative