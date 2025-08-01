# Manual C++ Development Tools Installation Guide

If the automated installer doesn't work, follow this manual installation guide.

## 🛠️ Required Tools

### 1. MSYS2 (MinGW-w64) - C++ Compiler
**Download:** https://www.msys2.org/
**Installation:**
1. Download the installer
2. Run as Administrator
3. Install to `C:\msys64\`
4. Add `C:\msys64\mingw64\bin` to your PATH

### 2. CMake - Build System
**Download:** https://cmake.org/download/
**Installation:**
1. Download the Windows installer
2. Run as Administrator
3. Choose "Add CMake to system PATH"
4. Install to default location

### 3. Git - Version Control
**Download:** https://git-scm.com/
**Installation:**
1. Download the installer
2. Run as Administrator
3. Use default settings
4. Choose "Git from the command line and also from 3rd-party software"

### 4. Visual Studio Build Tools (Optional)
**Download:** https://visualstudio.microsoft.com/downloads/
**Installation:**
1. Download "Build Tools for Visual Studio 2022"
2. Run as Administrator
3. Select "C++ build tools" workload
4. Install

## 🔧 After Installation

### 1. Restart Your Terminal/VS Code
Close and reopen your terminal or VS Code to refresh the PATH.

### 2. Test Installation
Open a new terminal and run:
```bash
gcc --version
cmake --version
git --version
```

### 3. Build Your Project
- Press `Ctrl+Shift+B` in VS Code
- Or run `.\build.bat` from command line
- Or press `F5` to debug

## 🚨 Troubleshooting

### If you get "command not found":
1. Restart your terminal/VS Code
2. Check if the tool is in your PATH
3. Try running the installer as Administrator

### If build fails:
1. Make sure you have a C++ compiler installed
2. Check that the compiler is in your PATH
3. Try the manual installation steps above

### If VS Code can't find the compiler:
1. Update the paths in `.vscode/c_cpp_properties.json`
2. Make sure the compiler is installed and in PATH
3. Restart VS Code

## 📁 Project Structure
```
YourProject/
├── src/
│   └── main.cpp          # Your C++ source code
├── include/              # Header files
├── build/               # Build output
├── .vscode/            # VS Code configuration
│   ├── launch.json     # Debug configurations
│   ├── tasks.json      # Build tasks
│   └── c_cpp_properties.json
├── CMakeLists.txt      # CMake configuration
└── build.bat          # Build script
```

## ✅ Success Indicators
- `gcc --version` shows version info
- `cmake --version` shows version info
- `Ctrl+Shift+B` builds your project
- `F5` starts debugging
- No "command not found" errors

## 🆘 Still Having Issues?
1. Try running installers as Administrator
2. Check Windows Defender/Antivirus isn't blocking
3. Make sure you have enough disk space
4. Try the manual installation steps above 