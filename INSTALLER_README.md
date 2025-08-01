# Standalone C++ Development Tools Installers

## 🚀 **Available Installers**

### **1. Full System Installer** (`standalone_installer.bat`)
**For:** Complete system installation with admin rights
**Features:**
- Installs MSYS2 (GCC compiler)
- Installs CMake (build system)
- Installs Git (version control)
- Adds tools to system PATH
- Creates project templates

**Usage:**
```bash
# Right-click and "Run as administrator"
standalone_installer.bat
```

### **2. Portable Installer** (`portable_installer.bat`)
**For:** Portable installation without admin rights
**Features:**
- Downloads portable tools
- Creates local environment
- No system modifications
- Project templates included

**Usage:**
```bash
# Run normally (no admin needed)
portable_installer.bat
```

## 📦 **What Each Installer Does**

### **Full System Installer:**
1. **Downloads** official installers from GitHub
2. **Installs** MSYS2, CMake, Git system-wide
3. **Configures** PATH environment variables
4. **Tests** all installations
5. **Creates** project templates

### **Portable Installer:**
1. **Creates** local directory structure
2. **Downloads** portable CMake
3. **Sets up** local environment
4. **Creates** project templates
5. **No system changes** required

## 🛠️ **Installation Options**

### **Option A: Full Installation (Recommended)**
```bash
# Run as Administrator
standalone_installer.bat
```
**Result:** Complete C++ development environment

### **Option B: Portable Installation**
```bash
# Run normally
portable_installer.bat
```
**Result:** Portable tools in user directory

### **Option C: Manual Installation**
Follow `MANUAL_INSTALL_GUIDE.md` for step-by-step instructions

## ✅ **After Installation**

### **For Full Installation:**
1. **Restart** terminal/VS Code
2. **Test** with: `gcc --version`, `cmake --version`, `git --version`
3. **Build** with: `Ctrl+Shift+B` in VS Code
4. **Debug** with: `F5` in VS Code

### **For Portable Installation:**
1. **Run** `setup_env.bat` in the portable directory
2. **Copy** project template to your workspace
3. **Build** with: `build.bat`
4. **Use** portable tools without system installation

## 🔧 **Project Templates**

### **Full Installation Template:**
- Location: `%USERPROFILE%\Documents\CppProjectTemplate`
- Includes: CMakeLists.txt, build.bat, sample code

### **Portable Template:**
- Location: `%USERPROFILE%\CppDevTools\projects\template`
- Includes: Complete project structure

## 🚨 **Troubleshooting**

### **If Full Installer Fails:**
1. **Check** administrator rights
2. **Disable** antivirus temporarily
3. **Ensure** internet connection
4. **Try** portable installer instead

### **If Portable Installer Fails:**
1. **Check** disk space
2. **Ensure** internet connection
3. **Try** manual installation
4. **Use** existing tools if available

## 📁 **File Structure**

### **Full Installation:**
```
C:\msys64\          # MSYS2 installation
C:\Program Files\CMake\  # CMake installation
%USERPROFILE%\Documents\CppProjectTemplate\  # Templates
```

### **Portable Installation:**
```
%USERPROFILE%\CppDevTools\
├── bin\            # Portable tools
├── include\        # Headers
├── lib\           # Libraries
└── projects\      # Templates
    └── template\  # Sample project
```

## 🎯 **Quick Start**

### **For New Users:**
1. **Run** `standalone_installer.bat` as Administrator
2. **Restart** VS Code
3. **Press** `F5` to start debugging
4. **Edit** code in `src/main.cpp`

### **For Advanced Users:**
1. **Run** `portable_installer.bat`
2. **Copy** template to your project
3. **Run** `setup_env.bat`
4. **Build** with `build.bat`

## 🆘 **Need Help?**

- **Check** `MANUAL_INSTALL_GUIDE.md` for manual steps
- **Use** `installation_helper.bat` for guided installation
- **Review** error messages for specific issues
- **Try** different installer options

## ✅ **Success Indicators**

- `gcc --version` shows version info
- `cmake --version` shows version info
- `git --version` shows version info
- `Ctrl+Shift+B` builds your project
- `F5` starts debugging
- No "command not found" errors 