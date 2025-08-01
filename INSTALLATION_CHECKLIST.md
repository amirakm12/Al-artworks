# C++ Development Tools Installation Checklist

## ✅ **Step-by-Step Manual Installation**

### **Step 1: MSYS2 (C++ Compiler)**
- [ ] Go to: https://www.msys2.org/
- [ ] Download the installer
- [ ] Right-click → "Run as administrator"
- [ ] Install to: `C:\msys64\`
- [ ] Complete installation

### **Step 2: CMake (Build System)**
- [ ] Go to: https://cmake.org/download/
- [ ] Download "Windows x64 Installer"
- [ ] Right-click → "Run as administrator"
- [ ] Choose: "Add CMake to system PATH for all users"
- [ ] Complete installation

### **Step 3: Git (Version Control)**
- [ ] Go to: https://git-scm.com/
- [ ] Download Windows installer
- [ ] Right-click → "Run as administrator"
- [ ] Use default settings (click Next)
- [ ] Choose: "Git from the command line and also from 3rd-party software"

### **Step 4: Test Installation**
After installing all tools:
- [ ] Restart your terminal/VS Code
- [ ] Open new terminal
- [ ] Run: `gcc --version`
- [ ] Run: `cmake --version`
- [ ] Run: `git --version`

### **Step 5: Build Your Project**
- [ ] Press `Ctrl+Shift+B` in VS Code
- [ ] Or run: `.\build.bat`
- [ ] Or press `F5` to debug

## 🔍 **Verification Commands**

After installation, run these commands to verify:

```bash
# Test C++ compiler
gcc --version

# Test build system
cmake --version

# Test version control
git --version

# Test building your project
.\build.bat
```

## 🚨 **If Commands Don't Work**

1. **Restart your terminal/VS Code**
2. **Check if tools are in PATH:**
   - MSYS2: `C:\msys64\mingw64\bin`
   - CMake: Usually auto-added
   - Git: Usually auto-added

3. **Manual PATH addition:**
   - Open System Properties → Environment Variables
   - Add `C:\msys64\mingw64\bin` to PATH

## 📁 **Expected Project Structure**

After installation, you should have:
```
YourProject/
├── src/
│   └── main.cpp          # Your C++ code
├── build/
│   └── debug/           # Build output
├── .vscode/            # VS Code config
├── CMakeLists.txt      # CMake config
└── build.bat          # Build script
```

## ✅ **Success Indicators**

- [ ] `gcc --version` shows version info
- [ ] `cmake --version` shows version info
- [ ] `git --version` shows version info
- [ ] `Ctrl+Shift+B` builds your project
- [ ] `F5` starts debugging
- [ ] No "command not found" errors

## 🆘 **Need Help?**

If any step fails:
1. Make sure you ran installers as Administrator
2. Check Windows Defender isn't blocking
3. Ensure you have enough disk space
4. Try restarting your computer after installation 