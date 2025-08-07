@echo off
echo 🚀 ChatGPT+ Clone - Build Script
echo =================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Please run install.ps1 first.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo 📦 Installing PyInstaller...
    pip install pyinstaller
)

REM Create build directory
if not exist "build" mkdir build
if not exist "dist" mkdir dist

echo 🔧 Building executable...

REM Create PyInstaller spec file
echo Creating PyInstaller specification...
(
echo # -*- mode: python ; coding: utf-8 -*-
echo.
echo block_cipher = None
echo.
echo a = Analysis(
echo     ['main.py'],
echo     pathex=[],
echo     binaries=[],
echo     datas=[
echo         ('config.json', '.'),
echo         ('requirements*.txt', '.'),
echo         ('plugins', 'plugins'),
echo         ('workspace', 'workspace'),
echo         ('memory', 'memory'),
echo         ('ui', 'ui'),
echo         ('llm', 'llm'),
echo         ('tools', 'tools'),
echo         ('vs_code_link', 'vs_code_link'),
echo     ],
echo     hiddenimports=[
echo         'PyQt6',
echo         'PyQt6.QtCore',
echo         'PyQt6.QtGui',
echo         'PyQt6.QtWidgets',
echo         'PyQt6.QtWebEngine',
echo         'PyQt6.QtOpenGLWidgets',
echo         'ollama',
echo         'transformers',
echo         'torch',
echo         'torchaudio',
echo         'whisper',
echo         'sounddevice',
echo         'numpy',
echo         'diffusers',
echo         'opencv-python',
echo         'Pillow',
echo         'requests',
echo         'beautifulsoup4',
echo         'playwright',
echo         'selenium',
echo         'chromadb',
echo         'sentence_transformers',
echo         'faiss',
echo         'aiofiles',
echo         'watchdog',
echo         'keyboard',
echo         'psutil',
echo         'python-dotenv',
echo         'tqdm',
echo         'rich',
echo         'jupyter',
echo         'ipykernel',
echo         'virtualenv',
echo         'importlib_metadata',
echo         'pluggy',
echo         'restrictedpython',
echo         'dill',
echo         'cloudpickle',
echo         'pydantic',
echo         'entrypoints',
echo         'setuptools',
echo     ],
echo     hookspath=[],
echo     hooksconfig={},
echo     runtime_hooks=[],
echo     excludes=[],
echo     win_no_prefer_redirects=False,
echo     win_private_assemblies=False,
echo     cipher=block_cipher,
echo     noarchive=False,
echo ^)
echo.
echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher^)
echo.
echo exe = EXE(
echo     pyz,
echo     a.scripts,
echo     a.binaries,
echo     a.zipfiles,
echo     a.datas,
echo     [],
echo     name='ChatGPTPlusClone',
echo     debug=False,
echo     bootloader_ignore_signals=False,
echo     strip=False,
echo     upx=True,
echo     upx_exclude=[],
echo     runtime_tmpdir=None,
echo     console=False,
echo     disable_windowed_traceback=False,
echo     argv_emulation=False,
echo     target_arch=None,
echo     codesign_identity=None,
echo     entitlements_file=None,
echo     icon='icon.ico',
echo ^)
echo.
echo coll = COLLECT(
echo     exe,
echo     a.binaries,
echo     a.zipfiles,
echo     a.datas,
echo     strip=False,
echo     upx=True,
echo     upx_exclude=[],
echo     name='ChatGPTPlusClone',
echo ^)
) > ChatGPTPlusClone.spec

REM Build the executable
echo Building with PyInstaller...
pyinstaller --clean ChatGPTPlusClone.spec

if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

REM Create distribution package
echo 📦 Creating distribution package...

REM Create dist directory structure
if not exist "dist\ChatGPTPlusClone" mkdir dist\ChatGPTPlusClone

REM Copy additional files
echo Copying additional files...
copy "config.json" "dist\ChatGPTPlusClone\" 2>nul
copy "requirements*.txt" "dist\ChatGPTPlusClone\" 2>nul
copy "README.md" "dist\ChatGPTPlusClone\" 2>nul
copy "LICENSE" "dist\ChatGPTPlusClone\" 2>nul

REM Create startup script for the executable
echo Creating startup script...
(
echo @echo off
echo echo 🚀 Starting ChatGPT+ Clone...
echo echo.
echo cd /d "%%~dp0"
echo ChatGPTPlusClone.exe
echo pause
) > "dist\ChatGPTPlusClone\start.bat"

REM Create installer script
echo Creating installer script...
(
echo @echo off
echo echo 🚀 ChatGPT+ Clone - Installer
echo echo ================================
echo echo.
echo echo Installing ChatGPT+ Clone...
echo echo.
echo.
echo REM Create desktop shortcut
echo set SCRIPT="%%TEMP%%\%random%.vbs"
echo echo Set oWS = WScript.CreateObject^("WScript.Shell"^) ^> %%SCRIPT%%
echo echo sLinkFile = "%%USERPROFILE%%\Desktop\ChatGPT+ Clone.lnk" ^>^> %%SCRIPT%%
echo echo Set oLink = oWS.CreateShortcut^(sLinkFile^) ^>^> %%SCRIPT%%
echo echo oLink.TargetPath = "%%~dp0ChatGPTPlusClone.exe" ^>^> %%SCRIPT%%
echo echo oLink.WorkingDirectory = "%%~dp0" ^>^> %%SCRIPT%%
echo echo oLink.Description = "ChatGPT+ Clone - AI Assistant" ^>^> %%SCRIPT%%
echo echo oLink.IconLocation = "%%~dp0ChatGPTPlusClone.exe,0" ^>^> %%SCRIPT%%
echo echo oLink.Save ^>^> %%SCRIPT%%
echo echo cscript //nologo %%SCRIPT%% ^>nul
echo echo del %%SCRIPT%%
echo.
echo REM Create start menu shortcut
echo set SCRIPT="%%TEMP%%\%random%.vbs"
echo echo Set oWS = WScript.CreateObject^("WScript.Shell"^) ^> %%SCRIPT%%
echo echo sLinkFile = "%%APPDATA%%\Microsoft\Windows\Start Menu\Programs\ChatGPT+ Clone.lnk" ^>^> %%SCRIPT%%
echo echo Set oLink = oWS.CreateShortcut^(sLinkFile^) ^>^> %%SCRIPT%%
echo echo oLink.TargetPath = "%%~dp0ChatGPTPlusClone.exe" ^>^> %%SCRIPT%%
echo echo oLink.WorkingDirectory = "%%~dp0" ^>^> %%SCRIPT%%
echo echo oLink.Description = "ChatGPT+ Clone - AI Assistant" ^>^> %%SCRIPT%%
echo echo oLink.IconLocation = "%%~dp0ChatGPTPlusClone.exe,0" ^>^> %%SCRIPT%%
echo echo oLink.Save ^>^> %%SCRIPT%%
echo echo cscript //nologo %%SCRIPT%% ^>nul
echo echo del %%SCRIPT%%
echo.
echo echo ✅ Installation completed!
echo echo.
echo echo 🎉 ChatGPT+ Clone has been installed successfully!
echo echo.
echo echo 📋 Next steps:
echo echo 1. Double-click "ChatGPT+ Clone.exe" to start the application
echo echo 2. Press Ctrl+Shift+V to activate voice input
echo echo 3. Use the tools panel to access different AI capabilities
echo echo.
echo echo 📚 For more information, check the README.md file
echo echo.
echo pause
) > "dist\ChatGPTPlusClone\install.bat"

REM Create README for distribution
echo Creating distribution README...
(
echo # ChatGPT+ Clone - Standalone Application
echo.
echo ## Quick Start
echo.
echo 1. **Install**: Run `install.bat` to install the application
echo 2. **Launch**: Double-click `ChatGPTPlusClone.exe` or use the desktop shortcut
echo 3. **Voice**: Press `Ctrl+Shift+V` to activate voice input
echo 4. **Tools**: Use the tools panel for different AI capabilities
echo.
echo ## Features
echo.
echo - 🤖 **AI Chat**: Advanced language model conversations
echo - 🎤 **Voice Input**: Speech-to-text with Whisper
echo - 💻 **Code Interpreter**: Python code execution
echo - 🌐 **Web Search**: Real-time web browsing
echo - 🎨 **Image Generation**: DALL-E style image creation
echo - 📁 **File Handling**: Upload and process files
echo - 🔌 **Plugin System**: Extensible with custom plugins
echo - 🧠 **Memory System**: Persistent conversation history
echo.
echo ## System Requirements
echo.
echo - Windows 10/11 (64-bit)
echo - 8GB RAM minimum (16GB recommended)
echo - 10GB free disk space
echo - Internet connection for web features
echo.
echo ## Troubleshooting
echo.
echo - **Voice not working**: Check microphone permissions
echo - **Slow performance**: Close other applications
echo - **Model errors**: Ensure Ollama is running
echo.
echo ## Support
echo.
echo For issues and updates, visit the project repository.
echo.
echo ---
echo Built with ❤️ using PyQt6 and Ollama
) > "dist\ChatGPTPlusClone\README.txt"

echo.
echo ✅ Build completed successfully!
echo.
echo 📦 Distribution package created in: dist\ChatGPTPlusClone\
echo.
echo 🚀 To install:
echo 1. Navigate to dist\ChatGPTPlusClone\
echo 2. Run install.bat
echo 3. Launch ChatGPTPlusClone.exe
echo.
echo 📋 Files created:
echo - ChatGPTPlusClone.exe (main executable)
echo - start.bat (startup script)
echo - install.bat (installer)
echo - README.txt (documentation)
echo - config.json (configuration)
echo.
pause