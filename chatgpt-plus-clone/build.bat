@echo off
echo ========================================
echo ChatGPT+ Clone - Build Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

REM Check if PyInstaller is available
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo Creating build directory...
if not exist "build" mkdir build
if not exist "dist" mkdir dist

echo Cleaning previous builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo.
echo ========================================
echo Building ChatGPT+ Clone Executable
echo ========================================
echo.

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
echo         ('plugins/*.py', 'plugins'),
echo         ('ui/*.py', 'ui'),
echo         ('*.py', '.'),
echo     ],
echo     hiddenimports=[
echo         'PyQt6',
echo         'PyQt6.QtCore',
echo         'PyQt6.QtWidgets',
echo         'PyQt6.QtGui',
echo         'PyQt6.QtWebEngineWidgets',
echo         'PyQt6.QtOpenGLWidgets',
echo         'keyboard',
echo         'watchdog',
echo         'restricted_python',
echo         'config_manager',
echo         'plugin_loader',
echo         'voice_hotkey',
echo         'overlay_ar',
echo         'error_handler',
echo         'ui.settings_dialog',
echo         'ui.plugin_test_dialog',
echo         'ui.chat_interface',
echo         'ui.tools_panel',
echo         'ui.file_browser',
echo         'ui.voice_panel',
echo         'ui.monaco_editor',
echo         'llm.agent_orchestrator',
echo         'memory.memory_manager',
echo         'tools.code_executor',
echo         'tools.web_browser',
echo         'tools.image_editor',
echo         'tools.voice_agent',
echo         'vs_code_link.vs_code_integration',
echo     ],
echo     hookspath=[],
echo     hooksconfig={},
echo     runtime_hooks=[],
echo     excludes=[],
echo     win_no_prefer_redirects=False,
echo     win_private_assemblies=False,
echo     cipher=block_cipher,
echo     noarchive=False,
echo )
echo.
echo pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
echo.
echo exe = EXE(
echo     pyz,
echo     a.scripts,
echo     [],
echo     exclude_binaries=True,
echo     name='ChatGPTPlusClone',
echo     debug=False,
echo     bootloader_ignore_signals=False,
echo     strip=False,
echo     upx=True,
echo     console=False,
echo     disable_windowed_traceback=False,
echo     argv_emulation=False,
echo     target_arch=None,
echo     codesign_identity=None,
echo     entitlements_file=None,
echo     icon='app_icon.ico' if os.path.exists('app_icon.ico') else None,
echo )
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
echo )
) > ChatGPTPlusClone.spec

echo Building executable with PyInstaller...
pyinstaller --clean ChatGPTPlusClone.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Completed Successfully!
echo ========================================
echo.
echo Executable location: dist\ChatGPTPlusClone\ChatGPTPlusClone.exe
echo.
echo Creating installation package...

REM Create installation directory
if not exist "installer" mkdir installer

REM Copy executable and dependencies
xcopy /E /I /Y "dist\ChatGPTPlusClone" "installer\ChatGPTPlusClone"

REM Create startup script
echo @echo off > installer\start.bat
echo echo Starting ChatGPT+ Clone... >> installer\start.bat
echo cd /d "%%~dp0" >> installer\start.bat
echo ChatGPTPlusClone.exe >> installer\start.bat
echo pause >> installer\start.bat

REM Create README for installer
echo ChatGPT+ Clone - AI Assistant > installer\README.txt
echo. >> installer\README.txt
echo Installation Instructions: >> installer\README.txt
echo 1. Extract all files to a directory >> installer\README.txt
echo 2. Run start.bat to launch the application >> installer\README.txt
echo 3. Or double-click ChatGPTPlusClone.exe >> installer\README.txt
echo. >> installer\README.txt
echo System Requirements: >> installer\README.txt
echo - Windows 10 or later >> installer\README.txt
echo - 4GB RAM minimum >> installer\README.txt
echo - 2GB free disk space >> installer\README.txt
echo. >> installer\README.txt
echo Features: >> installer\README.txt
echo - Voice hotkey support (Ctrl+Shift+V) >> installer\README.txt
echo - Plugin system with sandboxing >> installer\README.txt
echo - AR overlay interface >> installer\README.txt
echo - Live configuration toggles >> installer\README.txt
echo - Comprehensive error handling >> installer\README.txt

echo.
echo ========================================
echo Installation Package Created!
echo ========================================
echo.
echo Package location: installer\ChatGPTPlusClone\
echo.
echo To distribute:
echo 1. Zip the installer\ChatGPTPlusClone\ folder
echo 2. Share the zip file with users
echo.
echo Testing the build...
echo.

REM Test the executable
if exist "dist\ChatGPTPlusClone\ChatGPTPlusClone.exe" (
    echo Build test: SUCCESS
    echo Executable file exists and is ready for distribution
) else (
    echo Build test: FAILED
    echo Executable file not found
)

echo.
echo ========================================
echo Build process completed!
echo ========================================
echo.
pause