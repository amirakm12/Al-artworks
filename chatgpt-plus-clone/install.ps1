# ChatGPT+ Clone - Windows Installation Script
# One-command installation for all dependencies and setup

param(
    [switch]$SkipPython,
    [switch]$SkipOllama,
    [switch]$SkipModels,
    [switch]$DevMode,
    [string]$PythonVersion = "3.10"
)

Write-Host "🚀 ChatGPT+ Clone - Installation Script" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "⚠️  Warning: Not running as administrator. Some features may not work properly." -ForegroundColor Yellow
}

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to download file
function Download-File($url, $output) {
    try {
        Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
        return $true
    }
    catch {
        Write-Host "❌ Error downloading $url" -ForegroundColor Red
        return $false
    }
}

# Check and install Python
if (-not $SkipPython) {
    Write-Host "🐍 Checking Python installation..." -ForegroundColor Green
    
    if (-not (Test-Command "python")) {
        Write-Host "Python not found. Installing Python $PythonVersion..." -ForegroundColor Yellow
        
        # Download Python installer
        $pythonUrl = "https://www.python.org/ftp/python/$PythonVersion.0/python-$PythonVersion.0-amd64.exe"
        $pythonInstaller = "$env:TEMP\python-installer.exe"
        
        if (Download-File $pythonUrl $pythonInstaller) {
            Write-Host "Installing Python..." -ForegroundColor Yellow
            Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
            
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            Write-Host "✅ Python installed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Failed to download Python installer" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "✅ Python already installed" -ForegroundColor Green
    }
}

# Check and install Git
if (-not (Test-Command "git")) {
    Write-Host "📦 Installing Git..." -ForegroundColor Yellow
    
    # Download Git installer
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.40.0.windows.1/Git-2.40.0-64-bit.exe"
    $gitInstaller = "$env:TEMP\git-installer.exe"
    
    if (Download-File $gitUrl $gitInstaller) {
        Write-Host "Installing Git..." -ForegroundColor Yellow
        Start-Process -FilePath $gitInstaller -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "✅ Git installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Failed to download Git installer" -ForegroundColor Red
    }
}

# Check and install FFmpeg
if (-not (Test-Command "ffmpeg")) {
    Write-Host "🎵 Installing FFmpeg..." -ForegroundColor Yellow
    
    # Download FFmpeg
    $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    $ffmpegZip = "$env:TEMP\ffmpeg.zip"
    $ffmpegDir = "$env:PROGRAMFILES\ffmpeg"
    
    if (Download-File $ffmpegUrl $ffmpegZip) {
        # Extract FFmpeg
        Expand-Archive -Path $ffmpegZip -DestinationPath "$env:TEMP\ffmpeg" -Force
        
        # Copy to Program Files
        if (-not (Test-Path $ffmpegDir)) {
            New-Item -ItemType Directory -Path $ffmpegDir -Force
        }
        
        $extractedDir = Get-ChildItem "$env:TEMP\ffmpeg" -Directory | Select-Object -First 1
        Copy-Item "$($extractedDir.FullName)\bin\*" $ffmpegDir -Recurse -Force
        
        # Add to PATH
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($currentPath -notlike "*$ffmpegDir*") {
            [Environment]::SetEnvironmentVariable("Path", "$currentPath;$ffmpegDir", "Machine")
            $env:Path = "$env:Path;$ffmpegDir"
        }
        
        Write-Host "✅ FFmpeg installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Failed to download FFmpeg" -ForegroundColor Red
    }
}

# Install and setup Ollama
if (-not $SkipOllama) {
    Write-Host "🤖 Installing Ollama..." -ForegroundColor Green
    
    if (-not (Test-Command "ollama")) {
        # Download Ollama installer
        $ollamaUrl = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.msi"
        $ollamaInstaller = "$env:TEMP\ollama-installer.msi"
        
        if (Download-File $ollamaUrl $ollamaInstaller) {
            Write-Host "Installing Ollama..." -ForegroundColor Yellow
            Start-Process -FilePath "msiexec" -ArgumentList "/i", $ollamaInstaller, "/quiet" -Wait
            
            Write-Host "✅ Ollama installed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "❌ Failed to download Ollama installer" -ForegroundColor Red
        }
    }
    else {
        Write-Host "✅ Ollama already installed" -ForegroundColor Green
    }
}

# Create virtual environment
Write-Host "🔧 Setting up Python virtual environment..." -ForegroundColor Green

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}
else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Green

# Install core dependencies first
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
pip install -r requirements-core.txt

# Install voice dependencies
Write-Host "Installing voice processing dependencies..." -ForegroundColor Yellow
pip install -r requirements-voice.txt

# Install image generation dependencies
Write-Host "Installing image generation dependencies..." -ForegroundColor Yellow
pip install -r requirements-image.txt

# Install development dependencies
if ($DevMode) {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
}

# Install plugin system dependencies
Write-Host "Installing plugin system dependencies..." -ForegroundColor Yellow
pip install -r requirements-plugins.txt

# Install remaining dependencies
Write-Host "Installing remaining dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install Playwright browsers
Write-Host "🌐 Installing Playwright browsers..." -ForegroundColor Yellow
playwright install

# Download AI models
if (-not $SkipModels) {
    Write-Host "🤖 Downloading AI models..." -ForegroundColor Green
    
    # Pull Ollama models
    if (Test-Command "ollama") {
        Write-Host "Downloading dolphin-mixtral:8x22b model..." -ForegroundColor Yellow
        ollama pull dolphin-mixtral:8x22b
        
        Write-Host "Downloading llama2:13b model..." -ForegroundColor Yellow
        ollama pull llama2:13b
        
        Write-Host "Downloading mistral:7b model..." -ForegroundColor Yellow
        ollama pull mistral:7b
    }
}

# Create necessary directories
Write-Host "📁 Creating project directories..." -ForegroundColor Green

$directories = @(
    "workspace",
    "memory",
    "plugins",
    "logs",
    "temp"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    }
}

# Create sample plugin
Write-Host "🔌 Creating sample plugin..." -ForegroundColor Green

$samplePluginDir = "plugins\sample_plugin"
if (-not (Test-Path $samplePluginDir)) {
    New-Item -ItemType Directory -Path $samplePluginDir -Force
    
    # Create manifest
    $manifest = @{
        name = "sample_plugin"
        version = "1.0.0"
        description = "A sample plugin for ChatGPT+ Clone"
        author = "Plugin Developer"
        dependencies = @()
        hooks = @("message_received", "tool_executed")
        permissions = @("read_files", "execute_code")
    }
    
    $manifest | ConvertTo-Json -Depth 3 | Out-File "$samplePluginDir\manifest.json" -Encoding UTF8
    
    # Create plugin code
    $pluginCode = @"
"""
Sample Plugin - Example plugin for ChatGPT+ Clone
"""

def register(plugin_manager):
    """Register this plugin with the plugin manager"""
    print("Registering sample plugin...")
    
    # Register hooks
    plugin_manager.register_hook("message_received", handle_message)
    plugin_manager.register_hook("tool_executed", handle_tool)

def unregister(plugin_manager):
    """Unregister this plugin"""
    print("Unregistering sample plugin...")

def handle_message(message):
    """Handle incoming messages"""
    return f"Sample plugin processed: {message}"

def handle_tool(tool_name, result):
    """Handle tool execution results"""
    return f"Sample plugin handled tool: {tool_name}"
"@
    
    $pluginCode | Out-File "$samplePluginDir\plugin.py" -Encoding UTF8
    
    Write-Host "✅ Created sample plugin" -ForegroundColor Green
}

# Create configuration file
Write-Host "⚙️  Creating configuration file..." -ForegroundColor Green

$config = @{
    app_name = "ChatGPT+ Clone"
    version = "1.0.0"
    default_model = "dolphin-mixtral:8x22b"
    voice_hotkey = "ctrl+shift+v"
    max_memory_size = "1GB"
    enable_plugins = $true
    enable_voice = $true
    enable_image_generation = $true
    enable_code_execution = $true
    enable_web_search = $true
    log_level = "INFO"
    workspace_path = "workspace"
    memory_path = "memory"
    plugins_path = "plugins"
}

$config | ConvertTo-Json -Depth 3 | Out-File "config.json" -Encoding UTF8

# Create batch file for easy startup
Write-Host "🚀 Creating startup script..." -ForegroundColor Green

$startupScript = @"
@echo off
echo Starting ChatGPT+ Clone...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start the application
python main.py

pause
"@

$startupScript | Out-File "start.bat" -Encoding ASCII

# Create PowerShell startup script
$psStartupScript = @"
# ChatGPT+ Clone - Startup Script
Write-Host "🚀 Starting ChatGPT+ Clone..." -ForegroundColor Green

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Start the application
python main.py
"@

$psStartupScript | Out-File "start.ps1" -Encoding UTF8

Write-Host ""
Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Run 'start.bat' or 'start.ps1' to launch the application" -ForegroundColor White
Write-Host "2. Press Ctrl+Shift+V to activate voice input" -ForegroundColor White
Write-Host "3. Use the tools panel to access different AI capabilities" -ForegroundColor White
Write-Host "4. Check the 'plugins' directory to add custom plugins" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Development mode:" -ForegroundColor Cyan
Write-Host "- Run 'python main.py' directly for development" -ForegroundColor White
Write-Host "- Check 'logs' directory for application logs" -ForegroundColor White
Write-Host "- Modify 'config.json' to customize settings" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "- Check README.md for detailed usage instructions" -ForegroundColor White
Write-Host "- Visit the project repository for updates" -ForegroundColor White
Write-Host ""
Write-Host "✨ Enjoy your ChatGPT+ Clone!" -ForegroundColor Green