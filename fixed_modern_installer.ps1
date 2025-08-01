# Fixed Modern Al-artworks Installer
# PowerShell-based installer with advanced UI and comprehensive error handling

param(
    [switch]$Silent,
    [switch]$SkipDependencies,
    [string]$InstallPath = "C:\Al-artworks"
)

# Set console encoding and colors
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Al-artworks Fixed Modern Installer"

# Modern UI Functions
function Write-Header {
    param([string]$Title)
    Write-Host "`n" -NoNewline
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                    $Title" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Step, [string]$Description)
    Write-Host "🔧 $Step" -ForegroundColor Yellow
    Write-Host "   $Description" -ForegroundColor Gray
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Show-Progress {
    param([string]$Activity, [int]$PercentComplete)
    Write-Progress -Activity $Activity -PercentComplete $PercentComplete -Status "Installing..."
}

# Check for admin privileges
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Download function with progress and retry
function Invoke-DownloadWithProgress {
    param([string]$Uri, [string]$OutFile, [string]$Description)
    
    try {
        Write-Host "📥 $Description..." -ForegroundColor Blue
        
        # Create directory if it doesn't exist
        $directory = Split-Path $OutFile -Parent
        if (!(Test-Path $directory)) {
            New-Item -ItemType Directory -Path $directory -Force | Out-Null
        }
        
        # Set TLS 1.2 for compatibility
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Uri, $OutFile)
        
        if (Test-Path $OutFile) {
            Write-Success "$Description downloaded successfully"
            return $true
        } else {
            Write-Error "Download completed but file not found"
            return $false
        }
    }
    catch {
        Write-Error "Failed to download $Description`: $($_.Exception.Message)"
        return $false
    }
}

# Kill process function
function Stop-ProcessSafely {
    param([string]$ProcessName)
    try {
        Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    } catch {
        # Process not found or already stopped
    }
}

# Main installation function
function Install-AlArtworks {
    Write-Header "AL-ARTWORKS FIXED MODERN INSTALLER"
    Write-Host "🎨 AI-Powered Art Creation Suite" -ForegroundColor Magenta
    Write-Host "🚀 Modern Development Environment" -ForegroundColor Magenta
    Write-Host "⚡ Lightning-Fast Installation" -ForegroundColor Magenta
    Write-Host "🔧 Fixed and Robust" -ForegroundColor Magenta
    Write-Host ""

    # Check admin privileges
    if (-not (Test-Administrator)) {
        Write-Error "This installer requires administrator privileges"
        Write-Host "Please run PowerShell as administrator and try again" -ForegroundColor Red
        pause
        exit 1
    }

    Write-Success "Administrator privileges confirmed"
    Write-Host ""

    # Create installation directories
    Write-Step "Creating Installation Directories" "Setting up project structure"
    
    $directories = @(
        $InstallPath,
        "$InstallPath\tools",
        "$InstallPath\projects",
        "$InstallPath\logs",
        "$InstallPath\config"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    Write-Success "Directories created successfully"
    Write-Host ""

    # Installation steps with proper error handling
    $steps = @(
        @{Name="MSYS2"; Description="C++ Compiler"; Check="gcc"},
        @{Name="CMake"; Description="Build System"; Check="cmake"},
        @{Name="Git"; Description="Version Control"; Check="git"}
    )

    $stepNumber = 1
    foreach ($step in $steps) {
        Write-Step "Step $stepNumber`: Installing $($step.Name)" $step.Description
        
        # Check if command exists
        try {
            $null = Get-Command $step.Check -ErrorAction Stop
            Write-Success "$($step.Name) already installed"
            continue
        }
        catch {
            # Install the tool
            Install-Tool -ToolName $step.Name
        }
        
        $stepNumber++
        Write-Host ""
    }

    # Install Python dependencies
    Write-Step "Step 4: Installing Python Dependencies" "AI/ML Libraries"
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python detected: $pythonVersion"
            
            $packages = @(
                @{Name="Core"; Packages=@("numpy", "opencv-python", "pillow", "matplotlib", "seaborn")},
                @{Name="PyTorch"; Packages=@("torch", "torchvision", "torchaudio"); Index="https://download.pytorch.org/whl/cpu"},
                @{Name="AI/ML"; Packages=@("transformers", "diffusers", "accelerate")},
                @{Name="Web"; Packages=@("flask", "fastapi", "uvicorn")},
                @{Name="Utils"; Packages=@("requests", "beautifulsoup4", "lxml")}
            )

            # Upgrade pip first
            Write-Host "📦 Upgrading pip..." -ForegroundColor Blue
            python -m pip install --upgrade pip --quiet

            foreach ($packageGroup in $packages) {
                Write-Host "📦 Installing $($packageGroup.Name) packages..." -ForegroundColor Blue
                
                foreach ($package in $packageGroup.Packages) {
                    try {
                        if ($packageGroup.Index) {
                            python -m pip install $package --index-url $packageGroup.Index --quiet
                        } else {
                            python -m pip install $package --quiet
                        }
                        
                        if ($LASTEXITCODE -eq 0) {
                            Write-Success "$package installed"
                        } else {
                            Write-Warning "Failed to install $package"
                        }
                    } catch {
                        Write-Warning "Failed to install $package`: $($_.Exception.Message)"
                    }
                }
            }
        } else {
            Write-Warning "Python not found - please install Python 3.8+ manually"
        }
    }
    catch {
        Write-Warning "Python installation check failed`: $($_.Exception.Message)"
    }

    Write-Host ""

    # Create project structure
    Write-Step "Step 5: Creating Project Structure" "Setting up Al-artworks"
    
    $sourcePath = Split-Path $PSScriptRoot -Parent
    $projectPath = "$InstallPath\projects"
    
    # Copy project files with error handling
    if (Test-Path "$sourcePath\Al-artworks") {
        try {
            Copy-Item -Path "$sourcePath\Al-artworks" -Destination "$projectPath\Al-artworks" -Recurse -Force
            Write-Success "Al-artworks project copied"
        } catch {
            Write-Warning "Failed to copy Al-artworks project`: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "Al-artworks directory not found in source location"
    }
    
    if (Test-Path "$sourcePath\aisis") {
        try {
            Copy-Item -Path "$sourcePath\aisis" -Destination "$projectPath\aisis" -Recurse -Force
            Write-Success "Aisis project copied"
        } catch {
            Write-Warning "Failed to copy Aisis project`: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "Aisis directory not found in source location"
    }

    # Create build scripts
    New-BuildScripts -ProjectPath $projectPath
    Write-Success "Build scripts created"

    Write-Host ""

    # Create desktop shortcuts
    Write-Step "Step 6: Creating Desktop Shortcuts" "Quick access icons"
    
    New-DesktopShortcuts -ProjectPath $projectPath
    Write-Success "Desktop shortcuts created"

    Write-Host ""

    # Test installation
    Write-Step "Step 7: Testing Installation" "Verifying tools"
    
    Test-Installation
    Write-Success "Installation tests completed"

    Write-Host ""

    # Installation complete
    Write-Header "INSTALLATION COMPLETE!"
    Write-Host "🎉 Al-artworks has been successfully installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 Installation Location: $InstallPath" -ForegroundColor Cyan
    Write-Host "📁 Projects Location: $projectPath" -ForegroundColor Cyan
    Write-Host "📁 Logs Location: $InstallPath\logs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🚀 Quick Start:" -ForegroundColor Yellow
    Write-Host "   1. Double-click 'Al-artworks' on your desktop" -ForegroundColor White
    Write-Host "   2. Or run: $projectPath\run_ai_artworks.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "🔧 Development Tools:" -ForegroundColor Yellow
    Write-Host "   - MSYS2 (C++ Compiler)" -ForegroundColor White
    Write-Host "   - CMake (Build System)" -ForegroundColor White
    Write-Host "   - Git (Version Control)" -ForegroundColor White
    Write-Host "   - Python Dependencies" -ForegroundColor White
    Write-Host ""
    Write-Host "⚡ Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Restart your terminal/VS Code" -ForegroundColor White
    Write-Host "   2. Open the project in VS Code" -ForegroundColor White
    Write-Host "   3. Press Ctrl+Shift+B to build" -ForegroundColor White
    Write-Host "   4. Start creating AI-powered artwork!" -ForegroundColor White
    Write-Host ""
    Write-Host "🎨 Your AI Art Creation Suite is ready!" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "📋 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   - If tools don't work, restart your terminal" -ForegroundColor White
    Write-Host "   - Check logs in: $InstallPath\logs" -ForegroundColor White
    Write-Host "   - Manual installation guides available in project docs" -ForegroundColor White
    Write-Host ""
}

# Install specific tools with improved error handling
function Install-Tool {
    param([string]$ToolName)
    
    switch ($ToolName) {
        "MSYS2" {
            $installerPath = "$InstallPath\tools\msys2-installer.exe"
            $downloadUrl = "https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe"
            
            # Kill any existing MSYS2 installer processes
            Stop-ProcessSafely -ProcessName "msys2-installer"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "MSYS2 installer") {
                Write-Host "🔄 Installing MSYS2..." -ForegroundColor Blue
                Write-Host "   This may take several minutes..." -ForegroundColor Gray
                
                try {
                    $process = Start-Process -FilePath $installerPath -ArgumentList "--accept-messages", "--accept-licenses", "--root", "C:\msys64", "--noconfirm" -Wait -PassThru
                    
                    if ($process.ExitCode -eq 0) {
                        Write-Success "MSYS2 installed successfully"
                        
                        # Add to PATH
                        $currentPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Machine)
                        if ($currentPath -notlike "*C:\msys64\mingw64\bin*") {
                            $newPath = "$currentPath;C:\msys64\mingw64\bin"
                            [Environment]::SetEnvironmentVariable("PATH", $newPath, [EnvironmentVariableTarget]::Machine)
                            $env:PATH += ";C:\msys64\mingw64\bin"
                        }
                    } else {
                        Write-Error "MSYS2 installation failed with exit code: $($process.ExitCode)"
                    }
                } catch {
                    Write-Error "MSYS2 installation failed`: $($_.Exception.Message)"
                }
            }
        }
        
        "CMake" {
            $installerPath = "$InstallPath\tools\cmake-installer.msi"
            $downloadUrl = "https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "CMake installer") {
                Write-Host "🔄 Installing CMake..." -ForegroundColor Blue
                
                try {
                    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $installerPath, "/quiet", "ADD_TO_PATH=1", "/l*v", "$InstallPath\logs\cmake-install.log" -Wait -PassThru
                    
                    if ($process.ExitCode -eq 0) {
                        Write-Success "CMake installed successfully"
                    } else {
                        Write-Error "CMake installation failed with exit code: $($process.ExitCode)"
                    }
                } catch {
                    Write-Error "CMake installation failed`: $($_.Exception.Message)"
                }
            }
        }
        
        "Git" {
            $installerPath = "$InstallPath\tools\git-installer.exe"
            $downloadUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "Git installer") {
                Write-Host "🔄 Installing Git..." -ForegroundColor Blue
                
                try {
                    $process = Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT", "/NORESTART", "/COMPONENTS=icons,ext\reg\shellhere,ext\reg\guihere" -Wait -PassThru
                    
                    if ($process.ExitCode -eq 0) {
                        Write-Success "Git installed successfully"
                    } else {
                        Write-Error "Git installation failed with exit code: $($process.ExitCode)"
                    }
                } catch {
                    Write-Error "Git installation failed`: $($_.Exception.Message)"
                }
            }
        }
    }
}

# Create build scripts with improved error handling
function New-BuildScripts {
    param([string]$ProjectPath)
    
    # Build all script
    $buildScript = @"
@echo off
setlocal enabledelayedexpansion
echo Building Al-artworks project...
cd /d "$ProjectPath\aisis"
if exist "build" rmdir /s /q "build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
if !errorLevel! equ 0 (
    cmake --build build --config Release
    if !errorLevel! equ 0 (
        echo Build successful!
    ) else (
        echo Build failed!
    )
) else (
    echo CMake configuration failed!
)
pause
"@
    Set-Content -Path "$ProjectPath\build_all.bat" -Value $buildScript

    # Run AI artworks script
    $runScript = @"
@echo off
echo Starting Al-artworks AI Suite...
cd /d "$ProjectPath\Al-artworks"
if exist "app.py" (
    python app.py
) else if exist "main.py" (
    python main.py
) else (
    python -m flask run --host=0.0.0.0 --port=5000
)
pause
"@
    Set-Content -Path "$ProjectPath\run_ai_artworks.bat" -Value $runScript
}

# Create desktop shortcuts with error handling
function New-DesktopShortcuts {
    param([string]$ProjectPath)
    
    try {
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Al-artworks.lnk")
        $Shortcut.TargetPath = "$ProjectPath\run_ai_artworks.bat"
        $Shortcut.WorkingDirectory = $ProjectPath
        $Shortcut.Description = "Al-artworks AI Suite"
        $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,0"
        $Shortcut.Save()
        Write-Success "Desktop shortcut created"
    } catch {
        Write-Warning "Failed to create desktop shortcut`: $($_.Exception.Message)"
    }
}

# Test installation with detailed reporting
function Test-Installation {
    $tools = @(
        @{Name="GCC"; Command="gcc --version"},
        @{Name="CMake"; Command="cmake --version"},
        @{Name="Git"; Command="git --version"},
        @{Name="Python"; Command="python --version"}
    )

    $passedTests = 0
    $totalTests = $tools.Count

    foreach ($tool in $tools) {
        Write-Host "🧪 Testing $($tool.Name)..." -ForegroundColor Blue
        try {
            $null = Invoke-Expression $tool.Command 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$($tool.Name) working"
                $passedTests++
            } else {
                Write-Warning "$($tool.Name) not working - restart terminal after installation"
            }
        } catch {
            Write-Warning "$($tool.Name) not working - restart terminal after installation"
        }
    }

    Write-Host ""
    Write-Host "📊 Installation Test Results: $passedTests/$totalTests tests passed" -ForegroundColor Cyan
}

# Main execution
if ($Silent) {
    Install-AlArtworks
} else {
    Write-Host "Press any key to start installation..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Install-AlArtworks
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 