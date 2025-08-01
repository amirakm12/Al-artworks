# Modern Al-artworks Installer
# PowerShell-based installer with advanced UI and comprehensive features

param(
    [switch]$Silent,
    [switch]$SkipDependencies,
    [string]$InstallPath = "C:\Al-artworks"
)

# Set console encoding and colors
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Al-artworks Modern Installer"

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

# Download function with progress
function Invoke-DownloadWithProgress {
    param([string]$Uri, [string]$OutFile, [string]$Description)
    
    try {
        Write-Host "📥 $Description..." -ForegroundColor Blue
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Uri, $OutFile)
        Write-Success "$Description downloaded successfully"
        return $true
    }
    catch {
        Write-Error "Failed to download $Description"
        return $false
    }
}

# Main installation function
function Install-AlArtworks {
    Write-Header "AL-ARTWORKS MODERN INSTALLER"
    Write-Host "🎨 AI-Powered Art Creation Suite" -ForegroundColor Magenta
    Write-Host "🚀 Modern Development Environment" -ForegroundColor Magenta
    Write-Host "⚡ Lightning-Fast Installation" -ForegroundColor Magenta
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

    # Installation steps
    $steps = @(
        @{Name="MSYS2"; Description="C++ Compiler"; Check="C:\msys64\mingw64\bin\g++.exe"},
        @{Name="CMake"; Description="Build System"; Check="cmake"},
        @{Name="Git"; Description="Version Control"; Check="git"}
    )

    $stepNumber = 1
    foreach ($step in $steps) {
        Write-Step "Step $stepNumber`: Installing $($step.Name)" $step.Description
        
        if ($step.Check -eq "cmake" -or $step.Check -eq "git") {
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
        }
        else {
            # Check if file exists
            if (Test-Path $step.Check) {
                Write-Success "$($step.Name) already installed"
                continue
            }
            else {
                # Install the tool
                Install-Tool -ToolName $step.Name
            }
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
                "numpy", "opencv-python", "pillow", "matplotlib", "seaborn",
                "torch", "torchvision", "torchaudio",
                "transformers", "diffusers", "accelerate",
                "flask", "fastapi", "uvicorn",
                "requests", "beautifulsoup4", "lxml"
            )

            foreach ($package in $packages) {
                Write-Host "📦 Installing $package..." -ForegroundColor Blue
                pip install $package --quiet
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "$package installed"
                } else {
                    Write-Warning "Failed to install $package"
                }
            }
        } else {
            Write-Warning "Python not found - please install Python 3.8+ manually"
        }
    }
    catch {
        Write-Warning "Python installation check failed"
    }

    Write-Host ""

    # Create project structure
    Write-Step "Step 5: Creating Project Structure" "Setting up Al-artworks"
    
    $sourcePath = Split-Path $PSScriptRoot -Parent
    $projectPath = "$InstallPath\projects"
    
    # Copy project files
    if (Test-Path "$sourcePath\Al-artworks") {
        Copy-Item -Path "$sourcePath\Al-artworks" -Destination "$projectPath\Al-artworks" -Recurse -Force
        Write-Success "Al-artworks project copied"
    }
    
    if (Test-Path "$sourcePath\aisis") {
        Copy-Item -Path "$sourcePath\aisis" -Destination "$projectPath\aisis" -Recurse -Force
        Write-Success "Aisis project copied"
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
}

# Install specific tools
function Install-Tool {
    param([string]$ToolName)
    
    switch ($ToolName) {
        "MSYS2" {
            $installerPath = "$InstallPath\tools\msys2-installer.exe"
            $downloadUrl = "https://github.com/msys2/msys2-installer/releases/download/2024-01-13/msys2-x86_64-20240113.exe"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "MSYS2 installer") {
                Write-Host "🔄 Installing MSYS2..." -ForegroundColor Blue
                Start-Process -FilePath $installerPath -ArgumentList "--accept-messages", "--accept-licenses", "--root", "C:\msys64" -Wait
                
                if (Test-Path "C:\msys64\mingw64\bin\g++.exe") {
                    Write-Success "MSYS2 installed successfully"
                    # Add to PATH
                    $env:PATH += ";C:\msys64\mingw64\bin"
                    [Environment]::SetEnvironmentVariable("PATH", $env:PATH, [EnvironmentVariableTarget]::Machine)
                } else {
                    Write-Error "MSYS2 installation failed"
                }
            }
        }
        
        "CMake" {
            $installerPath = "$InstallPath\tools\cmake-installer.msi"
            $downloadUrl = "https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "CMake installer") {
                Write-Host "🔄 Installing CMake..." -ForegroundColor Blue
                Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $installerPath, "/quiet", "ADD_TO_PATH=1" -Wait
                
                try {
                    $null = Get-Command cmake -ErrorAction Stop
                    Write-Success "CMake installed successfully"
                } catch {
                    Write-Error "CMake installation failed"
                }
            }
        }
        
        "Git" {
            $installerPath = "$InstallPath\tools\git-installer.exe"
            $downloadUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
            
            if (Invoke-DownloadWithProgress -Uri $downloadUrl -OutFile $installerPath -Description "Git installer") {
                Write-Host "🔄 Installing Git..." -ForegroundColor Blue
                Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
                
                try {
                    $null = Get-Command git -ErrorAction Stop
                    Write-Success "Git installed successfully"
                } catch {
                    Write-Error "Git installation failed"
                }
            }
        }
    }
}

# Create build scripts
function New-BuildScripts {
    param([string]$ProjectPath)
    
    # Build all script
    $buildScript = @"
@echo off
echo Building Al-artworks project...
cd /d "$ProjectPath\aisis"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
echo Build complete!
pause
"@
    Set-Content -Path "$ProjectPath\build_all.bat" -Value $buildScript

    # Run AI artworks script
    $runScript = @"
@echo off
echo Starting Al-artworks AI Suite...
cd /d "$ProjectPath\Al-artworks"
python -m flask run --host=0.0.0.0 --port=5000
"@
    Set-Content -Path "$ProjectPath\run_ai_artworks.bat" -Value $runScript
}

# Create desktop shortcuts
function New-DesktopShortcuts {
    param([string]$ProjectPath)
    
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Al-artworks.lnk")
    $Shortcut.TargetPath = "$ProjectPath\run_ai_artworks.bat"
    $Shortcut.WorkingDirectory = $ProjectPath
    $Shortcut.Description = "Al-artworks AI Suite"
    $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,0"
    $Shortcut.Save()
}

# Test installation
function Test-Installation {
    $tools = @(
        @{Name="GCC"; Command="gcc --version"},
        @{Name="CMake"; Command="cmake --version"},
        @{Name="Git"; Command="git --version"}
    )

    foreach ($tool in $tools) {
        Write-Host "🧪 Testing $($tool.Name)..." -ForegroundColor Blue
        try {
            $null = Invoke-Expression $tool.Command 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$($tool.Name) working"
            } else {
                Write-Warning "$($tool.Name) not working - restart terminal after installation"
            }
        } catch {
            Write-Warning "$($tool.Name) not working - restart terminal after installation"
        }
    }
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