# Visual Studio Environment Setup Script
# Supports VS 2017, 2019, 2022, 2025+ with automatic detection and fallback paths

param(
    [string]$Architecture = "x64",
    [string]$PreferredVersion = "2022",
    [switch]$Verbose = $false,
    [switch]$Force = $false
)

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Color output functions
function Write-Success { param([string]$Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Warning { param([string]$Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param([string]$Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Info { param([string]$Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }

# Visual Studio installation paths with fallback support
$VSPaths = @{
    "2025" = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2025\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2025\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2025\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2025\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
    "2022" = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
    "2019" = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
    "2017" = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
}

function Find-VisualStudio {
    param([string]$PreferredVersion)
    
    Write-Info "Searching for Visual Studio installations..."
    
    # Try preferred version first
    if ($VSPaths.ContainsKey($PreferredVersion)) {
        foreach ($path in $VSPaths[$PreferredVersion]) {
            if (Test-Path $path) {
                Write-Success "Found Visual Studio $PreferredVersion at: $path"
                return @{ Version = $PreferredVersion; Path = $path }
            }
        }
        Write-Warning "Visual Studio $PreferredVersion not found, trying other versions..."
    }
    
    # Try all versions in order of preference (newest first)
    $versionOrder = @("2025", "2022", "2019", "2017")
    foreach ($version in $versionOrder) {
        if ($version -eq $PreferredVersion) { continue } # Already tried
        
        foreach ($path in $VSPaths[$version]) {
            if (Test-Path $path) {
                Write-Success "Found Visual Studio $version at: $path"
                return @{ Version = $version; Path = $path }
            }
        }
    }
    
    # Try vswhere.exe as last resort
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        Write-Info "Using vswhere.exe to locate Visual Studio..."
        try {
            $vsInfo = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($vsInfo) {
                $vcvarsPath = Join-Path $vsInfo "VC\Auxiliary\Build\vcvarsall.bat"
                if (Test-Path $vcvarsPath) {
                    Write-Success "Found Visual Studio via vswhere at: $vcvarsPath"
                    return @{ Version = "Unknown"; Path = $vcvarsPath }
                }
            }
        }
        catch {
            Write-Warning "vswhere.exe failed: $($_.Exception.Message)"
        }
    }
    
    return $null
}

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if running as Administrator (recommended for some operations)
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    $isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Warning "Not running as Administrator. Some operations may fail."
        Write-Info "Consider running: Right-click PowerShell -> 'Run as Administrator'"
    } else {
        Write-Success "Running with Administrator privileges"
    }
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Error "PowerShell 5.0 or later is required. Current version: $($PSVersionTable.PSVersion)"
        exit 1
    }
    Write-Success "PowerShell version: $($PSVersionTable.PSVersion)"
    
    # Check Windows version
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Warning "Windows 10 or later is recommended. Current version: $osVersion"
    } else {
        Write-Success "Windows version: $osVersion"
    }
}

function Set-EnvironmentVariables {
    param([hashtable]$VSInfo, [string]$Architecture)
    
    Write-Info "Setting up Visual Studio environment for $Architecture architecture..."
    
    # Create temporary batch file to capture environment
    $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
    $tempOut = [System.IO.Path]::GetTempFileName() + ".txt"
    
    try {
        # Create batch file that calls vcvarsall and outputs environment
        @"
@echo off
call "$($VSInfo.Path)" $Architecture
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to initialize Visual Studio environment
    exit /b %ERRORLEVEL%
)
set > "$tempOut"
"@ | Out-File -FilePath $tempBat -Encoding ASCII
        
        # Execute batch file
        $result = & cmd.exe /c $tempBat
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to initialize Visual Studio environment (Exit code: $LASTEXITCODE)"
        }
        
        # Parse environment variables
        if (Test-Path $tempOut) {
            $envVars = Get-Content $tempOut | Where-Object { $_ -match '^([^=]+)=(.*)$' }
            $setCount = 0
            
            foreach ($line in $envVars) {
                if ($line -match '^([^=]+)=(.*)$') {
                    $name = $matches[1]
                    $value = $matches[2]
                    
                    # Only set important VS-related variables
                    $importantVars = @(
                        'PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCINSTALLDIR', 'VSINSTALLDIR',
                        'WindowsSDKDir', 'WindowsSDKVersion', 'Platform', 'VSCMD_ARG_TGT_ARCH',
                        'CMAKE_PREFIX_PATH', 'CC', 'CXX'
                    )
                    
                    if ($importantVars -contains $name) {
                        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
                        $setCount++
                        if ($Verbose) {
                            Write-Info "Set $name = $value"
                        }
                    }
                }
            }
            
            Write-Success "Set $setCount environment variables"
        } else {
            throw "Failed to capture environment variables"
        }
        
    } finally {
        # Cleanup temporary files
        if (Test-Path $tempBat) { Remove-Item $tempBat -Force }
        if (Test-Path $tempOut) { Remove-Item $tempOut -Force }
    }
}

function Test-BuildTools {
    Write-Info "Verifying build tools..."
    
    $tools = @{
        'cl.exe' = 'MSVC Compiler'
        'link.exe' = 'MSVC Linker'
        'cmake.exe' = 'CMake'
        'ninja.exe' = 'Ninja Build System'
    }
    
    $allFound = $true
    foreach ($tool in $tools.Keys) {
        try {
            $null = Get-Command $tool -ErrorAction Stop
            Write-Success "$($tools[$tool]) found: $tool"
        }
        catch {
            Write-Warning "$($tools[$tool]) not found: $tool"
            if ($tool -eq 'cl.exe' -or $tool -eq 'link.exe') {
                $allFound = $false
            }
        }
    }
    
    if (-not $allFound) {
        Write-Error "Critical build tools are missing. Please install Visual Studio Build Tools."
        return $false
    }
    
    return $true
}

function Show-Summary {
    param([hashtable]$VSInfo, [string]$Architecture)
    
    Write-Info "Environment Setup Summary:"
    Write-Host "  Visual Studio Version: $($VSInfo.Version)" -ForegroundColor White
    Write-Host "  Architecture: $Architecture" -ForegroundColor White
    Write-Host "  vcvarsall.bat: $($VSInfo.Path)" -ForegroundColor White
    
    # Show key environment variables
    $keyVars = @('VCINSTALLDIR', 'WindowsSDKVersion', 'Platform', 'VSCMD_ARG_TGT_ARCH')
    foreach ($var in $keyVars) {
        $value = [System.Environment]::GetEnvironmentVariable($var)
        if ($value) {
            Write-Host "  $var = $value" -ForegroundColor White
        }
    }
}

# Main execution
try {
    Write-Info "AI-Artworks Visual Studio Environment Setup"
    Write-Info "=========================================="
    
    # Check prerequisites
    Test-Prerequisites
    
    # Find Visual Studio
    $vsInfo = Find-VisualStudio -PreferredVersion $PreferredVersion
    if (-not $vsInfo) {
        Write-Error "Visual Studio not found. Please install Visual Studio 2017 or later with C++ tools."
        Write-Info "Download from: https://visualstudio.microsoft.com/downloads/"
        exit 1
    }
    
    # Set up environment
    Set-EnvironmentVariables -VSInfo $vsInfo -Architecture $Architecture
    
    # Verify tools
    if (-not (Test-BuildTools)) {
        exit 1
    }
    
    # Show summary
    Show-Summary -VSInfo $vsInfo -Architecture $Architecture
    
    Write-Success "Visual Studio environment setup complete!"
    Write-Info "You can now run build commands in this PowerShell session."
    
} catch {
    Write-Error "Setup failed: $($_.Exception.Message)"
    if ($Verbose) {
        Write-Host $_.Exception.StackTrace -ForegroundColor Red
    }
    exit 1
}