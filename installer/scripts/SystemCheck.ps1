# AI-ARTWORKS System Requirements Check
# PowerShell script for comprehensive system validation

param(
    [string]$Mode = "INSTALL",
    [string]$LogFile = "$env:TEMP\AIArtworks_SystemCheck.log"
)

# Initialize logging
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $LogFile -Value $logEntry
}

function Test-OSCompatibility {
    Write-Log "Checking OS compatibility..."
    
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $osVersion = [System.Environment]::OSVersion.Version
    
    Write-Log "OS: $($os.Caption) $($os.Version)"
    Write-Log "Architecture: $($os.OSArchitecture)"
    
    # Check for Windows 7 or later (6.1+)
    if ($osVersion.Major -lt 6 -or ($osVersion.Major -eq 6 -and $osVersion.Minor -lt 1)) {
        Write-Log "ERROR: Windows 7 or later required" "ERROR"
        return $false
    }
    
    # Check for 64-bit architecture
    if ($os.OSArchitecture -notlike "*64*") {
        Write-Log "ERROR: 64-bit Windows required" "ERROR"
        return $false
    }
    
    Write-Log "OS compatibility check passed" "SUCCESS"
    return $true
}

function Test-SystemMemory {
    Write-Log "Checking system memory..."
    
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
    
    Write-Log "Total Physical Memory: $totalMemoryGB GB"
    
    if ($totalMemoryGB -lt 8) {
        Write-Log "WARNING: Minimum 8GB RAM recommended (found $totalMemoryGB GB)" "WARNING"
        return $false
    }
    
    if ($totalMemoryGB -ge 16) {
        Write-Log "Excellent: 16GB+ RAM available for optimal performance" "SUCCESS"
    }
    
    return $true
}

function Test-DiskSpace {
    Write-Log "Checking disk space..."
    
    $systemDrive = $env:SystemDrive
    $drive = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $systemDrive }
    
    if ($drive) {
        $freeSpaceGB = [math]::Round($drive.FreeSpace / 1GB, 2)
        $totalSpaceGB = [math]::Round($drive.Size / 1GB, 2)
        
        Write-Log "System Drive ($systemDrive): $freeSpaceGB GB free of $totalSpaceGB GB total"
        
        if ($freeSpaceGB -lt 10) {
            Write-Log "ERROR: Minimum 10GB free space required (found $freeSpaceGB GB)" "ERROR"
            return $false
        }
        
        if ($freeSpaceGB -lt 50) {
            Write-Log "WARNING: Recommend 50GB+ free space for models and cache" "WARNING"
        }
    }
    
    return $true
}

function Test-PythonInstallation {
    Write-Log "Checking Python installation..."
    
    $pythonVersions = @("3.12", "3.11", "3.10")
    $pythonFound = $false
    
    foreach ($version in $pythonVersions) {
        $regPath = "HKLM:\SOFTWARE\Python\PythonCore\$version\InstallPath"
        if (Test-Path $regPath) {
            $pythonPath = (Get-ItemProperty $regPath).'(default)'
            if ($pythonPath -and (Test-Path "$pythonPath\python.exe")) {
                Write-Log "Found Python $version at: $pythonPath" "SUCCESS"
                $pythonFound = $true
                break
            }
        }
    }
    
    if (-not $pythonFound) {
        # Check PATH for python
        try {
            $pythonCmd = Get-Command python -ErrorAction Stop
            $pythonVersion = & python --version 2>&1
            Write-Log "Found Python in PATH: $pythonVersion"
            
            if ($pythonVersion -match "Python 3\.1[0-9]") {
                $pythonFound = $true
            }
        }
        catch {
            Write-Log "Python 3.10+ not found in registry or PATH" "WARNING"
        }
    }
    
    return $pythonFound
}

function Test-VCRedistributable {
    Write-Log "Checking Visual C++ Redistributable..."
    
    $vcRedistKeys = @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    )
    
    $vcRedistFound = $false
    foreach ($key in $vcRedistKeys) {
        if (Test-Path $key) {
            $vcRedist = Get-ItemProperty $key -ErrorAction SilentlyContinue
            if ($vcRedist -and $vcRedist.Installed -eq 1) {
                Write-Log "Found VC++ Redistributable: Version $($vcRedist.Version)" "SUCCESS"
                $vcRedistFound = $true
                break
            }
        }
    }
    
    if (-not $vcRedistFound) {
        Write-Log "Visual C++ Redistributable 2015-2022 not found" "WARNING"
    }
    
    return $vcRedistFound
}

function Test-GPUCapabilities {
    Write-Log "Detecting GPU capabilities..."
    
    try {
        $gpus = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -notlike "*Basic*" }
        
        $nvidiaFound = $false
        $amdFound = $false
        $intelFound = $false
        
        foreach ($gpu in $gpus) {
            $gpuName = $gpu.Name
            $gpuMemory = if ($gpu.AdapterRAM) { [math]::Round($gpu.AdapterRAM / 1MB, 0) } else { "Unknown" }
            
            Write-Log "GPU: $gpuName (${gpuMemory}MB)"
            
            if ($gpuName -like "*NVIDIA*" -or $gpuName -like "*GeForce*" -or $gpuName -like "*Quadro*" -or $gpuName -like "*Tesla*") {
                $nvidiaFound = $true
                Write-Log "NVIDIA GPU detected - CUDA acceleration available" "SUCCESS"
            }
            elseif ($gpuName -like "*AMD*" -or $gpuName -like "*Radeon*") {
                $amdFound = $true
                Write-Log "AMD GPU detected - DirectML acceleration available" "SUCCESS"
            }
            elseif ($gpuName -like "*Intel*") {
                $intelFound = $true
                Write-Log "Intel GPU detected - Basic acceleration available" "INFO"
            }
        }
        
        # Check for CUDA installation if NVIDIA GPU found
        if ($nvidiaFound) {
            $cudaPath = $env:CUDA_PATH
            if ($cudaPath -and (Test-Path $cudaPath)) {
                Write-Log "CUDA Toolkit found at: $cudaPath" "SUCCESS"
            }
            else {
                Write-Log "CUDA Toolkit not found - will be installed if needed" "INFO"
            }
        }
        
        if (-not ($nvidiaFound -or $amdFound -or $intelFound)) {
            Write-Log "No suitable GPU found - CPU-only processing will be used" "WARNING"
        }
        
        return @{
            NVIDIA = $nvidiaFound
            AMD = $amdFound
            Intel = $intelFound
        }
    }
    catch {
        Write-Log "Error detecting GPU: $($_.Exception.Message)" "ERROR"
        return @{ NVIDIA = $false; AMD = $false; Intel = $false }
    }
}

function Test-NetworkConnectivity {
    Write-Log "Testing network connectivity..."
    
    $testUrls = @(
        "https://github.com",
        "https://huggingface.co",
        "https://pypi.org"
    )
    
    $allConnected = $true
    foreach ($url in $testUrls) {
        try {
            $response = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Log "Connection to $url: OK" "SUCCESS"
            }
            else {
                Write-Log "Connection to $url: Failed (Status: $($response.StatusCode))" "WARNING"
                $allConnected = $false
            }
        }
        catch {
            Write-Log "Connection to $url: Failed ($($_.Exception.Message))" "WARNING"
            $allConnected = $false
        }
    }
    
    return $allConnected
}

function Test-WindowsFeatures {
    Write-Log "Checking Windows features..."
    
    # Check for .NET Framework
    try {
        $dotNetVersions = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP" -Recurse |
                         Get-ItemProperty -Name Version -ErrorAction SilentlyContinue |
                         Where-Object { $_.Version -like "4.*" } |
                         Sort-Object Version -Descending |
                         Select-Object -First 1
        
        if ($dotNetVersions) {
            Write-Log ".NET Framework: $($dotNetVersions.Version)" "SUCCESS"
        }
        else {
            Write-Log ".NET Framework 4.x not found" "WARNING"
        }
    }
    catch {
        Write-Log "Error checking .NET Framework" "WARNING"
    }
    
    # Check Windows Defender exclusions recommendation
    $defenderExclusions = Get-MpPreference -ErrorAction SilentlyContinue
    if ($defenderExclusions) {
        Write-Log "Windows Defender active - consider adding installation path to exclusions for better performance" "INFO"
    }
}

# Main execution
function Main {
    Write-Log "=== AI-ARTWORKS System Requirements Check ===" "INFO"
    Write-Log "Mode: $Mode" "INFO"
    
    $results = @{
        OS = Test-OSCompatibility
        Memory = Test-SystemMemory
        DiskSpace = Test-DiskSpace
        Python = Test-PythonInstallation
        VCRedist = Test-VCRedistributable
        GPU = Test-GPUCapabilities
        Network = Test-NetworkConnectivity
    }
    
    Test-WindowsFeatures
    
    # Summary
    Write-Log "=== SYSTEM CHECK SUMMARY ===" "INFO"
    
    $criticalIssues = 0
    $warnings = 0
    
    if (-not $results.OS) {
        Write-Log "CRITICAL: OS compatibility failed" "ERROR"
        $criticalIssues++
    }
    
    if (-not $results.DiskSpace) {
        Write-Log "CRITICAL: Insufficient disk space" "ERROR"
        $criticalIssues++
    }
    
    if (-not $results.Memory) {
        Write-Log "WARNING: Low system memory" "WARNING"
        $warnings++
    }
    
    if (-not $results.Python) {
        Write-Log "INFO: Python 3.10+ will be installed" "INFO"
    }
    
    if (-not $results.VCRedist) {
        Write-Log "INFO: VC++ Redistributable will be installed" "INFO"
    }
    
    if (-not $results.Network) {
        Write-Log "WARNING: Network connectivity issues detected" "WARNING"
        $warnings++
    }
    
    Write-Log "Critical Issues: $criticalIssues" "INFO"
    Write-Log "Warnings: $warnings" "INFO"
    
    if ($criticalIssues -eq 0) {
        Write-Log "System check PASSED - Installation can proceed" "SUCCESS"
        exit 0
    }
    else {
        Write-Log "System check FAILED - Critical issues must be resolved" "ERROR"
        exit 1
    }
}

# Execute main function
Main