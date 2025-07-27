# PowerShell script to build with Clang on Windows
# Addresses lld-link library linking issues with multiple solutions

param(
    [string]$BuildType = "Debug",
    [string]$Generator = "Ninja",
    [switch]$UseClangCL = $true,
    [switch]$ForceLLD = $false,
    [switch]$Clean = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Windows Clang Build Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Clean build directory if requested
if ($Clean -and (Test-Path "build")) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build"
}

# Function to find Visual Studio installation
function Find-VisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio not found. Please install Visual Studio 2019 or later with C++ tools."
    }
    
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    
    if (-not $vsPath) {
        throw "Visual Studio installation with C++ tools not found."
    }
    
    return $vsPath
}

# Function to find MSVC tools
function Find-MSVCTools {
    param([string]$VSPath)
    
    $msvcPath = Join-Path $VSPath "VC\Tools\MSVC"
    $versions = Get-ChildItem $msvcPath | Sort-Object Name -Descending
    
    if ($versions.Count -eq 0) {
        throw "MSVC tools not found in Visual Studio installation."
    }
    
    return $versions[0].FullName
}

# Function to find Windows SDK
function Find-WindowsSDK {
    $sdkPaths = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10",
        "${env:ProgramFiles}\Windows Kits\10"
    )
    
    $sdkRoot = $null
    foreach ($path in $sdkPaths) {
        if (Test-Path $path) {
            $sdkRoot = $path
            break
        }
    }
    
    if (-not $sdkRoot) {
        throw "Windows SDK not found. Please install Windows 10 SDK."
    }
    
    # Find latest SDK version
    $libPath = Join-Path $sdkRoot "Lib"
    $versions = Get-ChildItem $libPath | Where-Object { 
        Test-Path (Join-Path $_.FullName "ucrt\x64") 
    } | Sort-Object Name -Descending
    
    if ($versions.Count -eq 0) {
        throw "Windows SDK libraries not found."
    }
    
    return @{
        Root = $sdkRoot
        Version = $versions[0].Name
    }
}

# Function to setup environment
function Setup-Environment {
    param(
        [string]$MSVCPath,
        [hashtable]$SDK
    )
    
    $env:INCLUDE = @(
        "$MSVCPath\include",
        "$($SDK.Root)\Include\$($SDK.Version)\ucrt",
        "$($SDK.Root)\Include\$($SDK.Version)\shared",
        "$($SDK.Root)\Include\$($SDK.Version)\um",
        "$($SDK.Root)\Include\$($SDK.Version)\winrt"
    ) -join ";"
    
    $env:LIB = @(
        "$MSVCPath\lib\x64",
        "$($SDK.Root)\Lib\$($SDK.Version)\ucrt\x64",
        "$($SDK.Root)\Lib\$($SDK.Version)\um\x64"
    ) -join ";"
    
    $env:LIBPATH = @(
        "$MSVCPath\lib\x64",
        "$($SDK.Root)\Lib\$($SDK.Version)\ucrt\x64"
    ) -join ";"
    
    # Add MSVC tools to PATH
    $env:PATH = "$MSVCPath\bin\Hostx64\x64;$env:PATH"
    
    Write-Host "Environment configured:" -ForegroundColor Green
    Write-Host "  INCLUDE: $env:INCLUDE" -ForegroundColor Gray
    Write-Host "  LIB: $env:LIB" -ForegroundColor Gray
    Write-Host "  LIBPATH: $env:LIBPATH" -ForegroundColor Gray
}

# Function to build with different strategies
function Build-WithStrategy {
    param(
        [string]$Strategy,
        [string]$MSVCPath,
        [hashtable]$SDK,
        [string]$BuildType,
        [string]$Generator
    )
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Trying Strategy: $Strategy" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $cmakeArgs = @(
        "..",
        "-G", $Generator,
        "-DCMAKE_BUILD_TYPE=$BuildType"
    )
    
    switch ($Strategy) {
        "ClangCL-MSVC" {
            $cmakeArgs += @(
                "-DCMAKE_C_COMPILER=clang-cl",
                "-DCMAKE_CXX_COMPILER=clang-cl",
                "-DCMAKE_LINKER=link.exe",
                "-DCMAKE_C_FLAGS=/clang:-fms-compatibility-version=19.29",
                "-DCMAKE_CXX_FLAGS=/clang:-fms-compatibility-version=19.29"
            )
        }
        "Clang-MSVC" {
            $cmakeArgs += @(
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=link.exe",
                "-DCMAKE_C_FLAGS=-target x86_64-pc-windows-msvc",
                "-DCMAKE_CXX_FLAGS=-target x86_64-pc-windows-msvc"
            )
        }
        "ClangCL-LLD" {
            $libPaths = @(
                "/LIBPATH:`"$MSVCPath\lib\x64`"",
                "/LIBPATH:`"$($SDK.Root)\Lib\$($SDK.Version)\ucrt\x64`"",
                "/LIBPATH:`"$($SDK.Root)\Lib\$($SDK.Version)\um\x64`""
            )
            $libPathStr = $libPaths -join " "
            
            $cmakeArgs += @(
                "-DCMAKE_C_COMPILER=clang-cl",
                "-DCMAKE_CXX_COMPILER=clang-cl",
                "-DCMAKE_LINKER=lld-link",
                "-DCMAKE_EXE_LINKER_FLAGS=$libPathStr",
                "-DCMAKE_SHARED_LINKER_FLAGS=$libPathStr"
            )
        }
        "Clang-LLD" {
            $libPaths = @(
                "/LIBPATH:`"$MSVCPath\lib\x64`"",
                "/LIBPATH:`"$($SDK.Root)\Lib\$($SDK.Version)\ucrt\x64`"",
                "/LIBPATH:`"$($SDK.Root)\Lib\$($SDK.Version)\um\x64`""
            )
            $libPathStr = $libPaths -join " "
            
            $cmakeArgs += @(
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld-link",
                "-DCMAKE_C_FLAGS=-target x86_64-pc-windows-msvc",
                "-DCMAKE_CXX_FLAGS=-target x86_64-pc-windows-msvc",
                "-DCMAKE_EXE_LINKER_FLAGS=$libPathStr",
                "-DCMAKE_SHARED_LINKER_FLAGS=$libPathStr"
            )
        }
    }
    
    Write-Host "Running CMake with args: $($cmakeArgs -join ' ')" -ForegroundColor Yellow
    
    $result = Start-Process -FilePath "cmake" -ArgumentList $cmakeArgs -Wait -PassThru -NoNewWindow
    
    if ($result.ExitCode -eq 0) {
        Write-Host "CMake configuration successful!" -ForegroundColor Green
        
        # Build the project
        Write-Host "Building project..." -ForegroundColor Yellow
        $buildResult = Start-Process -FilePath $Generator.ToLower() -Wait -PassThru -NoNewWindow
        
        if ($buildResult.ExitCode -eq 0) {
            Write-Host "Build successful with strategy: $Strategy" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Build failed with strategy: $Strategy" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "CMake configuration failed with strategy: $Strategy" -ForegroundColor Red
        return $false
    }
}

try {
    # Find required tools
    Write-Host "Finding Visual Studio installation..." -ForegroundColor Yellow
    $vsPath = Find-VisualStudio
    Write-Host "Found Visual Studio at: $vsPath" -ForegroundColor Green
    
    Write-Host "Finding MSVC tools..." -ForegroundColor Yellow
    $msvcPath = Find-MSVCTools -VSPath $vsPath
    Write-Host "Found MSVC tools at: $msvcPath" -ForegroundColor Green
    
    Write-Host "Finding Windows SDK..." -ForegroundColor Yellow
    $sdk = Find-WindowsSDK
    Write-Host "Found Windows SDK $($sdk.Version) at: $($sdk.Root)" -ForegroundColor Green
    
    # Setup environment
    Setup-Environment -MSVCPath $msvcPath -SDK $sdk
    
    # Create build directory
    if (-not (Test-Path "build")) {
        New-Item -ItemType Directory -Path "build" | Out-Null
    }
    Set-Location "build"
    
    # Define build strategies in order of preference
    $strategies = @()
    
    if ($ForceLLD) {
        $strategies += @("ClangCL-LLD", "Clang-LLD")
    } elseif ($UseClangCL) {
        $strategies += @("ClangCL-MSVC", "ClangCL-LLD", "Clang-MSVC", "Clang-LLD")
    } else {
        $strategies += @("Clang-MSVC", "ClangCL-MSVC", "Clang-LLD", "ClangCL-LLD")
    }
    
    # Try each strategy until one succeeds
    $success = $false
    foreach ($strategy in $strategies) {
        if (Build-WithStrategy -Strategy $strategy -MSVCPath $msvcPath -SDK $sdk -BuildType $BuildType -Generator $Generator) {
            $success = $true
            break
        }
        
        Write-Host "Strategy $strategy failed, trying next..." -ForegroundColor Yellow
    }
    
    if ($success) {
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "BUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "ALL BUILD STRATEGIES FAILED!" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "Please check:" -ForegroundColor Yellow
        Write-Host "1. Clang/LLVM is properly installed" -ForegroundColor Yellow
        Write-Host "2. Visual Studio 2019+ with C++ tools is installed" -ForegroundColor Yellow
        Write-Host "3. Windows 10 SDK is installed" -ForegroundColor Yellow
        Write-Host "4. All tools are in PATH" -ForegroundColor Yellow
        exit 1
    }
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
} finally {
    # Return to original directory
    if (Get-Location | Where-Object { $_.Path.EndsWith("build") }) {
        Set-Location ".."
    }
}