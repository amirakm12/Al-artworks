# AI-ARTWORKS Installer Build Script
# PowerShell script for building the complete installer package

param(
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [string]$OutputDir = ".\output",
    [switch]$Clean = $false,
    [switch]$SkipCppBuild = $false,
    [switch]$SkipPyInstaller = $false
)

# Script configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$InstallerRoot = $ProjectRoot

# Tool paths (adjust as needed)
$WixToolsetPath = "${env:ProgramFiles(x86)}\WiX Toolset v4\bin"
$MSBuildPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
$PythonPath = "python"  # Assumes Python is in PATH

# Verify required tools
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow
    
    # Check WiX Toolset
    if (-not (Test-Path "$WixToolsetPath\wix.exe")) {
        Write-Error "WiX Toolset v4 not found at $WixToolsetPath. Please install WiX Toolset."
    }
    
    # Check MSBuild
    if (-not (Test-Path $MSBuildPath)) {
        Write-Error "MSBuild not found at $MSBuildPath. Please install Visual Studio 2022."
    }
    
    # Check Python
    try {
        $pythonVersion = & $PythonPath --version 2>&1
        Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Error "Python not found in PATH. Please install Python 3.10+."
    }
    
    Write-Host "Prerequisites check passed!" -ForegroundColor Green
}

# Clean output directories
function Clean-Output {
    if ($Clean) {
        Write-Host "Cleaning output directories..." -ForegroundColor Yellow
        
        $cleanDirs = @(
            "$OutputDir",
            "$InstallerRoot\obj",
            "$InstallerRoot\bin",
            "$ProjectRoot\..\build"
        )
        
        foreach ($dir in $cleanDirs) {
            if (Test-Path $dir) {
                Remove-Item -Path $dir -Recurse -Force
                Write-Host "Cleaned: $dir" -ForegroundColor Gray
            }
        }
    }
}

# Build C++ components
function Build-CppComponents {
    if ($SkipCppBuild) {
        Write-Host "Skipping C++ build..." -ForegroundColor Yellow
        return
    }
    
    Write-Host "Building C++ components..." -ForegroundColor Yellow
    
    # Build main C++ library
    $cppBuildDir = "$ProjectRoot\..\build"
    New-Item -ItemType Directory -Path $cppBuildDir -Force | Out-Null
    
    Push-Location $cppBuildDir
    try {
        # Configure with CMake
        & cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=$Configuration
        if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
        
        # Build
        & cmake --build . --config $Configuration --parallel
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
        
        Write-Host "C++ build completed successfully!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
    
    # Build Custom Actions DLL
    Write-Host "Building Custom Actions DLL..." -ForegroundColor Yellow
    
    $customActionsProj = @"
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup>
    <Configuration>$Configuration</Configuration>
    <Platform>$Platform</Platform>
    <ProjectGuid>{12345678-1234-1234-1234-123456789012}</ProjectGuid>
    <RootNamespace>CustomActions</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <UseDebugLibraries Condition="'`$(Configuration)'=='Debug'">true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'`$(Configuration)|`$(Platform)'=='$Configuration|$Platform'">
    <OutDir>`$(SolutionDir)bin\`$(Platform)\`$(Configuration)\</OutDir>
    <IntDir>`$(SolutionDir)obj\`$(Platform)\`$(Configuration)\`$(ProjectName)\</IntDir>
    <TargetName>CustomActions</TargetName>
  </PropertyGroup>
  <ItemDefinitionGroup>
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>NotUsing</PrecompiledHeader>
      <Optimization Condition="'`$(Configuration)'=='Release'">MaxSpeed</Optimization>
      <FunctionLevelLinking Condition="'`$(Configuration)'=='Release'">true</FunctionLevelLinking>
      <IntrinsicFunctions Condition="'`$(Configuration)'=='Release'">true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;_WINDOWS;_USRDLL;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <AdditionalIncludeDirectories>`$(ProjectDir);%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <EnableCOMDATFolding Condition="'`$(Configuration)'=='Release'">true</EnableCOMDATFolding>
      <OptimizeReferences Condition="'`$(Configuration)'=='Release'">true</OptimizeReferences>
      <ModuleDefinitionFile>CustomActions.def</ModuleDefinitionFile>
      <AdditionalDependencies>msi.lib;advapi32.lib;psapi.lib;d3d11.lib;dxgi.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="CustomActions.cpp" />
  </ItemGroup>
</Project>
"@
    
    $customActionsProj | Out-File -FilePath "$InstallerRoot\scripts\CustomActions.vcxproj" -Encoding UTF8
    
    # Create module definition file
    $customActionsDef = @"
EXPORTS
SetupPythonEnvironment
DownloadAIModels
DetectAndConfigureGPU
"@
    
    $customActionsDef | Out-File -FilePath "$InstallerRoot\scripts\CustomActions.def" -Encoding ASCII
    
    # Build the DLL
    & $MSBuildPath "$InstallerRoot\scripts\CustomActions.vcxproj" /p:Configuration=$Configuration /p:Platform=$Platform
    if ($LASTEXITCODE -ne 0) { throw "Custom Actions DLL build failed" }
    
    Write-Host "Custom Actions DLL built successfully!" -ForegroundColor Green
}

# Create Python executable using PyInstaller
function Build-PythonExecutable {
    if ($SkipPyInstaller) {
        Write-Host "Skipping PyInstaller build..." -ForegroundColor Yellow
        return
    }
    
    Write-Host "Building Python executable with PyInstaller..." -ForegroundColor Yellow
    
    $aiArtworksDir = "$ProjectRoot\..\AI-ARTWORKS"
    $distDir = "$OutputDir\python_dist"
    
    Push-Location $aiArtworksDir
    try {
        # Install PyInstaller if not available
        & $PythonPath -m pip install pyinstaller --quiet
        
        # Create PyInstaller spec file
        $specContent = @"
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['$aiArtworksDir'],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('config', 'config'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'torch',
        'torchvision',
        'transformers',
        'diffusers',
        'numpy',
        'PIL',
        'cv2',
        'loguru',
        'whisper',
        'bark'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI-ARTWORKS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/app_icon.ico' if os.path.exists('resources/app_icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI-ARTWORKS',
)
"@
        
        $specContent | Out-File -FilePath "ai_artworks.spec" -Encoding UTF8
        
        # Build with PyInstaller
        & $PythonPath -m PyInstaller ai_artworks.spec --distpath $distDir --clean --noconfirm
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
        
        Write-Host "PyInstaller build completed successfully!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# Build WiX MSI packages
function Build-MSIPackages {
    Write-Host "Building MSI packages..." -ForegroundColor Yellow
    
    $wixPath = "$WixToolsetPath\wix.exe"
    $wixObjDir = "$InstallerRoot\obj"
    $wixBinDir = "$OutputDir"
    
    New-Item -ItemType Directory -Path $wixObjDir -Force | Out-Null
    New-Item -ItemType Directory -Path $wixBinDir -Force | Out-Null
    
    # Build main application MSI
    Write-Host "Building AI-ARTWORKS Core MSI..." -ForegroundColor Gray
    & $wixPath build "$InstallerRoot\wix\AIArtworks.wxs" -o "$wixBinDir\AIArtworks.msi" -d "ProjectDir=$InstallerRoot"
    if ($LASTEXITCODE -ne 0) { throw "Main MSI build failed" }
    
    # Build C++ runtime MSI (if needed)
    $cppMsiContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Product Id="*" Name="AI-ARTWORKS C++ Runtime" Language="1033" Version="1.0.0.0" 
           Manufacturer="AI-ARTWORKS" UpgradeCode="BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB">
    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />
    <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
    <MediaTemplate EmbedCab="yes" />
    
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFiles64Folder">
        <Directory Id="INSTALLFOLDER" Name="AI-ARTWORKS">
          <Directory Id="BinFolder" Name="bin">
            <Component Id="CppRuntime" Guid="CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC">
              <File Id="UltimateLib" Source="`$(var.ProjectDir)\..\build\lib\$Configuration\ultimate.lib" KeyPath="yes" />
            </Component>
          </Directory>
        </Directory>
      </Directory>
    </Directory>
    
    <Feature Id="ProductFeature" Title="C++ Runtime" Level="1">
      <ComponentRef Id="CppRuntime" />
    </Feature>
  </Product>
</Wix>
"@
    
    $cppMsiContent | Out-File -FilePath "$InstallerRoot\wix\AIArtworksCPP.wxs" -Encoding UTF8
    
    Write-Host "Building AI-ARTWORKS C++ Runtime MSI..." -ForegroundColor Gray
    & $wixPath build "$InstallerRoot\wix\AIArtworksCPP.wxs" -o "$wixBinDir\AIArtworksCPP.msi" -d "ProjectDir=$InstallerRoot"
    if ($LASTEXITCODE -ne 0) { throw "C++ Runtime MSI build failed" }
    
    Write-Host "MSI packages built successfully!" -ForegroundColor Green
}

# Build WiX Bundle
function Build-Bundle {
    Write-Host "Building WiX Bundle..." -ForegroundColor Yellow
    
    $wixPath = "$WixToolsetPath\wix.exe"
    $bundleOutput = "$OutputDir\AI-ARTWORKS-Setup.exe"
    
    & $wixPath build "$InstallerRoot\wix\Bundle.wxs" -o $bundleOutput -d "ProjectDir=$InstallerRoot"
    if ($LASTEXITCODE -ne 0) { throw "Bundle build failed" }
    
    Write-Host "Bundle built successfully: $bundleOutput" -ForegroundColor Green
}

# Code signing (optional)
function Sign-Installer {
    param([string]$FilePath)
    
    # Configure your code signing certificate here
    $certThumbprint = $env:CODE_SIGN_CERT_THUMBPRINT
    $timestampUrl = "http://timestamp.digicert.com"
    
    if ($certThumbprint) {
        Write-Host "Code signing: $FilePath" -ForegroundColor Yellow
        
        & signtool sign /sha1 $certThumbprint /t $timestampUrl /d "AI-ARTWORKS Ultimate Creative Studio" $FilePath
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Code signing completed successfully!" -ForegroundColor Green
        }
        else {
            Write-Warning "Code signing failed, but continuing..."
        }
    }
    else {
        Write-Host "No code signing certificate configured (set CODE_SIGN_CERT_THUMBPRINT environment variable)" -ForegroundColor Yellow
    }
}

# Download dependencies
function Download-Dependencies {
    Write-Host "Downloading installer dependencies..." -ForegroundColor Yellow
    
    $depsDir = "$InstallerRoot\dependencies"
    New-Item -ItemType Directory -Path $depsDir -Force | Out-Null
    
    # URLs for dependencies (update as needed)
    $dependencies = @{
        "python-3.12.0-amd64.exe" = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
        "VC_redist.x64.exe" = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        # CUDA installer is too large for automatic download - manual step required
    }
    
    foreach ($dep in $dependencies.GetEnumerator()) {
        $filePath = "$depsDir\$($dep.Key)"
        if (-not (Test-Path $filePath)) {
            Write-Host "Downloading $($dep.Key)..." -ForegroundColor Gray
            try {
                Invoke-WebRequest -Uri $dep.Value -OutFile $filePath -UseBasicParsing
                Write-Host "Downloaded: $($dep.Key)" -ForegroundColor Green
            }
            catch {
                Write-Warning "Failed to download $($dep.Key): $($_.Exception.Message)"
            }
        }
        else {
            Write-Host "Already exists: $($dep.Key)" -ForegroundColor Gray
        }
    }
}

# Main build process
function Build-Installer {
    Write-Host "=== AI-ARTWORKS Installer Build ===" -ForegroundColor Cyan
    Write-Host "Configuration: $Configuration" -ForegroundColor Cyan
    Write-Host "Platform: $Platform" -ForegroundColor Cyan
    Write-Host "Output Directory: $OutputDir" -ForegroundColor Cyan
    Write-Host ""
    
    try {
        Test-Prerequisites
        Clean-Output
        Download-Dependencies
        Build-CppComponents
        Build-PythonExecutable
        Build-MSIPackages
        Build-Bundle
        
        $bundlePath = "$OutputDir\AI-ARTWORKS-Setup.exe"
        Sign-Installer $bundlePath
        
        Write-Host ""
        Write-Host "=== BUILD COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
        Write-Host "Installer: $bundlePath" -ForegroundColor Green
        Write-Host "Size: $([math]::Round((Get-Item $bundlePath).Length / 1MB, 2)) MB" -ForegroundColor Green
        
        # Generate build summary
        $buildSummary = @"
AI-ARTWORKS Installer Build Summary
===================================
Build Time: $(Get-Date)
Configuration: $Configuration
Platform: $Platform
Output: $bundlePath

Components Built:
- WiX Bundle with prerequisite chaining
- MSI packages for core application and C++ runtime
- Custom Actions DLL for system detection
- PowerShell system check script
- Professional UI theme

Features:
- Smart dependency resolution (Python, VC++ Redist, CUDA)
- GPU detection and configuration
- Progress tracking with real-time updates
- Automatic rollback on failure
- Code signing ready
- Professional branding and UI

Next Steps:
1. Test installer on clean Windows systems
2. Verify GPU detection and CUDA installation
3. Test model download functionality
4. Validate uninstall and rollback scenarios
"@
        
        $buildSummary | Out-File -FilePath "$OutputDir\BuildSummary.txt" -Encoding UTF8
        Write-Host $buildSummary -ForegroundColor Cyan
        
    }
    catch {
        Write-Host ""
        Write-Host "=== BUILD FAILED ===" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Execute main build
Build-Installer