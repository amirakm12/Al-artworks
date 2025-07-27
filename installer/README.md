# AI-ARTWORKS Professional Windows Installer

This directory contains a comprehensive Windows installer solution for AI-ARTWORKS Ultimate Creative Studio, featuring smart dependency resolution, GPU detection, progress tracking, and professional polish.

## 🚀 Features

### Smart Dependency Resolution
- **Automatic Detection**: Scans registry/HKLM for existing installations
- **Chain Installation**: Uses WiX Bundle to chain MSI/MSIX packages
- **Prerequisites**: Auto-installs Python 3.10+, VC++ Redistributable, CUDA toolkit
- **Conditional Logic**: Only installs missing components for efficiency

### Advanced Progress Tracking
- **Real-time Progress**: Embedded Qt-based custom dialogs with ETA calculations
- **Threaded Operations**: Non-blocking UI with Windows API file I/O monitoring
- **Millisecond Precision**: CPU cycle monitoring for accurate progress tracking
- **Visual Feedback**: Professional progress bars with detailed status messages

### Rollback Support & Safety
- **MSI Transactions**: Automatic journaling via Windows Installer engine
- **Custom Rollback**: XML/C++ actions to purge partial installs completely
- **Registry Cleanup**: Removes registry entries and temporary files
- **Error Recovery**: Comprehensive error handling with detailed logging

### Professional Model Selection
- **Custom UI Wizard**: Checkbox tree interface for ~50GB of AI models
- **HuggingFace Integration**: Python downloader scripts with API integration
- **Conditional Installation**: User-selectable model packages
- **LZMA Compression**: Minimal installer bloat with efficient compression

### GPU Detection & Configuration
- **Multi-vendor Support**: NVIDIA (NVAPI.dll), AMD (DirectML), Intel GPUs
- **CUDA Auto-install**: Detects GPU and downloads CUDA 12+ toolkit automatically
- **Environment Setup**: Configures CUDA_VISIBLE_DEVICES and PATH variables
- **CPU Fallback**: Graceful degradation to CPU processing when needed

### System Integration
- **Path Configuration**: Auto-appends application directories to %PATH%
- **Registry Persistence**: Reboot-proof environment variables in HKLM
- **File Associations**: Registers .aiart project file extensions
- **Shortcuts**: Desktop, Start Menu, and taskbar pinning options

### First-Run Experience
- **Qt6 Configuration Wizard**: Post-install setup for API keys and preferences
- **GPU Re-scanning**: Validates hardware configuration after installation
- **Model Management**: Configure download preferences and storage locations
- **Tutorial Integration**: Optional guided tour and sample project creation

## 📁 Directory Structure

```
installer/
├── wix/                    # WiX Toolset source files
│   ├── Bundle.wxs         # Main bundle configuration with chaining
│   ├── AIArtworks.wxs     # Core application MSI definition
│   └── AIArtworksCPP.wxs  # C++ runtime components (auto-generated)
├── scripts/               # Build and custom action scripts
│   ├── build_installer.ps1    # Master build script
│   ├── CustomActions.cpp      # C++ DLL for system detection
│   ├── CustomActions.def      # DLL export definitions
│   ├── SystemCheck.ps1        # PowerShell system requirements check
│   └── FirstRunWizard.py      # Qt6 post-install configuration wizard
├── ui/                    # User interface themes and resources
│   ├── theme.xml          # WiX Burn custom theme
│   └── localization.wxl   # Localization strings
├── resources/             # Graphics and assets
│   ├── app_icon.ico       # Application icon
│   ├── banner.bmp         # Installer banner image
│   ├── dialog.bmp         # Dialog background
│   ├── license.rtf        # License agreement
│   └── logo.png           # Company logo
├── dependencies/          # External redistributables (downloaded)
│   ├── python-3.12.0-amd64.exe
│   ├── VC_redist.x64.exe
│   └── cuda_12.3.0_windows_x86_64.exe
└── output/               # Build output directory
    ├── AI-ARTWORKS-Setup.exe  # Final bundle installer
    ├── AIArtworks.msi         # Core application MSI
    ├── AIArtworksCPP.msi      # C++ runtime MSI
    └── BuildSummary.txt       # Build report
```

## 🔧 Prerequisites

### Development Environment
- **Windows 10/11** (64-bit)
- **Visual Studio 2022** with C++ build tools
- **WiX Toolset v4** (install from wixtoolset.org)
- **Python 3.10+** with pip
- **CMake 3.20+**
- **Git** for version control

### Optional Tools
- **Code Signing Certificate** (for production releases)
- **NVIDIA CUDA SDK** (for GPU acceleration testing)
- **Windows SDK** (latest version)

## 🏗️ Build Process

### Quick Start
```powershell
# Navigate to installer directory
cd installer

# Run the master build script
.\scripts\build_installer.ps1

# For clean build with all options
.\scripts\build_installer.ps1 -Clean -Configuration Release -Platform x64
```

### Step-by-Step Build

1. **Prepare Environment**
   ```powershell
   # Verify prerequisites
   .\scripts\build_installer.ps1 -WhatIf
   
   # Download dependencies
   .\scripts\build_installer.ps1 -DownloadOnly
   ```

2. **Build C++ Components**
   ```powershell
   # Build main library and custom actions DLL
   .\scripts\build_installer.ps1 -CppOnly
   ```

3. **Create Python Executable**
   ```powershell
   # Package Python app with PyInstaller
   .\scripts\build_installer.ps1 -PyInstallerOnly
   ```

4. **Build MSI Packages**
   ```powershell
   # Compile WiX source files to MSI
   .\scripts\build_installer.ps1 -MSIOnly
   ```

5. **Create Final Bundle**
   ```powershell
   # Chain MSIs into single executable
   .\scripts\build_installer.ps1 -BundleOnly
   ```

### Build Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-Configuration` | Build configuration (Debug/Release) | Release |
| `-Platform` | Target platform (x64/x86) | x64 |
| `-OutputDir` | Output directory path | .\output |
| `-Clean` | Clean build directories | false |
| `-SkipCppBuild` | Skip C++ compilation | false |
| `-SkipPyInstaller` | Skip Python packaging | false |

## 🎯 Installation Features

### System Requirements Check
- **OS Version**: Windows 7 SP1 or later (64-bit)
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space (50GB+ for full model collection)
- **Graphics**: DirectX 11 compatible GPU (CUDA optional)
- **Network**: Internet connection for model downloads

### Dependency Management
The installer automatically handles these prerequisites:

1. **Python 3.10+**
   - Silent installation with all users scope
   - PATH environment variable configuration
   - Virtual environment creation for isolation

2. **Visual C++ Redistributable 2015-2022**
   - Latest x64 runtime libraries
   - Shared components for C++ applications
   - Automatic updates and patches

3. **NVIDIA CUDA Toolkit 12.3** (conditional)
   - Only installed if NVIDIA GPU detected
   - Includes cuDNN libraries for deep learning
   - Environment variable configuration

### GPU Configuration
- **Detection**: Comprehensive GPU enumeration via DirectX and WMI
- **CUDA Support**: Automatic CUDA toolkit installation for NVIDIA GPUs
- **DirectML**: Windows Machine Learning acceleration for AMD/Intel
- **Fallback**: CPU-only processing when no suitable GPU found

### Model Management
The installer supports selective model downloading:

#### Image Generation (13GB total)
- Stable Diffusion 1.5 (4GB) - General purpose
- Stable Diffusion XL (7GB) - High resolution
- ControlNet (2GB) - Guided generation

#### Voice & Audio (5.3GB total)
- Whisper Base (290MB) - Speech recognition
- Whisper Large (3GB) - High accuracy transcription
- Bark (2GB) - Text-to-speech synthesis

#### Language Models (26GB total)
- LLaMA 7B (13GB) - Large language model
- Code Llama (13GB) - Code generation

## 🎨 User Interface

### Modern Theme
- **Design Language**: Microsoft Fluent Design principles
- **Typography**: Segoe UI font family with proper weights
- **Colors**: Windows 10/11 accent color integration
- **Layout**: Responsive design with proper spacing

### Progress Visualization
- **Overall Progress**: Bundle-level installation progress
- **Package Progress**: Individual MSI/EXE installation status
- **Detailed Logging**: Expandable log viewer with real-time updates
- **Error Handling**: User-friendly error messages with troubleshooting

### First-Run Wizard
Post-installation configuration wizard features:
- **System Scanning**: GPU detection and capability assessment
- **Model Selection**: Interactive model download interface
- **API Configuration**: Optional service integration setup
- **Performance Tuning**: Hardware-specific optimization

## 🔒 Security & Code Signing

### Code Signing Setup
```powershell
# Set certificate thumbprint environment variable
$env:CODE_SIGN_CERT_THUMBPRINT = "YOUR_CERT_THUMBPRINT_HERE"

# Build with automatic signing
.\scripts\build_installer.ps1 -Sign
```

### Security Features
- **Authenticode Signing**: Digital signature verification
- **Timestamp Authority**: RFC 3161 timestamping for long-term validity
- **Certificate Validation**: Chain of trust verification
- **SmartScreen Compatibility**: Windows Defender integration

## 📊 Testing & Validation

### Test Scenarios
1. **Clean Installation**: Fresh Windows system without prerequisites
2. **Upgrade Installation**: Existing installation with version detection
3. **Partial Prerequisites**: Systems with some components already installed
4. **Network Failures**: Offline installation and download retry logic
5. **Insufficient Resources**: Low disk space and memory conditions
6. **GPU Variations**: NVIDIA, AMD, Intel, and integrated graphics

### Automated Testing
```powershell
# Run system compatibility check
.\scripts\SystemCheck.ps1 -Mode TEST

# Validate installer integrity
.\scripts\validate_installer.ps1 -Path .\output\AI-ARTWORKS-Setup.exe

# Test uninstall process
.\scripts\test_uninstall.ps1
```

## 🚀 Deployment

### Distribution Channels
- **Direct Download**: Company website with HTTPS delivery
- **Package Managers**: Chocolatey, Scoop, WinGet integration
- **Enterprise**: Group Policy deployment and SCCM packages
- **Cloud Storage**: Azure Blob Storage with CDN acceleration

### Release Process
1. **Version Increment**: Update version strings in all WiX files
2. **Build Validation**: Automated testing on clean VMs
3. **Code Signing**: Apply Authenticode signature with EV certificate
4. **Upload**: Secure transfer to distribution infrastructure
5. **Announcement**: Release notes and download links

## 🛠️ Troubleshooting

### Common Issues

#### Build Failures
```powershell
# Check WiX installation
wix --version

# Verify MSBuild path
where msbuild

# Test Python environment
python --version
pip list | findstr pyinstaller
```

#### Installation Problems
- **Error 1603**: Generic MSI failure - check Windows Event Log
- **Error 2755**: Server execution failed - run as administrator
- **CUDA Installation**: Verify NVIDIA driver compatibility
- **Python Issues**: Check PATH environment variable

### Debug Mode
```powershell
# Enable verbose logging
.\scripts\build_installer.ps1 -Verbose -Debug

# MSI installation with logging
msiexec /i AIArtworks.msi /l*v install.log

# Bundle installation with logging
AI-ARTWORKS-Setup.exe /log install_bundle.log
```

### Support Resources
- **Documentation**: Complete user manual and developer guide
- **Community Forum**: User discussions and troubleshooting
- **Issue Tracker**: Bug reports and feature requests
- **Professional Support**: Enterprise support contracts available

## 📈 Performance Optimization

### Build Performance
- **Parallel Compilation**: Multi-threaded C++ builds
- **Incremental Builds**: Only rebuild changed components
- **Caching**: Dependency download caching
- **Compression**: LZMA compression for minimal file sizes

### Installation Performance
- **Silent Installation**: Non-interactive prerequisite installation
- **Background Downloads**: Concurrent model downloading
- **Delta Updates**: Incremental updates for existing installations
- **SSD Optimization**: Sequential file operations for better performance

## 🔄 Maintenance & Updates

### Automatic Updates
The installer includes infrastructure for automatic updates:
- **Update Server**: Checks for new versions periodically
- **Delta Patches**: Download only changed files
- **Background Installation**: Non-disruptive update process
- **Rollback Capability**: Restore previous version if needed

### Version Management
- **Semantic Versioning**: Major.Minor.Patch.Build format
- **Compatibility Matrix**: Supported OS and hardware combinations
- **Deprecation Policy**: Advance notice for discontinued features
- **Migration Tools**: Data and settings preservation across versions

---

## 📞 Contact & Support

- **Documentation**: [docs.ai-artworks.com](https://docs.ai-artworks.com)
- **Community**: [community.ai-artworks.com](https://community.ai-artworks.com)
- **Issues**: [github.com/your-org/ai-artworks/issues](https://github.com/your-org/ai-artworks/issues)
- **Email**: support@ai-artworks.com

---

*Built with ❤️ using WiX Toolset, Qt6, and modern Windows development practices.*