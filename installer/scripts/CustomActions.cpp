#include <windows.h>
#include <msiquery.h>
#include <msi.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <thread>
#include <chrono>
#include <winreg.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <nvapi.h>  // NVIDIA API for GPU detection
#include <d3d11.h>  // DirectX for GPU enumeration
#include <dxgi.h>

#pragma comment(lib, "msi.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Logging utility
void LogMessage(MSIHANDLE hInstall, const std::wstring& message) {
    PMSIHANDLE hRecord = MsiCreateRecord(1);
    MsiRecordSetStringW(hRecord, 1, message.c_str());
    MsiProcessMessage(hInstall, INSTALLMESSAGE_INFO, hRecord);
}

// Registry utility functions
bool ReadRegistryString(HKEY hKey, const std::wstring& subKey, const std::wstring& valueName, std::wstring& result) {
    HKEY hSubKey;
    if (RegOpenKeyExW(hKey, subKey.c_str(), 0, KEY_READ, &hSubKey) != ERROR_SUCCESS) {
        return false;
    }

    DWORD dataSize = 0;
    if (RegQueryValueExW(hSubKey, valueName.c_str(), nullptr, nullptr, nullptr, &dataSize) != ERROR_SUCCESS) {
        RegCloseKey(hSubKey);
        return false;
    }

    std::vector<wchar_t> buffer(dataSize / sizeof(wchar_t));
    if (RegQueryValueExW(hSubKey, valueName.c_str(), nullptr, nullptr, 
                        reinterpret_cast<LPBYTE>(buffer.data()), &dataSize) == ERROR_SUCCESS) {
        result = std::wstring(buffer.data());
        RegCloseKey(hSubKey);
        return true;
    }

    RegCloseKey(hSubKey);
    return false;
}

// GPU Detection using NVAPI and DirectX
struct GPUInfo {
    std::wstring name;
    std::wstring vendor;
    size_t dedicatedMemory;
    bool supportsCUDA;
    bool supportsDirectML;
};

std::vector<GPUInfo> DetectGPUs() {
    std::vector<GPUInfo> gpus;

    // Initialize NVAPI for NVIDIA detection
    NvAPI_Status nvStatus = NvAPI_Initialize();
    bool nvapiInitialized = (nvStatus == NVAPI_OK);

    // Use DXGI to enumerate all adapters
    IDXGIFactory* pFactory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory))) {
        UINT adapterIndex = 0;
        IDXGIAdapter* pAdapter = nullptr;

        while (pFactory->EnumAdapters(adapterIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC desc;
            if (SUCCEEDED(pAdapter->GetDesc(&desc))) {
                GPUInfo gpu;
                gpu.name = desc.Description;
                gpu.dedicatedMemory = desc.DedicatedVideoMemory;
                
                // Determine vendor
                if (desc.VendorId == 0x10DE) {
                    gpu.vendor = L"NVIDIA";
                    gpu.supportsCUDA = true;
                } else if (desc.VendorId == 0x1002) {
                    gpu.vendor = L"AMD";
                    gpu.supportsCUDA = false;
                } else if (desc.VendorId == 0x8086) {
                    gpu.vendor = L"Intel";
                    gpu.supportsCUDA = false;
                } else {
                    gpu.vendor = L"Unknown";
                    gpu.supportsCUDA = false;
                }

                // Check DirectML support (Windows 10 1903+)
                gpu.supportsDirectML = true; // Assume modern DirectX support

                gpus.push_back(gpu);
            }
            pAdapter->Release();
            adapterIndex++;
        }
        pFactory->Release();
    }

    if (nvapiInitialized) {
        NvAPI_Unload();
    }

    return gpus;
}

// System Requirements Check
bool CheckSystemRequirements(MSIHANDLE hInstall) {
    LogMessage(hInstall, L"Starting system requirements check...");

    // Check OS version (Windows 7 or later)
    OSVERSIONINFOEXW osvi = {};
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXW);
    GetVersionExW((OSVERSIONINFOW*)&osvi);

    if (osvi.dwMajorVersion < 6 || (osvi.dwMajorVersion == 6 && osvi.dwMinorVersion < 1)) {
        LogMessage(hInstall, L"ERROR: Windows 7 or later required");
        return false;
    }

    // Check available memory (minimum 8GB)
    MEMORYSTATUSEX memStatus = {};
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memStatus);
    
    DWORDLONG minMemory = 8ULL * 1024 * 1024 * 1024; // 8GB
    if (memStatus.ullTotalPhys < minMemory) {
        LogMessage(hInstall, L"WARNING: Minimum 8GB RAM recommended");
    }

    // Check available disk space (minimum 10GB)
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes;
    wchar_t systemDrive[MAX_PATH];
    GetEnvironmentVariableW(L"SystemDrive", systemDrive, MAX_PATH);
    wcscat_s(systemDrive, L"\\");

    if (GetDiskFreeSpaceExW(systemDrive, &freeBytesAvailable, &totalNumberOfBytes, nullptr)) {
        ULONGLONG minDiskSpace = 10ULL * 1024 * 1024 * 1024; // 10GB
        if (freeBytesAvailable.QuadPart < minDiskSpace) {
            LogMessage(hInstall, L"ERROR: Insufficient disk space (10GB required)");
            return false;
        }
    }

    LogMessage(hInstall, L"System requirements check completed successfully");
    return true;
}

// Python Environment Setup
extern "C" __declspec(dllexport) UINT __stdcall SetupPythonEnvironment(MSIHANDLE hInstall) {
    LogMessage(hInstall, L"Setting up Python environment...");

    try {
        // Get installation directory
        DWORD installDirSize = 0;
        MsiGetPropertyW(hInstall, L"INSTALLFOLDER", L"", &installDirSize);
        std::vector<wchar_t> installDir(installDirSize);
        MsiGetPropertyW(hInstall, L"INSTALLFOLDER", installDir.data(), &installDirSize);

        std::wstring installPath(installDir.data());
        std::wstring requirementsPath = installPath + L"config\\requirements.txt";

        // Check if Python is available
        std::wstring pythonPath;
        bool pythonFound = false;
        
        // Try Python 3.12, 3.11, 3.10
        std::vector<std::wstring> pythonVersions = {L"3.12", L"3.11", L"3.10"};
        for (const auto& version : pythonVersions) {
            std::wstring regKey = L"SOFTWARE\\Python\\PythonCore\\" + version + L"\\InstallPath";
            if (ReadRegistryString(HKEY_LOCAL_MACHINE, regKey, L"", pythonPath)) {
                pythonFound = true;
                break;
            }
        }

        if (!pythonFound) {
            LogMessage(hInstall, L"ERROR: Python 3.10+ not found");
            return ERROR_INSTALL_FAILURE;
        }

        // Create virtual environment
        std::wstring venvPath = installPath + L"venv";
        std::wstring createVenvCmd = pythonPath + L"python.exe -m venv \"" + venvPath + L"\"";
        
        STARTUPINFOW si = {};
        PROCESS_INFORMATION pi = {};
        si.cb = sizeof(STARTUPINFOW);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;

        if (CreateProcessW(nullptr, &createVenvCmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, 30000); // 30 second timeout
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        } else {
            LogMessage(hInstall, L"ERROR: Failed to create virtual environment");
            return ERROR_INSTALL_FAILURE;
        }

        // Install requirements
        std::wstring pipPath = venvPath + L"\\Scripts\\pip.exe";
        std::wstring installCmd = L"\"" + pipPath + L"\" install -r \"" + requirementsPath + L"\" --no-warn-script-location";
        
        if (CreateProcessW(nullptr, &installCmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, 300000); // 5 minute timeout
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        } else {
            LogMessage(hInstall, L"ERROR: Failed to install Python requirements");
            return ERROR_INSTALL_FAILURE;
        }

        // Set environment variables
        MsiSetPropertyW(hInstall, L"PYTHON_VENV_PATH", venvPath.c_str());
        
        LogMessage(hInstall, L"Python environment setup completed successfully");
        return ERROR_SUCCESS;

    } catch (const std::exception& e) {
        LogMessage(hInstall, L"ERROR: Exception in SetupPythonEnvironment");
        return ERROR_INSTALL_FAILURE;
    }
}

// AI Models Download
extern "C" __declspec(dllexport) UINT __stdcall DownloadAIModels(MSIHANDLE hInstall) {
    LogMessage(hInstall, L"Starting AI models download...");

    try {
        // Get installation directory
        DWORD installDirSize = 0;
        MsiGetPropertyW(hInstall, L"INSTALLFOLDER", L"", &installDirSize);
        std::vector<wchar_t> installDir(installDirSize);
        MsiGetPropertyW(hInstall, L"INSTALLFOLDER", installDir.data(), &installDirSize);

        std::wstring installPath(installDir.data());
        std::wstring modelsPath = installPath + L"models";
        std::wstring venvPath = installPath + L"venv";
        std::wstring pythonPath = venvPath + L"\\Scripts\\python.exe";

        // Create models directory
        CreateDirectoryW(modelsPath.c_str(), nullptr);

        // Download script content
        std::wstring downloadScript = LR"(
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

def download_with_progress(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def main():
    models_dir = Path(sys.argv[1])
    models_dir.mkdir(exist_ok=True)
    
    # Download essential models
    models = [
        {
            'repo_id': 'runwayml/stable-diffusion-v1-5',
            'filename': 'v1-5-pruned-emaonly.ckpt',
            'subfolder': 'checkpoints'
        },
        {
            'repo_id': 'openai/whisper-base',
            'filename': 'pytorch_model.bin',
            'subfolder': 'whisper'
        }
    ]
    
    for model in models:
        try:
            print(f"Downloading {model['repo_id']}...")
            local_dir = models_dir / model['subfolder']
            local_dir.mkdir(exist_ok=True)
            
            hf_hub_download(
                repo_id=model['repo_id'],
                filename=model['filename'],
                local_dir=str(local_dir),
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {model['filename']}")
        except Exception as e:
            print(f"Error downloading {model['repo_id']}: {e}")
    
    print("Model download completed!")

if __name__ == "__main__":
    main()
)";

        // Write download script to temp file
        std::wstring scriptPath = installPath + L"temp_download_models.py";
        std::wofstream scriptFile(scriptPath);
        scriptFile << downloadScript;
        scriptFile.close();

        // Execute download script
        std::wstring downloadCmd = L"\"" + pythonPath + L"\" \"" + scriptPath + L"\" \"" + modelsPath + L"\"";
        
        STARTUPINFOW si = {};
        PROCESS_INFORMATION pi = {};
        si.cb = sizeof(STARTUPINFOW);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;

        if (CreateProcessW(nullptr, &downloadCmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, 1800000); // 30 minute timeout
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }

        // Clean up temp script
        DeleteFileW(scriptPath.c_str());

        LogMessage(hInstall, L"AI models download completed");
        return ERROR_SUCCESS;

    } catch (const std::exception& e) {
        LogMessage(hInstall, L"ERROR: Exception in DownloadAIModels");
        return ERROR_INSTALL_FAILURE;
    }
}

// GPU Detection and Configuration
extern "C" __declspec(dllexport) UINT __stdcall DetectAndConfigureGPU(MSIHANDLE hInstall) {
    LogMessage(hInstall, L"Detecting GPU configuration...");

    try {
        auto gpus = DetectGPUs();
        
        std::wstringstream gpuInfo;
        bool nvidiaFound = false;
        bool amdFound = false;
        
        for (const auto& gpu : gpus) {
            gpuInfo << gpu.vendor << L" " << gpu.name << L" (" 
                   << (gpu.dedicatedMemory / (1024 * 1024)) << L"MB); ";
            
            if (gpu.vendor == L"NVIDIA") {
                nvidiaFound = true;
            } else if (gpu.vendor == L"AMD") {
                amdFound = true;
            }
        }

        // Set MSI properties based on GPU detection
        MsiSetPropertyW(hInstall, L"GPU_INFO", gpuInfo.str().c_str());
        MsiSetPropertyW(hInstall, L"NVIDIA_GPU_DETECTED", nvidiaFound ? L"1" : L"0");
        MsiSetPropertyW(hInstall, L"AMD_GPU_DETECTED", amdFound ? L"1" : L"0");
        MsiSetPropertyW(hInstall, L"CUDA_RECOMMENDED", nvidiaFound ? L"1" : L"0");

        LogMessage(hInstall, L"GPU detection completed: " + gpuInfo.str());
        return ERROR_SUCCESS;

    } catch (const std::exception& e) {
        LogMessage(hInstall, L"ERROR: Exception in DetectAndConfigureGPU");
        return ERROR_INSTALL_FAILURE;
    }
}

// DLL Entry Point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}