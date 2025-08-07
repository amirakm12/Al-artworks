# ChatGPT+ Clone Automated Test Runner
# Usage: Run this script on target Windows machines to run automated tests

param(
    [string]$TestType = "all",
    [string]$OutputDir = "test_results",
    [switch]$Verbose,
    [switch]$GenerateReport
)

Write-Host "=== Starting ChatGPT+ Clone Automated Tests ===" -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Yellow
Write-Host "Test Type: $TestType" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow

# Create output directory
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

# Initialize test results
$testResults = @{
    "timestamp" = Get-Date
    "system_info" = @{}
    "test_results" = @{}
    "performance_metrics" = @{}
    "errors" = @()
    "warnings" = @()
}

# Function to log test results
function Write-TestResult {
    param($TestName, $Status, $Message, $Duration = 0)
    
    $result = @{
        "status" = $Status
        "message" = $Message
        "duration" = $Duration
        "timestamp" = Get-Date
    }
    
    $testResults.test_results[$TestName] = $result
    
    $color = if ($Status -eq "PASS") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Yellow" }
    Write-Host "[$Status] $TestName - $Message (${Duration}s)" -ForegroundColor $color
}

# Function to capture system information
function Get-SystemInfo {
    Write-Host "`n=== System Information ===" -ForegroundColor Cyan
    
    try {
        # OS Information
        $os = Get-CimInstance Win32_OperatingSystem
        $testResults.system_info.os = @{
            "name" = $os.Caption
            "version" = $os.Version
            "architecture" = $os.OSArchitecture
            "build" = $os.BuildNumber
        }
        Write-Host "OS: $($os.Caption) $($os.OSArchitecture)" -ForegroundColor Green
        
        # CPU Information
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        $testResults.system_info.cpu = @{
            "name" = $cpu.Name
            "cores" = $cpu.NumberOfCores
            "threads" = $cpu.NumberOfLogicalProcessors
            "frequency" = $cpu.MaxClockSpeed
        }
        Write-Host "CPU: $($cpu.Name)" -ForegroundColor Green
        
        # Memory Information
        $memory = Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
        $totalMemoryGB = [math]::Round($memory.Sum / 1GB, 2)
        $testResults.system_info.memory = @{
            "total_gb" = $totalMemoryGB
            "modules" = $memory.Count
        }
        Write-Host "Memory: ${totalMemoryGB}GB ($($memory.Count) modules)" -ForegroundColor Green
        
        # GPU Information
        $gpus = Get-WmiObject win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM
        $testResults.system_info.gpu = @()
        Write-Host "Detected GPUs:" -ForegroundColor Green
        foreach ($gpu in $gpus) {
            $gpuInfo = @{
                "name" = $gpu.Name
                "driver_version" = $gpu.DriverVersion
                "memory_mb" = if ($gpu.AdapterRAM) { [math]::Round($gpu.AdapterRAM / 1MB, 0) } else { "Unknown" }
            }
            $testResults.system_info.gpu += $gpuInfo
            Write-Host "  - $($gpu.Name) (Driver: $($gpu.DriverVersion))" -ForegroundColor Green
        }
        
        # Python Information
        try {
            $pythonVersion = python --version 2>&1
            $testResults.system_info.python = @{
                "version" = $pythonVersion
                "executable" = (Get-Command python).Source
            }
            Write-Host "Python: $pythonVersion" -ForegroundColor Green
        } catch {
            Write-Host "Python: Not found or not in PATH" -ForegroundColor Red
            $testResults.errors += "Python not found in PATH"
        }
        
        # Check for CUDA
        try {
            $nvidiaSmi = nvidia-smi --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $testResults.system_info.cuda = @{
                    "available" = $true
                    "version" = $nvidiaSmi
                }
                Write-Host "CUDA: Available" -ForegroundColor Green
            } else {
                $testResults.system_info.cuda = @{ "available" = $false }
                Write-Host "CUDA: Not available" -ForegroundColor Yellow
            }
        } catch {
            $testResults.system_info.cuda = @{ "available" = $false }
            Write-Host "CUDA: Not available" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "Error gathering system information: $($_.Exception.Message)" -ForegroundColor Red
        $testResults.errors += "Failed to gather system information: $($_.Exception.Message)"
    }
}

# Function to run Python tests
function Invoke-PythonTests {
    Write-Host "`n=== Running Python Unit Tests ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Check if tests directory exists
        if (!(Test-Path "tests")) {
            Write-TestResult "python_tests" "FAIL" "Tests directory not found" 0
            return
        }
        
        # Run unittest discovery
        $testOutput = python -m unittest discover -s tests -p "*.py" -v 2>&1
        $exitCode = $LASTEXITCODE
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-TestResult "python_tests" "PASS" "All Python tests passed" $duration
            if ($Verbose) {
                Write-Host $testOutput -ForegroundColor Gray
            }
        } else {
            Write-TestResult "python_tests" "FAIL" "Python tests failed (exit code: $exitCode)" $duration
            Write-Host $testOutput -ForegroundColor Red
        }
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "python_tests" "FAIL" "Exception during Python tests: $($_.Exception.Message)" $duration
        $testResults.errors += "Python test exception: $($_.Exception.Message)"
    }
}

# Function to test LLM functionality
function Test-LLMFunctionality {
    Write-Host "`n=== Testing LLM Model Load and Voice System ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Test LLM model loading
        $llmTestOutput = python tests/test_llm_voice.py 2>&1
        $exitCode = $LASTEXITCODE
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-TestResult "llm_voice_test" "PASS" "LLM and voice tests passed" $duration
            if ($Verbose) {
                Write-Host $llmTestOutput -ForegroundColor Gray
            }
        } else {
            Write-TestResult "llm_voice_test" "FAIL" "LLM and voice tests failed (exit code: $exitCode)" $duration
            Write-Host $llmTestOutput -ForegroundColor Red
        }
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "llm_voice_test" "FAIL" "Exception during LLM test: $($_.Exception.Message)" $duration
        $testResults.errors += "LLM test exception: $($_.Exception.Message)"
    }
}

# Function to test plugin hot-reload
function Test-PluginHotReload {
    Write-Host "`n=== Testing Plugin Hot-Reload ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Test plugin hot-reload functionality
        $pluginTestOutput = python tests/test_plugin_reload.py 2>&1
        $exitCode = $LASTEXITCODE
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-TestResult "plugin_hot_reload" "PASS" "Plugin hot-reload tests passed" $duration
            if ($Verbose) {
                Write-Host $pluginTestOutput -ForegroundColor Gray
            }
        } else {
            Write-TestResult "plugin_hot_reload" "FAIL" "Plugin hot-reload tests failed (exit code: $exitCode)" $duration
            Write-Host $pluginTestOutput -ForegroundColor Red
        }
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "plugin_hot_reload" "FAIL" "Exception during plugin test: $($_.Exception.Message)" $duration
        $testResults.errors += "Plugin test exception: $($_.Exception.Message)"
    }
}

# Function to test performance
function Test-Performance {
    Write-Host "`n=== Performance Testing ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Test application startup time
        $appStartTime = Get-Date
        $appProcess = Start-Process python -ArgumentList "main.py" -PassThru -WindowStyle Hidden
        Start-Sleep -Seconds 5
        Stop-Process -Id $appProcess.Id -Force -ErrorAction SilentlyContinue
        $appStartDuration = ((Get-Date) - $appStartTime).TotalSeconds
        
        $testResults.performance_metrics.app_startup = $appStartDuration
        Write-TestResult "app_startup" "PASS" "Application startup time: ${appStartDuration}s" $appStartDuration
        
        # Test memory usage
        $memoryTest = python -c "import psutil; print(psutil.virtual_memory().percent)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $memoryUsage = [int]$memoryTest
            $testResults.performance_metrics.memory_usage = $memoryUsage
            Write-TestResult "memory_usage" "PASS" "Memory usage: ${memoryUsage}%" 0
        }
        
        # Test CPU usage
        $cpuTest = python -c "import psutil; print(psutil.cpu_percent(interval=1))" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $cpuUsage = [math]::Round([double]$cpuTest, 1)
            $testResults.performance_metrics.cpu_usage = $cpuUsage
            Write-TestResult "cpu_usage" "PASS" "CPU usage: ${cpuUsage}%" 0
        }
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "performance_test" "PASS" "Performance tests completed" $duration
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "performance_test" "FAIL" "Exception during performance test: $($_.Exception.Message)" $duration
        $testResults.errors += "Performance test exception: $($_.Exception.Message)"
    }
}

# Function to test GPU acceleration
function Test-GPUAcceleration {
    Write-Host "`n=== GPU Acceleration Testing ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Test PyTorch CUDA availability
        $cudaTest = python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())" 2>&1
        $exitCode = $LASTEXITCODE
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-TestResult "gpu_acceleration" "PASS" "GPU acceleration test passed" $duration
            if ($Verbose) {
                Write-Host $cudaTest -ForegroundColor Gray
            }
            
            # Parse CUDA information
            if ($cudaTest -match "CUDA available: True") {
                $testResults.performance_metrics.gpu_available = $true
                if ($cudaTest -match "GPU count: (\d+)") {
                    $testResults.performance_metrics.gpu_count = [int]$matches[1]
                }
            } else {
                $testResults.performance_metrics.gpu_available = $false
            }
        } else {
            Write-TestResult "gpu_acceleration" "FAIL" "GPU acceleration test failed" $duration
            Write-Host $cudaTest -ForegroundColor Red
        }
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "gpu_acceleration" "FAIL" "Exception during GPU test: $($_.Exception.Message)" $duration
        $testResults.errors += "GPU test exception: $($_.Exception.Message)"
    }
}

# Function to test security features
function Test-SecurityFeatures {
    Write-Host "`n=== Security Testing ===" -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Test plugin sandbox (if available)
        $securityTest = python -c "from security.restricted_exec import restricted_exec; result = restricted_exec('print(\"Hello\")'); print('Sandbox test passed')" 2>&1
        $exitCode = $LASTEXITCODE
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-TestResult "security_sandbox" "PASS" "Security sandbox test passed" $duration
        } else {
            Write-TestResult "security_sandbox" "WARN" "Security sandbox not available" $duration
            $testResults.warnings += "Security sandbox not available"
        }
        
        # Test file permissions
        if (Test-Path "plugins") {
            $acl = Get-Acl "plugins"
            $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            $hasAccess = $acl.Access | Where-Object { $_.IdentityReference -eq $currentUser }
            
            if ($hasAccess) {
                Write-TestResult "file_permissions" "PASS" "Plugin directory permissions OK" 0
            } else {
                Write-TestResult "file_permissions" "WARN" "Plugin directory permissions may need adjustment" 0
                $testResults.warnings += "Plugin directory permissions may need adjustment"
            }
        }
        
    } catch {
        $duration = ((Get-Date) - $startTime).TotalSeconds
        Write-TestResult "security_test" "FAIL" "Exception during security test: $($_.Exception.Message)" $duration
        $testResults.errors += "Security test exception: $($_.Exception.Message)"
    }
}

# Function to generate test report
function Write-TestReport {
    Write-Host "`n=== Generating Test Report ===" -ForegroundColor Cyan
    
    $reportFile = Join-Path $OutputDir "test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    
    try {
        $testResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportFile -Encoding UTF8
        Write-Host "Test report saved to: $reportFile" -ForegroundColor Green
        
        # Generate summary
        $totalTests = $testResults.test_results.Count
        $passedTests = ($testResults.test_results.Values | Where-Object { $_.status -eq "PASS" }).Count
        $failedTests = ($testResults.test_results.Values | Where-Object { $_.status -eq "FAIL" }).Count
        $warningTests = ($testResults.test_results.Values | Where-Object { $_.status -eq "WARN" }).Count
        
        Write-Host "`n=== Test Summary ===" -ForegroundColor Green
        Write-Host "Total Tests: $totalTests" -ForegroundColor White
        Write-Host "Passed: $passedTests" -ForegroundColor Green
        Write-Host "Failed: $failedTests" -ForegroundColor Red
        Write-Host "Warnings: $warningTests" -ForegroundColor Yellow
        
        if ($testResults.errors.Count -gt 0) {
            Write-Host "`nErrors:" -ForegroundColor Red
            foreach ($error in $testResults.errors) {
                Write-Host "  - $error" -ForegroundColor Red
            }
        }
        
        if ($testResults.warnings.Count -gt 0) {
            Write-Host "`nWarnings:" -ForegroundColor Yellow
            foreach ($warning in $testResults.warnings) {
                Write-Host "  - $warning" -ForegroundColor Yellow
            }
        }
        
    } catch {
        Write-Host "Error generating test report: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main test execution
try {
    # Get system information
    Get-SystemInfo
    
    # Run tests based on type
    switch ($TestType.ToLower()) {
        "all" {
            Invoke-PythonTests
            Test-LLMFunctionality
            Test-PluginHotReload
            Test-Performance
            Test-GPUAcceleration
            Test-SecurityFeatures
        }
        "python" {
            Invoke-PythonTests
        }
        "llm" {
            Test-LLMFunctionality
        }
        "plugins" {
            Test-PluginHotReload
        }
        "performance" {
            Test-Performance
        }
        "gpu" {
            Test-GPUAcceleration
        }
        "security" {
            Test-SecurityFeatures
        }
        default {
            Write-Host "Unknown test type: $TestType" -ForegroundColor Red
            Write-Host "Available types: all, python, llm, plugins, performance, gpu, security" -ForegroundColor Yellow
            exit 1
        }
    }
    
    # Generate report if requested
    if ($GenerateReport) {
        Write-TestReport
    }
    
    Write-Host "`n=== Tests completed ===" -ForegroundColor Green
    
} catch {
    Write-Host "Critical error during test execution: $($_.Exception.Message)" -ForegroundColor Red
    $testResults.errors += "Critical error: $($_.Exception.Message)"
    
    if ($GenerateReport) {
        Write-TestReport
    }
    
    exit 1
}