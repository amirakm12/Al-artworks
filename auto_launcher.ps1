# ChatGPT+ Clone Auto-Launcher
# Run this as administrator to set up automatic startup

param(
    [string]$PythonPath = "C:\Path\To\Python\python.exe",
    [string]$ScriptPath = "C:\Path\To\chatgpt-plus-clone\main.py",
    [string]$TaskName = "ChatGPTPlusCloneAutoRun"
)

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrator privileges. Please run as administrator."
    exit 1
}

Write-Host "Setting up ChatGPT+ Clone auto-launch..." -ForegroundColor Green

# Validate paths
if (-not (Test-Path $PythonPath)) {
    Write-Error "Python executable not found at: $PythonPath"
    Write-Host "Please update the PythonPath parameter to point to your Python installation."
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Error "Main script not found at: $ScriptPath"
    Write-Host "Please update the ScriptPath parameter to point to your main.py file."
    exit 1
}

# Create scheduled task action
$action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory (Split-Path $ScriptPath)

# Create trigger for user logon
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Create principal with highest privileges
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

# Register the scheduled task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
    Write-Host "✓ Scheduled Task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "  The ChatGPT+ Clone will now start automatically when you log in." -ForegroundColor Yellow
} catch {
    Write-Error "Failed to register scheduled task: $_"
    exit 1
}

# Verify the task was created
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "✓ Task verification successful" -ForegroundColor Green
    Write-Host "  Task Name: $($task.TaskName)"
    Write-Host "  Task State: $($task.State)"
    Write-Host "  Next Run Time: $($task.NextRunTime)"
} else {
    Write-Warning "Task was created but could not be verified"
}

Write-Host "`nSetup complete! The ChatGPT+ Clone will start automatically on your next login." -ForegroundColor Green
Write-Host "To test immediately, you can run: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow