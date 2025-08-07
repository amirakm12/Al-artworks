# ChatGPT+ Clone Security Hardening Script
# Hardens file permissions, IPC, and system security for the application

param(
    [string]$InstallPath = ".",
    [switch]$Verbose,
    [switch]$DryRun
)

Write-Host "=== ChatGPT+ Clone Security Hardening ===" -ForegroundColor Green
Write-Host "Install Path: $InstallPath" -ForegroundColor Yellow
Write-Host "Dry Run: $DryRun" -ForegroundColor Yellow

# Initialize security log
$securityLog = @{
    "timestamp" = Get-Date
    "actions_taken" = @()
    "warnings" = @()
    "errors" = @()
}

function Write-SecurityLog {
    param($Action, $Status, $Details = "")
    
    $logEntry = @{
        "action" = $Action
        "status" = $Status
        "details" = $Details
        "timestamp" = Get-Date
    }
    
    $securityLog.actions_taken += $logEntry
    
    $color = if ($Status -eq "SUCCESS") { "Green" } elseif ($Status -eq "WARNING") { "Yellow" } else { "Red" }
    Write-Host "[$Status] $Action" -ForegroundColor $color
    if ($Details) {
        Write-Host "  Details: $Details" -ForegroundColor Gray
    }
}

function Test-AdminRights {
    """Test if running with administrative privileges"""
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Host "⚠️  Warning: Not running with administrative privileges" -ForegroundColor Yellow
        Write-Host "Some security hardening operations may fail" -ForegroundColor Yellow
        $securityLog.warnings += "Not running with administrative privileges"
    }
    
    return $isAdmin
}

function Harden-PluginDirectory {
    """Harden plugin directory permissions"""
    $pluginsPath = Join-Path $InstallPath "plugins"
    
    if (!(Test-Path $pluginsPath)) {
        Write-SecurityLog "Create plugins directory" "SUCCESS" "Directory created"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $pluginsPath -Force | Out-Null
        }
    }
    
    try {
        # Get current user
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        
        # Remove inheritance and set restrictive permissions
        if (-not $DryRun) {
            # Remove inheritance
            icacls $pluginsPath /inheritance:r /T
            
            # Grant full control to current user only
            icacls $pluginsPath /grant:r "${currentUser}:(OI)(CI)F" /T
            
            # Remove other users
            icacls $pluginsPath /remove "Users" "Everyone" "Authenticated Users" "INTERACTIVE" /T
            
            # Deny access to system accounts (except current user)
            icacls $pluginsPath /deny "SYSTEM:(OI)(CI)F" /T
            icacls $pluginsPath /deny "Administrators:(OI)(CI)F" /T
        }
        
        Write-SecurityLog "Harden plugin directory" "SUCCESS" "Restrictive permissions applied to $pluginsPath"
        
    } catch {
        Write-SecurityLog "Harden plugin directory" "ERROR" "Failed to set permissions: $($_.Exception.Message)"
        $securityLog.errors += "Plugin directory hardening failed: $($_.Exception.Message)"
    }
}

function Harden-ConfigFiles {
    """Harden configuration file permissions"""
    $configFiles = @(
        "config.json",
        "performance_config.json",
        "app.log",
        "errors.log"
    )
    
    foreach ($file in $configFiles) {
        $filePath = Join-Path $InstallPath $file
        
        if (Test-Path $filePath) {
            try {
                $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
                
                if (-not $DryRun) {
                    # Set restrictive permissions on config files
                    icacls $filePath /inheritance:r
                    icacls $filePath /grant:r "${currentUser}:F"
                    icacls $filePath /remove "Users" "Everyone" "Authenticated Users" /T
                }
                
                Write-SecurityLog "Harden config file" "SUCCESS" "Permissions hardened for $file"
                
            } catch {
                Write-SecurityLog "Harden config file" "ERROR" "Failed to harden $file: $($_.Exception.Message)"
                $securityLog.errors += "Config file hardening failed for $file: $($_.Exception.Message)"
            }
        }
    }
}

function Harden-TempDirectories {
    """Harden temporary directories"""
    $tempDirs = @(
        "temp",
        "cache",
        "logs"
    )
    
    foreach ($dir in $tempDirs) {
        $dirPath = Join-Path $InstallPath $dir
        
        if (Test-Path $dirPath) {
            try {
                $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
                
                if (-not $DryRun) {
                    # Set restrictive permissions on temp directories
                    icacls $dirPath /inheritance:r /T
                    icacls $dirPath /grant:r "${currentUser}:(OI)(CI)F" /T
                    icacls $dirPath /remove "Users" "Everyone" "Authenticated Users" /T
                }
                
                Write-SecurityLog "Harden temp directory" "SUCCESS" "Permissions hardened for $dir"
                
            } catch {
                Write-SecurityLog "Harden temp directory" "ERROR" "Failed to harden $dir: $($_.Exception.Message)"
                $securityLog.errors += "Temp directory hardening failed for $dir: $($_.Exception.Message)"
            }
        }
    }
}

function Disable-DangerousFeatures {
    """Disable potentially dangerous Windows features"""
    try {
        if (-not $DryRun) {
            # Disable PowerShell execution policy for current user only
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            
            # Disable Windows Script Host (optional)
            # reg add "HKCU\Software\Microsoft\Windows Script Host\Settings" /v Enabled /t REG_DWORD /d 0 /f
            
            # Disable autorun (optional)
            # reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer" /v NoDriveTypeAutoRun /t REG_DWORD /d 0xFF /f
        }
        
        Write-SecurityLog "Disable dangerous features" "SUCCESS" "Execution policy set to RemoteSigned"
        
    } catch {
        Write-SecurityLog "Disable dangerous features" "ERROR" "Failed to disable features: $($_.Exception.Message)"
        $securityLog.errors += "Feature disabling failed: $($_.Exception.Message)"
    }
}

function Set-NetworkSecurity {
    """Configure network security settings"""
    try {
        if (-not $DryRun) {
            # Block outbound connections to known malicious domains (example)
            # This is a basic example - in production, you'd want a more comprehensive approach
            
            # Set firewall rules for the application
            $appPath = Join-Path $InstallPath "main.py"
            if (Test-Path $appPath) {
                # Allow inbound connections for the app
                New-NetFirewallRule -DisplayName "ChatGPT+ Clone" -Direction Inbound -Program $appPath -Action Allow -ErrorAction SilentlyContinue
                
                # Allow outbound connections for the app
                New-NetFirewallRule -DisplayName "ChatGPT+ Clone Outbound" -Direction Outbound -Program $appPath -Action Allow -ErrorAction SilentlyContinue
            }
        }
        
        Write-SecurityLog "Set network security" "SUCCESS" "Firewall rules configured"
        
    } catch {
        Write-SecurityLog "Set network security" "ERROR" "Failed to configure network security: $($_.Exception.Message)"
        $securityLog.errors += "Network security configuration failed: $($_.Exception.Message)"
    }
}

function Test-SecurityIntegrity {
    """Test security integrity of the installation"""
    Write-Host "`n=== Security Integrity Check ===" -ForegroundColor Cyan
    
    $integrityChecks = @()
    
    # Check plugin directory permissions
    $pluginsPath = Join-Path $InstallPath "plugins"
    if (Test-Path $pluginsPath) {
        $acl = Get-Acl $pluginsPath
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $hasRestrictiveAccess = $acl.Access | Where-Object { $_.IdentityReference -eq $currentUser -and $_.FileSystemRights -eq "FullControl" }
        
        if ($hasRestrictiveAccess) {
            $integrityChecks += @{ "check" = "Plugin directory permissions"; "status" = "PASS" }
        } else {
            $integrityChecks += @{ "check" = "Plugin directory permissions"; "status" = "FAIL" }
        }
    }
    
    # Check config file permissions
    $configFiles = @("config.json", "performance_config.json")
    foreach ($file in $configFiles) {
        $filePath = Join-Path $InstallPath $file
        if (Test-Path $filePath) {
            $acl = Get-Acl $filePath
            $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            $hasRestrictiveAccess = $acl.Access | Where-Object { $_.IdentityReference -eq $currentUser -and $_.FileSystemRights -eq "FullControl" }
            
            if ($hasRestrictiveAccess) {
                $integrityChecks += @{ "check" = "Config file permissions ($file)"; "status" = "PASS" }
            } else {
                $integrityChecks += @{ "check" = "Config file permissions ($file)"; "status" = "FAIL" }
            }
        }
    }
    
    # Check execution policy
    $executionPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($executionPolicy -eq "RemoteSigned" -or $executionPolicy -eq "Restricted") {
        $integrityChecks += @{ "check" = "Execution policy"; "status" = "PASS" }
    } else {
        $integrityChecks += @{ "check" = "Execution policy"; "status" = "WARN" }
    }
    
    # Display results
    foreach ($check in $integrityChecks) {
        $color = if ($check.status -eq "PASS") { "Green" } elseif ($check.status -eq "WARN") { "Yellow" } else { "Red" }
        Write-Host "[$($check.status)] $($check.check)" -ForegroundColor $color
    }
    
    return $integrityChecks
}

function Write-SecurityReport {
    """Generate security hardening report"""
    $reportFile = Join-Path $InstallPath "security_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    
    try {
        # Add integrity check results
        $securityLog.integrity_checks = Test-SecurityIntegrity
        
        # Convert to JSON and save
        $securityLog | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportFile -Encoding UTF8
        
        Write-Host "`nSecurity report saved to: $reportFile" -ForegroundColor Green
        
        # Summary
        $totalActions = $securityLog.actions_taken.Count
        $successActions = ($securityLog.actions_taken | Where-Object { $_.status -eq "SUCCESS" }).Count
        $errorActions = ($securityLog.actions_taken | Where-Object { $_.status -eq "ERROR" }).Count
        $warningActions = ($securityLog.actions_taken | Where-Object { $_.status -eq "WARNING" }).Count
        
        Write-Host "`n=== Security Hardening Summary ===" -ForegroundColor Green
        Write-Host "Total Actions: $totalActions" -ForegroundColor White
        Write-Host "Successful: $successActions" -ForegroundColor Green
        Write-Host "Errors: $errorActions" -ForegroundColor Red
        Write-Host "Warnings: $warningActions" -ForegroundColor Yellow
        
        if ($securityLog.errors.Count -gt 0) {
            Write-Host "`nErrors:" -ForegroundColor Red
            foreach ($error in $securityLog.errors) {
                Write-Host "  - $error" -ForegroundColor Red
            }
        }
        
        if ($securityLog.warnings.Count -gt 0) {
            Write-Host "`nWarnings:" -ForegroundColor Yellow
            foreach ($warning in $securityLog.warnings) {
                Write-Host "  - $warning" -ForegroundColor Yellow
            }
        }
        
    } catch {
        Write-Host "Error generating security report: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main execution
try {
    Write-Host "Starting security hardening process..." -ForegroundColor Green
    
    # Check admin rights
    $isAdmin = Test-AdminRights
    
    # Perform security hardening
    Harden-PluginDirectory
    Harden-ConfigFiles
    Harden-TempDirectories
    Disable-DangerousFeatures
    Set-NetworkSecurity
    
    # Generate report
    Write-SecurityReport
    
    Write-Host "`n=== Security hardening completed ===" -ForegroundColor Green
    
    if ($DryRun) {
        Write-Host "This was a dry run. No changes were made." -ForegroundColor Yellow
        Write-Host "Run without -DryRun to apply security hardening." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "Critical error during security hardening: $($_.Exception.Message)" -ForegroundColor Red
    $securityLog.errors += "Critical error: $($_.Exception.Message)"
    Write-SecurityReport
    exit 1
}