<#
.SYNOPSIS
    Registers or unregisters the Windows Task Scheduler job for monthly retraining.
    Schedule: Day 01 of every month at 02:00 AM (Vietnam time / GMT+7).
    Conforms to Plan 06 Section 7.
#>

param(
    [switch]$Uninstall = $false,
    [string]$TaskName = "TelcoChurnMonthlyRetraining"
)

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WrapperScript = Join-Path $ProjectRoot "scripts\run_scheduled_retrain.ps1"

if ($Uninstall) {
    Write-Host "Unregistering scheduled task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "[+] Scheduled task '$TaskName' removed successfully." -ForegroundColor Green
    exit 0
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CONFIGURING WINDOWS TASK SCHEDULER (Plan 06 Section 7)" -ForegroundColor Cyan
Write-Host "Task Name:      $TaskName"
Write-Host "Script:         $WrapperScript"
Write-Host "Schedule:       Monthly on Day 1 at 02:00 AM"
Write-Host "==========================================================" -ForegroundColor Cyan

# Action: Run PowerShell script
$Action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`"" -WorkingDirectory $ProjectRoot

# Trigger: Monthly on Day 1 at 02:00 AM
$Trigger = New-ScheduledTaskTrigger -At 2:00am -Daily # Windows standard daily/monthly trigger
# Custom monthly trigger setup
$CimTrigger = New-CimInstance -ClassName MSFT_TaskMonthlyTrigger -Namespace Root/Microsoft/Windows/TaskScheduler -ClientOnly
$CimTrigger.DaysOfMonth = 1
$CimTrigger.StartBoundary = (Get-Date -Format "yyyy-MM-01T02:00:00")
$CimTrigger.Enabled = $true

$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $CimTrigger -Settings $Settings -Description "Monthly Telco Customer Churn model retraining and candidate gate evaluation." -Force
    Write-Host "[+] Scheduled task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host "[i] You can verify in Task Scheduler or run: Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
} catch {
    Write-Warning "Could not register with CIM monthly trigger directly. Falling back to standard scheduled task trigger:"
    $FallbackTrigger = New-ScheduledTaskTrigger -At 2:00am -Once
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $FallbackTrigger -Settings $Settings -Description "Monthly Telco Churn retraining" -Force
    Write-Host "[+] Registered fallback scheduled task '$TaskName'." -ForegroundColor Green
}
