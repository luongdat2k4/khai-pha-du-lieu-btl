<#
.SYNOPSIS
    Script wrapper to run scheduled monthly retraining with date-based logging.
    Conforms to Plan 06 Section 7.
#>

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

# Log directory
$LogDir = Join-Path $ProjectRoot "reports\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "retrain_$Timestamp.log"

Write-Output "=========================================================="
Write-Output "TELCO CHURN AUTOMATED RETRAINING TASK"
Write-Output "Execution Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "Project Root:   $ProjectRoot"
Write-Output "Log File:       $LogFile"
Write-Output "=========================================================="

# Resolve Python executable (virtualenv or current environment)
$PythonPath = (Get-Command python).Source
if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) {
    $PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}

Write-Output "Using Python: $PythonPath"

# Run retrain script enforcing schedule gate
& $PythonPath (Join-Path $ProjectRoot "scripts\retrain.py") --check-schedule 2>&1 | Tee-Object -FilePath $LogFile

Write-Output "Retraining finished with ExitCode: $LASTEXITCODE"
exit $LASTEXITCODE
