# Stability gate for the dual-V100 TP2 transport.
#
# Runs the production collectives test (500 host-drained iterations of both transport routes,
# including graph capture and skew cases) several times and checks the Windows System event log for
# driver-level failures that never surface as CUDA return codes: NVLink fatal errors and Xid
# resets. This is the authoritative Phase 0 gate; a one-off smoke probe is not a substitute.
#
# Usage: powershell -ExecutionPolicy Bypass -File tools\tp2\win\run_stability.ps1 [-Rounds 3]
param(
    [int]$Rounds = 3,
    [string]$TestExe = ""
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
if ($TestExe -eq "") { $TestExe = Join-Path $repo "build-win\tests\ninfer_allreduce_test.exe" }
if (-not (Test-Path $TestExe)) {
    throw "missing $TestExe; configure with -DBUILD_TESTING=ON and run: scripts\windows\build.cmd ninfer_allreduce_test"
}

function Get-DriverEvents([datetime]$since) {
    try {
        Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $since } -ErrorAction Stop |
            Where-Object { $_.ProviderName -match 'nvlddmkm|Display' -or $_.Message -match 'NVLink|Xid|GPU has fallen' }
    } catch {
        @()
    }
}

$start = Get-Date
$failed = 0
for ($round = 1; $round -le $Rounds; $round++) {
    Write-Host "round $round/$Rounds"
    & $TestExe
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "round $round exited $LASTEXITCODE"
        $failed++
    }
}

$events = @(Get-DriverEvents $start)
Write-Host "driver events during the run: $($events.Count)"
foreach ($event in $events) {
    Write-Host ("  {0} id={1} {2}: {3}" -f $event.TimeCreated, $event.Id, $event.LevelDisplayName,
        ($event.Message -replace "`r?`n", ' '))
}

if ($failed -eq 0 -and $events.Count -eq 0) {
    Write-Host "RESULT: GO (all rounds passed, no driver events)"
    exit 0
}
Write-Host "RESULT: NO-GO ($failed failed round(s), $($events.Count) driver event(s))"
exit 1
