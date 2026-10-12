param([string]$RunDirectory = '', [string]$JobId = '')
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$Jobs = @{
    R2 = @('scripts/run_revision_horizon_20seed.py', '--workers', '1')
    R3 = @('scripts/run_revision_event_protocol_20seed.py', '--workers', '1')
    R4 = @('scripts/run_revision_neural_20seed_full.py', '--seed-workers', '1', '--resume')
    R5 = @('scripts/run_revision_packet_loss_sensitivity.py', '--workers', '1', '--resume')
}
if ($JobId) {
    if (-not $Jobs.ContainsKey($JobId)) { throw "Unknown job: $JobId" }
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:PYTHONUNBUFFERED = '1'
    $statusPath = Join-Path $RunDirectory "$JobId.status.json"
    $status = [ordered]@{ job = $JobId; pid = $PID; status = 'RUNNING'; started_at = [DateTimeOffset]::Now.ToString('o'); command = $Jobs[$JobId] }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    try {
        $pythonArgs = @('-u') + $Jobs[$JobId]
        & python @pythonArgs
        $code = $LASTEXITCODE
        if ($code -ne 0) { throw "Python exited with code $code" }
        $status.status = 'COMPLETE'
        $status['exit_code'] = 0
    } catch {
        $status.status = 'FAILED'
        $status['error'] = $_.Exception.Message
        $status['exit_code'] = 1
        Write-Output $_
    } finally {
        $status['finished_at'] = [DateTimeOffset]::Now.ToString('o')
        $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    }
    exit $status.exit_code
}
if (-not $RunDirectory) {
    $RunDirectory = Join-Path $RepoRoot ('00_PROJECT_CONTROL/06_RUN_LOGS/parallel_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
New-Item -ItemType Directory -Path $RunDirectory -Force | Out-Null
$manifest = @()
foreach ($id in @('R2', 'R3', 'R4', 'R5')) {
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -RunDirectory "{1}" -JobId {2}' -f $PSCommandPath, $RunDirectory, $id
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunDirectory "$id.stdout.log") -RedirectStandardError (Join-Path $RunDirectory "$id.stderr.log") -PassThru
    $manifest += [ordered]@{ job = $id; supervisor_pid = $process.Id; workers = 1; numerical_threads = 1; command = $Jobs[$id] }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RunDirectory 'launch_manifest.json') -Encoding UTF8
Write-Output "Run directory: $RunDirectory"
$manifest | ForEach-Object { Write-Output ("{0}: supervisor PID {1}" -f $_.job, $_.supervisor_pid) }
