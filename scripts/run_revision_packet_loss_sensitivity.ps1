param(
    [int]$Workers = 2,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$ArgsList = @("scripts\run_revision_packet_loss_sensitivity.py", "--workers", "$Workers")
if ($Resume) { $ArgsList += "--resume" }
python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "revision_packet_loss_sensitivity failed with exit code $LASTEXITCODE" }
