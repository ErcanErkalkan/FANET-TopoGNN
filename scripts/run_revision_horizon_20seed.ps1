param([int]$Workers = 2)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
python scripts\run_revision_horizon_20seed.py --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "revision_horizon_20seed failed with exit code $LASTEXITCODE" }
