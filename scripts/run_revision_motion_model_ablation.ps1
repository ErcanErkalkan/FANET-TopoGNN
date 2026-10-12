param([int]$Workers = 2)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
python scripts\run_revision_motion_model_ablation.py --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "revision_motion_model_ablation failed with exit code $LASTEXITCODE" }
