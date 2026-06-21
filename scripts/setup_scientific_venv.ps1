$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "Base scientific environment ready at .venv"
Write-Host "Note: PyTorch is not installed. Use scripts\setup_deep_venv.ps1 for manuscript-facing neural GNN/temporal baselines."
