$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-pyg.txt

Write-Host "PyTorch Geometric environment ready at .venv"
