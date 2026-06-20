$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host "Scientific environment ready at .venv"
