$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    Write-Error ".venv not found. Run scripts/setup_scientific_venv.ps1 first."
}

$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python main.py --config configs/quickstart.json
