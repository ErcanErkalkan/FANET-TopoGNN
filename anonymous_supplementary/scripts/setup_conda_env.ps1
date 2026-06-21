$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

conda env create -f environment.yml
Write-Host "Run: conda activate fanet-topognn"
Write-Host "Note: this is the base scientific environment and does not install PyTorch. Install requirements-deep.txt or use scripts\setup_deep_venv.ps1 for manuscript-facing neural GNN/temporal baselines."
