$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

conda env create -f environment.yml
Write-Host "Run: conda activate fanet-topognn-q1"
