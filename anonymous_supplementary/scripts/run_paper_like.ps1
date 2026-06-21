param(
    [switch]$Resume,
    [switch]$Compact,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

$env:PYTHONDONTWRITEBYTECODE = "1"

if ($ConfigPath) {
    $configToRun = $ConfigPath
} elseif ($Compact) {
    $configToRun = "configs/publication_compact.json"
} else {
    $configToRun = "configs/publication_compact.json"
}

$args = @("-u", "main.py", "--config", $configToRun)
if ($Resume) {
    $args += "--resume"
}

if (Test-Path .venv312) {
    $pythonExe = ".\.venv312\Scripts\python"
} elseif (Test-Path .venv) {
    $pythonExe = ".\.venv\Scripts\python"
} else {
    $pythonExe = "python"
}

& $pythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyTorch is not installed in the selected environment. Neural model names in publication_compact or paper_like_submission will use base-environment surrogate estimators. Run scripts\setup_deep_venv.ps1 or install requirements-deep.txt before reporting neural rows as PyTorch GNN/temporal results."
}

& $pythonExe @args
