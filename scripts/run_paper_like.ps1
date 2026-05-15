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
    $configToRun = "configs/q1_publication_compact.json"
} else {
    $configToRun = "configs/q1_publication_compact.json"
}

$args = @("-u", "main.py", "--config", $configToRun)
if ($Resume) {
    $args += "--resume"
}

if (Test-Path .venv312) {
    .\.venv312\Scripts\python @args
} elseif (Test-Path .venv) {
    .\.venv\Scripts\python @args
} else {
    python @args
}
