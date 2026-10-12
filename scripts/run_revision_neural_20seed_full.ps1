param(
    [int]$Workers = 1,
    [switch]$PrepareOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ArgsList = @("scripts\run_revision_neural_20seed_full.py", "--seed-workers", "$Workers")
if ($PrepareOnly) { $ArgsList += "--prepare-only" }
if ($Resume) { $ArgsList += "--resume" }

python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "revision_neural_20seed_full failed with exit code $LASTEXITCODE"
}
