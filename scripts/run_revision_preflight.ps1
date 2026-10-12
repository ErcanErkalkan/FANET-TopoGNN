param(
    [switch]$CheckGpu
)

$ErrorActionPreference = 'Stop'

function Pass($msg) { Write-Host "[PASS] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

Write-Host "=== FANET-TopoGNN Revision Preflight ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"

$required = @(
    'configs\revision_neural_20seed_full.json',
    'configs\revision_horizon_20seed.json',
    'configs\revision_event_protocol_20seed.json',
    'configs\revision_motion_model_ablation.json',
    'configs\revision_packet_loss_sensitivity.json',
    'scripts\run_revision_neural_20seed_full.py',
    'scripts\run_revision_horizon_20seed.py',
    'scripts\run_revision_event_protocol_20seed.py',
    'scripts\run_revision_motion_model_ablation.py',
    'scripts\run_revision_packet_loss_sensitivity.py',
    'scripts\audit_revision_claims.py',
    'scripts\audit_rendered_references.py',
    'paper\revision\revision_manuscript_skeleton.tex',
    'paper\revision\revision_method_core.tex'
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) { Fail "Missing required file: $path" }
}
Pass "All required revision files are present."

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { Fail "python command not found in PATH." }
$pyver = python --version 2>&1
Pass "Python available: $pyver"

python -c "import numpy, pandas, sklearn; print('numpy/pandas/sklearn imports OK')"
if ($LASTEXITCODE -ne 0) { Fail "Core Python dependencies failed." }
Pass "Core scientific Python dependencies import successfully."

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Fail "PyTorch is required for the 20-seed neural extension." }
Pass "PyTorch imports successfully."

if ($CheckGpu) {
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
}

# Syntax checks only; do not execute experiments here.
$pythonScripts = @(
    'scripts\run_revision_neural_20seed_full.py',
    'scripts\run_revision_horizon_20seed.py',
    'scripts\run_revision_event_protocol_20seed.py',
    'scripts\run_revision_motion_model_ablation.py',
    'scripts\run_revision_packet_loss_sensitivity.py',
    'scripts\audit_revision_claims.py',
    'scripts\audit_rendered_references.py'
)
foreach ($script in $pythonScripts) {
    python -m py_compile $script
    if ($LASTEXITCODE -ne 0) { Fail "Python syntax check failed: $script" }
}
Pass "All revision Python scripts pass syntax compilation."

# Safe prepare-only check for the largest neural job.
python .\scripts\run_revision_neural_20seed_full.py --prepare-only
if ($LASTEXITCODE -ne 0) { Fail "Neural 20-seed prepare-only check failed." }
Pass "Neural 20-seed protocol prepare-only check passed."

Write-Host ""
Write-Host "PRE-FLIGHT COMPLETE." -ForegroundColor Cyan
Write-Host "Recommended first real experiment:" -ForegroundColor Cyan
Write-Host ".\scripts\run_revision_motion_model_ablation.ps1 -Workers 2"
