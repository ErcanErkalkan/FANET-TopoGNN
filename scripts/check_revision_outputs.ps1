param([string]$ProjectRoot = (Get-Location).Path)
$ErrorActionPreference = 'Stop'

$checks = @(
    @{ Code='R1'; Path='outputs/revision_motion_model_ablation'; Required=@('protocol.json','per_seed_metrics.csv','summary.csv') },
    @{ Code='R2'; Path='outputs/revision_horizon_20seed'; Required=@('protocol.json','per_seed_metrics.csv','summary.csv') },
    @{ Code='R3'; Path='outputs/revision_event_protocol_20seed'; Required=@('protocol.json','per_seed_metrics.csv','summary.csv') },
    @{ Code='R4'; Path='outputs/revision_neural_20seed_full'; Required=@('protocol.json','per_seed_metrics.csv','summary.csv') },
    @{ Code='R5'; Path='outputs/revision_packet_loss_sensitivity'; Required=@('protocol.json','per_seed_metrics.csv','summary.csv') }
)

Write-Host '=== Revision Output Audit ===' -ForegroundColor Cyan
foreach ($c in $checks) {
    $dir = Join-Path $ProjectRoot $c.Path
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
        Write-Host "[NOT RUN] $($c.Code) $($c.Path)" -ForegroundColor DarkGray
        continue
    }

    $missing = @()
    foreach ($f in $c.Required) {
        if (-not (Test-Path -LiteralPath (Join-Path $dir $f) -PathType Leaf)) { $missing += $f }
    }

    if ($missing.Count -eq 0) {
        Write-Host "[FILES PRESENT] $($c.Code) $($c.Path)" -ForegroundColor Green
    } else {
        Write-Host "[INCOMPLETE] $($c.Code) missing: $($missing -join ', ')" -ForegroundColor Yellow
    }
}

Write-Host 'Note: file presence is not scientific completion. Use experiment-specific completion rules before citing results.' -ForegroundColor Yellow
