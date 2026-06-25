param(
    [string]$OutputDir = "data\external_validation\raw"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$targetRoot = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$datasets = @(
    @{
        Name = "aerpaw_dataset22"
        Url = "https://drive.google.com/drive/folders/1S7jqTYkPfZD9gf_5tFq1ufyuqEnXgQPn"
        Required = "Logs\lte.csv"
    },
    @{
        Name = "aerpaw_dataset23"
        Url = "https://drive.google.com/drive/folders/1yo3B6xCxHx46ceTwO_NL6zhvplD0l-RV"
        Required = "Logs\4G_lte.csv", "Logs\5G_nr.csv", "Logs\iperf_throughput.csv"
    }
)

foreach ($dataset in $datasets) {
    $destination = Join-Path $targetRoot $dataset.Name
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Write-Host "[download] $($dataset.Name)"
    & python -m gdown --folder --continue $dataset.Url -O $destination
    if ($LASTEXITCODE -ne 0) {
        throw "gdown failed for $($dataset.Name) with exit code $LASTEXITCODE"
    }
    foreach ($relative in $dataset.Required) {
        $requiredPath = Join-Path $destination $relative
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Missing expected AERPAW file: $requiredPath"
        }
        Write-Host "[verified] $requiredPath"
    }
}
