param(
    [string]$Profile = "publication_compact",
    [string]$ManuscriptDir = "paper"
)

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$outputsRoot = Join-Path (Get-Location) "outputs"
$sourceRoot = Join-Path $outputsRoot $Profile
$sourceDir = Join-Path $sourceRoot "figures"
$manuscriptRoot = Resolve-Path $ManuscriptDir
$targetFigureRoot = Join-Path $manuscriptRoot "figures"
$targetDir = Join-Path $targetFigureRoot "generated"
$targetTableRoot = Join-Path $manuscriptRoot "tables"
$targetTableDir = Join-Path $targetTableRoot "generated"

if (-not (Test-Path $sourceDir)) {
    throw "Source figure directory not found: $sourceDir"
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $targetTableDir | Out-Null

foreach ($pattern in @("*.png", "*.pdf")) {
    Get-ChildItem -LiteralPath $sourceDir -Filter $pattern | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $targetDir $_.Name) -Force
    }
}

foreach ($name in @("manuscript_tables.tex", "claims_summary.md", "manuscript_summary.json")) {
    $source = Join-Path $sourceRoot $name
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $targetTableDir $name) -Force
    }
}

Write-Output "Synced manuscript assets from $sourceRoot to $ManuscriptDir"
