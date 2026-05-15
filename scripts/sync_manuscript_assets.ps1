param(
    [string]$Profile = "q1_publication_compact",
    [string]$ManuscriptDir = "..\\FANET_TopoGNN"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

$sourceDir = Join-Path (Get-Location) "outputs\$Profile\figures"
$targetDir = Join-Path (Resolve-Path $ManuscriptDir) "figures\generated"
$sourceRoot = Join-Path (Get-Location) "outputs\$Profile"
$targetTableDir = Join-Path (Resolve-Path $ManuscriptDir) "tables\generated"

if (-not (Test-Path $sourceDir)) {
    throw "Source figure directory not found: $sourceDir"
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $targetTableDir | Out-Null

Get-ChildItem -LiteralPath $sourceDir -Filter "*.png" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $targetDir $_.Name) -Force
}

foreach ($name in @("manuscript_tables.tex", "claims_summary.md", "manuscript_summary.json")) {
    $source = Join-Path $sourceRoot $name
    if (Test-Path $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $targetTableDir $name) -Force
    }
}

Write-Output "Synced manuscript assets from $sourceRoot to $ManuscriptDir"
