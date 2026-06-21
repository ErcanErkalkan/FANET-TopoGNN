param(
    [string]$OutputDir = "data\external_validation\raw"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $target | Out-Null

$files = @(
    @{
        Name = "uav6_swarm_formation_2024-12-04-15-29-12_0_ok1.bag"
        Url = "https://zenodo.org/api/records/14701641/files/uav6_swarm_formation_2024-12-04-15-29-12_0_ok1.bag/content"
        Md5 = "d9d88fbb378e51bdf6ac57a4ca2be08f"
    },
    @{
        Name = "uav8_swarm_formation_2024-12-04-15-29-08_0_ok1.bag"
        Url = "https://zenodo.org/api/records/14701641/files/uav8_swarm_formation_2024-12-04-15-29-08_0_ok1.bag/content"
        Md5 = "ef86998f372689d95a4be696b3d92b88"
    },
    @{
        Name = "uav9_swarm_formation_2024-12-04-15-29-04_0_ok1.bag"
        Url = "https://zenodo.org/api/records/14701641/files/uav9_swarm_formation_2024-12-04-15-29-04_0_ok1.bag/content"
        Md5 = "adb55c9c6282b7bbd609cda5361b00f6"
    }
)

foreach ($file in $files) {
    $destination = Join-Path $target $file.Name
    Write-Host "[download] $($file.Name)"
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $destination $file.Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $($file.Name) with exit code $LASTEXITCODE"
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actual -ne $file.Md5) {
        throw "MD5 mismatch for $($file.Name): expected $($file.Md5), got $actual"
    }
    Write-Host "[verified] $($file.Name) md5=$actual"
}
