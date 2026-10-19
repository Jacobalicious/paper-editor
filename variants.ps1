# Make a new round of variants and update the gallery page.
# Usage:  .\variants.ps1 samples\example-2plus2.png 4     (4 = number of fresh sketches; 3 versions each)
param([string]$Photo = "samples\example-2plus2.png", [int]$Sketches = 4)
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path "out\$([IO.Path]::GetFileNameWithoutExtension($Photo))-erased.png")) {
    & $py grade.py $Photo
    if ($LASTEXITCODE -ne 0) { return }
    & $py erase.py $Photo
    if ($LASTEXITCODE -ne 0) { return }
}
& $py variants.py $Photo $Sketches
if ($LASTEXITCODE -ne 0) { return }
git add docs
git commit -q -m "New variants round"
git push -q
Start-Process "docs\index.html"
