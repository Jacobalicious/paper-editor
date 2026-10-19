# Run the whole pipeline on a photo. Usage:  .\run.ps1 samples\example-2plus2.png
param([string]$Photo = "samples\example-2plus2.png")
Set-Location $PSScriptRoot
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Made a .env file. Open it, paste your Claude API key, save, then run this again."
    notepad ".env"
    return
}
$py = ".\.venv\Scripts\python.exe"
& $py grade.py $Photo
if ($LASTEXITCODE -ne 0) { return }
& $py erase.py $Photo
if ($LASTEXITCODE -ne 0) { return }
& $py write.py $Photo --redraw
