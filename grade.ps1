# Grade a photo. Usage:  .\grade.ps1 samples\example-2plus2.png
param([string]$Photo = "samples\example-2plus2.png")
Set-Location $PSScriptRoot
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Made a .env file. Open it, paste your Claude API key, save, then run this again."
    notepad ".env"
    return
}
& ".\.venv\Scripts\python.exe" grade.py $Photo
