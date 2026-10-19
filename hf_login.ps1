# Log this PC in to Hugging Face using the token currently on your clipboard.
# Copy the token first, then run this. The token is never shown or saved in the project.
Set-Location $PSScriptRoot
$token = (Get-Clipboard | Out-String).Trim()
if (-not $token.StartsWith("hf_")) {
    Write-Host "The clipboard doesn't hold a Hugging Face token (they start with hf_). Copy it and try again."
    return
}
& ".\.venv\Scripts\hf.exe" auth login --token $token
$ok = $LASTEXITCODE -eq 0
Set-Clipboard -Value " "
if ($ok) { Write-Host "Logged in. Clipboard cleared." }
else { Write-Host "Login FAILED - check the token is a Write token and try again." }
