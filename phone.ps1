# Start the phone app. Leave this window open while you use it; close it to stop.
Set-Location $PSScriptRoot
Write-Host ""
Write-Host "On your phone, open one of these:"
Write-Host "  Same Wi-Fi as this PC:   http://10.12.7.253:7870"
Write-Host "  Anywhere (Tailscale):    http://hamster:7870"
Write-Host "Log in as: jacob   Password: the APP_PASSWORD line in .env"
Write-Host ""
& ".\.venv\Scripts\python.exe" app.py
