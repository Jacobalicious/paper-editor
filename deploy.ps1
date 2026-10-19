# Put the latest app on the private Hugging Face Space.
Set-Location $PSScriptRoot
& ".\.venv\Scripts\python.exe" deploy_space.py
