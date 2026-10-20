# Put the latest committed code on the tower and restart the phone app there.
# Usage:  .\tower.ps1      (commit first: only committed changes go over)
Set-Location $PSScriptRoot

# The tower keeps its own copy of this repo; a push updates its files directly.
ssh tower "test -d paper-editor/.git || (git init -q -b main paper-editor && git -C paper-editor config receive.denyCurrentBranch updateInstead)"
git push -q tower:paper-editor main
if ($LASTEXITCODE -ne 0) { Write-Host "Push to the tower failed."; return }

# The Claude key lives in .env, which git never sends, so copy it once.
ssh tower "test -f paper-editor/.env"
if ($LASTEXITCODE -ne 0) { scp -q ".env" "tower:paper-editor/.env" }

ssh tower "sh paper-editor/tower/setup.sh"
if ($LASTEXITCODE -ne 0) { Write-Host "Setup on the tower failed."; return }
Write-Host ""
Write-Host "Running. On your phone (Tailscale on):  http://cat:7870"
