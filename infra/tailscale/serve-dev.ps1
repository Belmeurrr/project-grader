# Expose the local dev stack over Tailscale HTTPS so a phone signed into
# the same tailnet can hit the capture wizard with camera permission
# (Safari/Chrome refuse getUserMedia on plain HTTP from a non-localhost
# origin). The web is mounted at https://<tailnet-host>/ and the API at
# https://<tailnet-host>:8443.
#
# Pre-reqs:
#   - Tailscale running and signed in on this machine
#   - HTTPS certificates enabled in the tailnet admin (Settings → DNS)
#   - Web (`npm run dev`) on :3000 and API (uvicorn) on :8000 already up
#
# Usage:
#   pwsh infra/tailscale/serve-dev.ps1            # bring up
#   pwsh infra/tailscale/serve-dev.ps1 -Stop      # tear down

param([switch]$Stop)

$ErrorActionPreference = "Stop"

if ($Stop) {
    tailscale serve reset
    Write-Host "tailscale serve config cleared."
    return
}

$status = tailscale status --self --json | ConvertFrom-Json
$dns = $status.Self.DNSName.TrimEnd('.')
if (-not $dns) {
    throw "Could not derive tailnet hostname from 'tailscale status'."
}

tailscale serve --bg --https=443 http://127.0.0.1:3000
tailscale serve --bg --https=8443 http://127.0.0.1:8000

Write-Host ""
Write-Host "Tailnet host: $dns"
Write-Host "Web:  https://$dns/grade"
Write-Host "API:  https://$dns`:8443"
Write-Host ""
Write-Host "On your phone (signed into the same tailnet), open the Web URL"
Write-Host "and allow camera on the first shot. -Stop tears the config down."
