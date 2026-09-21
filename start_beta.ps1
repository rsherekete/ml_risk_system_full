# Anti-Fraud Engine -- BETA instance (internal teams).
#
# Serves the SAME codebase as the full app with AF_BETA=1, which exposes three
# sections only: Latency Arbitrage, Toxic Flow, Account Detail (with per-trade
# replay on the Order tab). Everything else stays on the development instance.
#
# The beta is a READER. It starts no background work: no Kafka materialiser, no
# scans, no warehouse top-up, no copy trader. It renders the artefacts the full
# instance writes. This is deliberate -- the tick store is DuckDB and allows one
# writer, so a second scanner would invalidate the tape for both instances.
#
# Usage:  .\start_beta.ps1            # restarts the beta on port 3310
#         .\start_beta.ps1 -Port 3311

param(
    [int]$Port = 3310,
    [string]$BindHost = "0.0.0.0"
)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $dir ".launch-venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "launch venv not found at $py" }

# Stop whatever currently holds the port, so a re-run is a clean restart.
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $existing) {
    Write-Host "stopping pid $($c.OwningProcess) on port $Port"
    taskkill /PID $c.OwningProcess /T /F 2>&1 | Out-Null
    Start-Sleep -Seconds 2
}

# Rotate the previous run's logs rather than appending to them.
$stamp = Get-Date -Format "HHmm"
# Windows can still hold the handle for a moment after taskkill returns, so a
# straight Move-Item throws and -- because $ErrorActionPreference is Stop --
# aborts the whole script AFTER the old instance has been killed, leaving the
# beta down. Retry briefly, then give up and keep appending rather than fail:
# a tidy log is never worth not starting.
foreach ($suffix in @("log", "log.err")) {
    $path = Join-Path $dir "beta_uvicorn.$suffix"
    if (-not (Test-Path $path)) { continue }
    $moved = $false
    foreach ($attempt in 1..5) {
        try {
            Move-Item $path (Join-Path $dir "beta_uvicorn.$stamp.prev.$suffix") -Force -ErrorAction Stop
            $moved = $true; break
        } catch { Start-Sleep -Milliseconds 600 }
    }
    if (-not $moved) { Write-Host "note: could not rotate $suffix (file still held); appending" }
}

# Set for the child, then RESTORED below. Leaving AF_BETA=1 in the session is
# not harmless: anything else launched from the same shell afterwards inherits
# it, and the full instance silently comes up in beta mode -- /admin and every
# other non-beta route redirect away, with no error to explain it. That is
# exactly what happened on 2026-09-18.
$priorBeta = $env:AF_BETA
$env:AF_BETA = "1"

# IP allowlist: the office VPN egress addresses plus this host. Only these may
# reach the beta; loopback is always allowed so the box itself keeps working.
# Empty or unset = no IP restriction, so clearing this cannot lock anyone out.
# NOTE: this only DENIES. What lets a remote office reach the port at all is
# the firewall rule in tools\firewall_allow_offices.ps1, which needs admin.
# 192.168.0.0/24 is the local LAN: the two host addresses in the notes
# (192.168.0.6 / .11) are THIS machine's own, so listing only those would still
# block every other machine on that LAN. The /24 is what keeps LAN access working.
# SUBNETS, NOT HOSTS -- and the evidence for it is in the log. The four original
# addresses were single hosts, and they turned out to be VPN-assigned CLIENT IPs
# drawn from a pool: this machine alone held 10.240.6.103, then 10.8.33.21, then
# 10.240.6.101 in one day, and a teammate on 10.240.6.100 was refused with
# "[ip-allowlist] blocked 10.240.6.100" purely because the pool handed them a
# different number. Per-device /32s cannot work against a DHCP-style pool.
#
# This host's own address must stay covered too: reaching the app by hostname or
# by IP from this machine arrives from the interface address, not 127.0.0.1, so
# it is checked like any other client.
#
# Exposure this accepts: anyone already on one of these corporate VPN segments
# can reach the LOGIN PAGE. It is a network gate, not the access control --
# authentication is. Keep per-user accounts and a real admin password.
# NOT SET HERE ANY MORE. The allowlist is managed in the admin console, which
# stores it in app.db and applies changes without a restart. Hardcoding it here
# fought that: the console could save "cleared" and this line would put the
# ranges straight back on the next restart.
#
# To make a deployment come up restricted BEFORE anyone has logged in to
# configure it, set the variable in the environment before running this, e.g.
#   $env:AF_IP_ALLOWLIST = "10.240.6.0/24,192.168.0.0/24"; .\start_beta.ps1
# Anything saved in the console then takes precedence from that point on.
if ($env:AF_IP_ALLOWLIST) {
    Write-Host "ip allowlist (bootstrap): $env:AF_IP_ALLOWLIST (+ loopback)"
} else {
    Write-Host "ip allowlist: not set here - governed by the admin console"
}

$proc = Start-Process -FilePath $py `
    -ArgumentList '-m', 'uvicorn', 'webapp.main:app', '--host', $BindHost, '--port', "$Port" `
    -WorkingDirectory $dir `
    -RedirectStandardOutput (Join-Path $dir "beta_uvicorn.log") `
    -RedirectStandardError (Join-Path $dir "beta_uvicorn.log.err") `
    -WindowStyle Hidden -PassThru

# The child has inherited it; put the session back as it was.
if ($null -eq $priorBeta) { Remove-Item Env:\AF_BETA -ErrorAction SilentlyContinue }
else { $env:AF_BETA = $priorBeta }

Write-Host "beta starting (pid $($proc.Id)) on ${BindHost}:${Port}"

$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    $listen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listen) {
        Write-Host "beta is listening on $($listen[0].LocalAddress):$Port (pid $($listen[0].OwningProcess))"
        # PRINT EVERY ADDRESS, NOT JUST THE FIRST. The bind is 0.0.0.0, so the
        # app answers on all of this host's interfaces -- but this used to print
        # only the first, and the sort put DHCP last, so it always advertised
        # the Ethernet/VPN address (10.240.6.x) and never the Wi-Fi one
        # (10.240.5.x). Those are DIFFERENT /24s with different gateways, so a
        # teammate whose machine sits on the other subnet got a URL that could
        # not route to them, and the page simply timed out. Each teammate wants
        # the address on THEIR OWN subnet; print them all and let them pick.
        $ips = Get-NetIPAddress -AddressFamily IPv4 |
               Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                              $_.PrefixOrigin -ne 'WellKnown' }
        Write-Host ""
        Write-Host "team URLs -- give each person the one matching THEIR subnet:"
        foreach ($a in $ips) {
            $net = ($a.IPAddress -replace '\.\d+$', '.x')
            Write-Host ("  http://{0}:{1}    (for machines on {2}, via {3})" -f `
                        $a.IPAddress, $Port, $net, $a.InterfaceAlias)
        }
        Write-Host ""
        Write-Host "If a teammate's browser hangs then times out, they are on neither"
        Write-Host "subnet -- that is a VPN routing request for IT, not an app problem."
        exit 0
    }
    Start-Sleep -Seconds 2
}

Write-Warning "beta did not bind port $Port within 90s -- check beta_uvicorn.log.err"
exit 1
