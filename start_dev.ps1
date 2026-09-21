# Anti-Fraud Engine -- DEVELOPMENT instance (the full app).
#
# The counterpart to start_beta.ps1, which existed while dev had to be
# restarted by hand from a remembered command line. Getting that command wrong
# is not harmless: Get-CimInstance reports the running dev as the BASE
# interpreter (AppData\...\Python312\python.exe), and copying that verbatim
# dies instantly with "No module named uvicorn" -- leaving dev DOWN. The venv
# below is the only complete environment, so this script hardcodes it.
#
# RESTARTING DEV RESUMES THE VANTAGE COPIER. vantage.yaml carries mode: live
# and the kill switch is NOT persisted, so it comes back disengaged whatever it
# was before. Check /vantage after this returns.
#
# Usage:  .\start_dev.ps1              # restart dev on port 3302
#         .\start_dev.ps1 -Port 3303

param(
    [int]$Port = 3302,
    [string]$BindHost = "0.0.0.0"
)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $dir ".launch-venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "launch venv not found at $py" }

# DO NOT restart mid-write. The warehouse top-up runs every 15 minutes and
# rewrites whole month partitions; killing it part-way is exactly how
# mt4_live02/2026-09.parquet ended up truncated on 17 Sep. Atomic writes make
# this safe from the app's next start onward, but the CURRENT process may still
# be the old code, so check before pulling the rug.
$tmp = Get-ChildItem -Path (Join-Path $dir "webapp\warehouse") -Recurse -Filter "*.tmp" -ErrorAction SilentlyContinue
if ($tmp) {
    Write-Host "WARNING: a partition write is in flight:"
    $tmp | ForEach-Object { Write-Host "  $($_.FullName)" }
    Write-Host "Wait for it to finish, then re-run. Nothing has been stopped."
    exit 1
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $existing) {
    Write-Host "stopping pid $($c.OwningProcess) on port $Port"
    taskkill /PID $c.OwningProcess /T /F 2>&1 | Out-Null
    Start-Sleep -Seconds 2
}

# Rotate logs rather than appending. Retry briefly: Windows can hold the handle
# for a moment after taskkill returns, and a tidy log is never worth failing to
# start the app for -- so fall back to appending instead of throwing.
$stamp = Get-Date -Format "HHmm"
foreach ($suffix in @("log", "log.err")) {
    $path = Join-Path $dir "scratch_uvicorn_relaunch.$suffix"
    if (-not (Test-Path $path)) { continue }
    $moved = $false
    foreach ($attempt in 1..5) {
        try {
            Move-Item $path (Join-Path $dir "scratch_uvicorn_relaunch.$stamp.prev.$suffix") -Force -ErrorAction Stop
            $moved = $true; break
        } catch { Start-Sleep -Milliseconds 600 }
    }
    if (-not $moved) { Write-Host "note: could not rotate $suffix (file still held); appending" }
}

# AF_BETA must NOT leak in from the calling shell. If it does, the full app
# comes up in beta mode and every non-beta route silently redirects away, with
# no error to explain it -- which is exactly what happened on 2026-09-18.
if ($env:AF_BETA) {
    Write-Host "note: AF_BETA was set in this shell ($env:AF_BETA); clearing it for the child"
    Remove-Item Env:\AF_BETA -ErrorAction SilentlyContinue
}

$proc = Start-Process -FilePath $py `
    -ArgumentList '-m', 'uvicorn', 'webapp.main:app', '--host', $BindHost, '--port', "$Port" `
    -WorkingDirectory $dir `
    -RedirectStandardOutput (Join-Path $dir "scratch_uvicorn_relaunch.log") `
    -RedirectStandardError (Join-Path $dir "scratch_uvicorn_relaunch.log.err") `
    -WindowStyle Hidden -PassThru

Write-Host "dev starting (pid $($proc.Id)) on ${BindHost}:${Port}"

$deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    $listen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listen) {
        Write-Host "dev is listening on $($listen[0].LocalAddress):$Port (pid $($listen[0].OwningProcess))"
        # Every address, not just the first: the bind is 0.0.0.0 and this host
        # sits on two different /24s, so a teammate needs the one on THEIR
        # subnet. Printing only the first sent everyone the Ethernet address.
        $ips = Get-NetIPAddress -AddressFamily IPv4 |
               Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                              $_.PrefixOrigin -ne 'WellKnown' }
        Write-Host ""
        Write-Host "URLs -- give each person the one matching THEIR subnet:"
        foreach ($a in $ips) {
            $net = ($a.IPAddress -replace '\.\d+$', '.x')
            Write-Host ("  http://{0}:{1}    (for machines on {2}, via {3})" -f `
                        $a.IPAddress, $Port, $net, $a.InterfaceAlias)
        }
        Write-Host ""
        Write-Host "A bound port is not a ready app: dev warms caches for about a"
        Write-Host "further minute before it answers. THE COPIER IS NOW RUNNING --"
        Write-Host "check /vantage if that is not what you want."
        exit 0
    }
    Start-Sleep -Seconds 2
}
Write-Host "dev did not bind within 180s -- check scratch_uvicorn_relaunch.log.err"
exit 1
