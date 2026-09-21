# Friendly hostnames for the Anti-Fraud web apps.
#
#     http://beta.zfxrisk.com:3310    -> beta  (latency, toxic, account detail)
#     http://dev.zfxrisk.com:3302     -> main  (full app)
#
# MUST RUN ELEVATED: the hosts file is not writable by a normal user.
#
# SCOPE, and the reason this is only a stopgap. The hosts file is PER MACHINE.
# Running this here makes the names work on THIS box only; every teammate who
# wants them would have to run it on theirs. The real fix is two DNS A records
# (see -Dns below), after which nobody touches a hosts file at all.
#
# The port stays in the URL on purpose. Dropping it would need a reverse proxy
# on port 80, which would also make every request arrive from the proxy rather
# than the client -- and the app's AF_IP_ALLOWLIST checks the client address,
# so the allowlist would see 127.0.0.1 for everyone and stop discriminating.
# Keeping the port keeps that control working.
#
# Usage (elevated):
#     .\tools\hostnames_local.ps1            # add / refresh entries
#     .\tools\hostnames_local.ps1 -Remove    # take them out
#     .\tools\hostnames_local.ps1 -Dns       # just print the records for IT
#     .\tools\hostnames_local.ps1 -Ip 10.8.33.21   # pin a specific address

param(
    [switch]$Remove,
    [switch]$Dns,
    [string]$Ip
)

$ErrorActionPreference = "Stop"

$NAMES = @("beta.zfxrisk.com", "dev.zfxrisk.com")
$TAG   = "# zfx-antifraud"

# Detect this host's routable IPv4. Auto-detected rather than hardcoded because
# it has already moved once (10.240.6.103 -> 10.8.33.21 on 17 Sep 2026) and a
# stale hosts entry fails in the most confusing way possible: the name resolves,
# and then nothing answers.
if (-not $Ip) {
    $Ip = (Get-NetIPAddress -AddressFamily IPv4 |
           Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and
                          $_.PrefixOrigin -ne 'WellKnown' } |
           Sort-Object { $_.PrefixOrigin -eq 'Dhcp' } |
           Select-Object -First 1).IPAddress
}
if (-not $Ip) { throw "Could not determine this host's IPv4 address. Pass -Ip explicitly." }

if ($Dns) {
    Write-Host "Ask IT for these records (zone: zfxrisk.com):`n"
    foreach ($n in $NAMES) { "  {0,-20} A     {1}" -f $n.Split('.')[0], $Ip }
    Write-Host "`nOnce these exist, nobody needs the hosts file and the URLs are:"
    Write-Host "  http://beta.zfxrisk.com:3310"
    Write-Host "  http://dev.zfxrisk.com:3302"
    Write-Host "`nThe host's address is static-assigned; if it changes, the records must follow."
    exit 0
}

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Not elevated. Re-run from an Administrator PowerShell, or use -Dns to just print the records."
}

$hostsFile = "$env:SystemRoot\System32\drivers\etc\hosts"
$backup    = "$hostsFile.zfx-backup"

if (-not (Test-Path $backup)) {
    Copy-Item $hostsFile $backup
    Write-Host "backed up hosts -> $backup"
}

# Drop any previous entries of ours, then re-add. Rewriting rather than
# appending keeps a changed IP from leaving two lines for the same name, where
# the stale one wins.
$lines = Get-Content $hostsFile | Where-Object { $_ -notmatch [regex]::Escape($TAG) }

if ($Remove) {
    Set-Content -Path $hostsFile -Value $lines -Encoding ASCII
    Write-Host "removed entries for: $($NAMES -join ', ')"
    exit 0
}

foreach ($n in $NAMES) { $lines += ("{0}`t{1}`t{2}" -f $Ip, $n, $TAG) }
Set-Content -Path $hostsFile -Value $lines -Encoding ASCII

Write-Host "mapped -> $Ip"
foreach ($n in $NAMES) { "  $n" }

Write-Host "`n---- verify ----"
foreach ($n in $NAMES) {
    $port = if ($n -like "beta.*") { 3310 } else { 3302 }
    try {
        $r = Invoke-WebRequest "http://${n}:${port}/login" -UseBasicParsing -TimeoutSec 20
        "  http://{0}:{1}  -> {2}" -f $n, $port, $r.StatusCode
    } catch {
        "  http://{0}:{1}  -> {2}" -f $n, $port, $_.Exception.Message
    }
}

Write-Host "`nThis machine only. For the team, run with -Dns and give IT the records."
