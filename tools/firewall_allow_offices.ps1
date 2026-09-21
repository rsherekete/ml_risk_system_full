# Inbound firewall rules for the Anti-Fraud web app.
#
# MUST RUN ELEVATED. New-NetFirewallRule needs Administrator; a normal shell
# fails with "Access is denied" and no rule is created.
#
# !! ELEVATION MAY NOT BE ENOUGH ON THIS MACHINE (checked 2026-09-17). All three
# firewall profiles report:
#
#     LocalFirewallRules    N/A (GPO-store only)
#
# meaning local rules are NOT merged with the policy store: a rule created here
# can be created and still never take effect, because Group Policy owns the
# firewall. Inbound default is Block on Domain, Private and Public, and nothing
# in the 306 effective inbound rules mentions 3310 or 3302 -- which is why
# off-machine clients get no response AND leave no trace in the app log.
#
# So run this, then TEST from another machine. If the connection still fails,
# the rule is being ignored and the only fix is IT pushing an equivalent rule
# through Group Policy. Give them the ports, the sources and the profiles below.
#
# This is the change that actually LETS the offices connect. The app-level
# AF_IP_ALLOWLIST is a second lock that can only ever deny -- without a
# firewall rule the port is unreachable from off-machine no matter what the
# allowlist says.
#
# Usage (from an elevated PowerShell):
#     .\tools\firewall_allow_offices.ps1
#     .\tools\firewall_allow_offices.ps1 -Cidr        # widen to /24 per office
#     .\tools\firewall_allow_offices.ps1 -Remove      # undo
#
# -Cidr matters when the office VPN does NOT NAT everyone to one egress
# address. If a teammate in an allowed office still cannot connect, that is
# the first thing to try: their host is in the office subnet but is not the
# single address listed below.

param(
    [switch]$Cidr,
    [switch]$Remove,
    [int[]]$Ports = @(3310, 3302)
)

$ErrorActionPreference = "Stop"

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Not elevated. Re-run this script from an Administrator PowerShell."
}

# Office VPN egress addresses. Keep this list in step with AF_IP_ALLOWLIST in
# start_beta.ps1 -- the firewall opens the port, the allowlist decides who the
# app answers, and a teammate needs to pass BOTH.
# STRONGLY CONSIDER -Cidr HERE. These look like VPN-assigned CLIENT addresses,
# not fixed office gateways: this one host has itself held 10.240.6.103, then
# 10.8.33.21, then 10.240.6.101 over a single day as VPN profiles changed. If
# that is what they are, every teammate in those offices gets a DIFFERENT
# address in the same /24, and these single hosts will admit almost nobody.
$hosts = @(
    "10.240.6.101",     # this host (current, corporate VPN)
    "10.240.6.103",     # this host's previous address on the same VPN
    "10.0.33.21",       # office VPN
    "10.4.33.2",        # office VPN
    "10.8.33.21",       # office VPN (also this host's address on 17 Sep)
    "192.168.0.0/24"    # local LAN (this host is .11; the /24 covers the rest)
)

if ($Cidr) {
    # Widen each single address to its /24. Use when the VPN does not NAT.
    # Entries that already carry a prefix are left alone -- appending a second
    # /24 to "192.168.0.0/24" would produce an invalid address and the rule
    # would be rejected outright.
    $remote = $hosts | ForEach-Object {
        if ($_ -match '/') { $_ } else { ($_ -replace '\.\d+$', '.0') + "/24" }
    }
} else {
    $remote = $hosts
}

$label = @{ 3310 = "beta"; 3302 = "dev" }

foreach ($port in $Ports) {
    $name = "ZFX Anti-Fraud $($label[$port]) ($port)"

    $existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "removing existing rule: $name"
        $existing | Remove-NetFirewallRule
    }
    if ($Remove) { continue }

    New-NetFirewallRule -DisplayName $name `
        -Description "Anti-Fraud Engine $($label[$port]) instance, office VPN sources only." `
        -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort $port -RemoteAddress $remote `
        -Profile Domain, Private | Out-Null

    Write-Host "created: $name"
    Write-Host "  sources: $($remote -join ', ')"
}

if ($Remove) {
    Write-Host "`nRules removed. The ports are now unreachable from off-machine."
    exit 0
}

Write-Host "`n---- verify ----"
foreach ($port in $Ports) {
    $name = "ZFX Anti-Fraud $($label[$port]) ($port)"
    $rule = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($rule) {
        $f = $rule | Get-NetFirewallAddressFilter
        "{0,-34} {1} / {2}  <- {3}" -f $rule.DisplayName, $rule.Enabled, $rule.Action, ($f.RemoteAddress -join ", ")
    }
}

Write-Host "`nNote: Public profile is deliberately excluded. If a teammate is on a"
Write-Host "network Windows has classified as Public, the rule will not apply to them."
