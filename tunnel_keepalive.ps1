# Self-healing public tunnel for the ZFX webapp (port 8000).
# Uses the built-in OpenSSH client + localhost.run (no install, no admin).
# Auto-reconnects if the tunnel drops; writes the live URL to tunnel_url.txt.
$key = "$env:USERPROFILE\.ssh\tunnel_ed25519"
$urlFile = "$PSScriptRoot\tunnel_url.txt"
$logFile = "$env:TEMP\tunnel_live.log"
while ($true) {
  "[$(Get-Date -Format HH:mm:ss)] connecting..." | Out-File $logFile -Append
  # -tt forces a pseudo-terminal so localhost.run prints the URL to stdout
  & ssh -i $key -o StrictHostKeyChecking=no -o IdentitiesOnly=yes `
        -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes `
        -o ExitOnForwardFailure=yes `
        -R 80:127.0.0.1:8000 nokey@localhost.run 2>&1 | ForEach-Object {
    $_ | Out-File $logFile -Append
    if ($_ -match "(https://[a-z0-9]+\.lhr\.life)") {
      $matches[1] | Out-File $urlFile -Encoding ascii
    }
  }
  "[$(Get-Date -Format HH:mm:ss)] tunnel dropped, reconnecting in 3s" | Out-File $logFile -Append
  Start-Sleep -Seconds 3
}
