Add-MpPreference -ExclusionPath 'c:\Users\RoyVivasi\Documents\notebook\tools'
Add-MpPreference -ExclusionProcess 'ngrok.exe'
'ok' | Out-File 'c:\Users\RoyVivasi\Documents\notebook\tools\excl_done.txt' -Encoding ascii
