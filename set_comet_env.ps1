# set_comet_env.ps1
# Cambia el valor de COMET_API_KEY abajo y ejecuta este script en PowerShell
# - Para la sesión actual: .\set_comet_env.ps1
# - Para persistir la variable (usuario): descomenta la línea setx y ejecuta como administrador o reinicia la terminal

# <-- PON TU API KEY AQUI (entre comillas)
$COMET_API_KEY = "q2wf2jK1D6ttAht5p6ia3yTEJ"

# Setea para la sesión actual
$env:COMET_API_KEY = $COMET_API_KEY
Write-Host "COMET_API_KEY seteada para la sesión actual"

# Guardar permanentemente en variables de usuario (opcional)
# Descomenta la siguiente línea si quieres guardarla permanentemente:
cmd /c "setx COMET_API_KEY \"$COMET_API_KEY\""
# Write-Host "COMET_API_KEY guardada permanentemente (reinicia terminal para que tenga efecto)"
