# stop_all_services_v2.ps1 - Script d'arrêt mis à jour
Write-Host "Arret d'AutoSci Pipeline v2.0..." -ForegroundColor Red

# Vérifier les jobs PowerShell
$jobs = Get-Job -ErrorAction SilentlyContinue
if ($jobs) {
    Write-Host "Mode developpement detecte - Arret des jobs PowerShell" -ForegroundColor Yellow
    
    # Arreter tous les jobs
    Stop-Job -Name "*Job" -ErrorAction SilentlyContinue
    Remove-Job -Name "*Job" -ErrorAction SilentlyContinue
    
    # Arreter les processus spécifiques
    $processNames = @("python", "uvicorn", "streamlit", "mlflow")
    foreach ($processName in $processNames) {
        $processes = Get-Process -Name $processName -ErrorAction SilentlyContinue
        if ($processes) {
            $processes | Stop-Process -Force -ErrorAction SilentlyContinue
            Write-Host "SUCCESS: $processName arrete" -ForegroundColor Green
        }
    }
} else {
    # Vérifier si Docker est en cours
    try {
        $dockerProcesses = docker-compose -f docker-compose-v2.yml ps -q 2>$null
        if ($dockerProcesses) {
            Write-Host "Mode Docker detecte" -ForegroundColor Yellow
            docker-compose -f docker-compose-v2.yml down
            Write-Host "SUCCESS: Conteneurs Docker arretes" -ForegroundColor Green
        } else {
            Write-Host "Aucun service detecte en cours d'execution" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Aucun service Docker detecte" -ForegroundColor Yellow
    }
}

Write-Host "AutoSci Pipeline v2.0 arrete avec succes" -ForegroundColor Green