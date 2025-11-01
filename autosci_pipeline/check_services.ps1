# check_services.ps1
Write-Host "Verification de sante d'AutoSci Pipeline" -ForegroundColor Cyan
Write-Host "========================================"

$services = @(
    @{Name="MLflow Server"; Url="http://localhost:5000"; Expected=200},
    @{Name="AutoSci API"; Url="http://localhost:8000/docs"; Expected=200},
    @{Name="Deployment Platform"; Url="http://localhost:8002/docs"; Expected=200},
    @{Name="Streamlit Dashboard"; Url="http://localhost:8501"; Expected=200}
)

$totalChecks = 0
$failedChecks = 0

foreach ($service in $services) {
    $totalChecks++
    Write-Host -NoNewline "Verification de $($service.Name)... "
    
    try {
        $response = Invoke-WebRequest -Uri $service.Url -Method Head -TimeoutSec 10 -ErrorAction Stop
        if ($response.StatusCode -eq $service.Expected) {
            Write-Host "OK ($($response.StatusCode))" -ForegroundColor Green
        } else {
            Write-Host "ERREUR ($($response.StatusCode))" -ForegroundColor Red
            $failedChecks++
        }
    } catch {
        Write-Host "ERREUR (Non accessible)" -ForegroundColor Red
        $failedChecks++
    }
}

Write-Host ""
Write-Host "Resume de sante:" -ForegroundColor Cyan
Write-Host "  - Services verifies : $totalChecks"
Write-Host "  - Services OK       : $($totalChecks - $failedChecks)"
Write-Host "  - Services en erreur: $failedChecks"

if ($failedChecks -eq 0) {
    Write-Host "  - Statut global     : SAIN" -ForegroundColor Green
} else {
    Write-Host "  - Statut global     : PROBLEMES DETECTES" -ForegroundColor Red
    Write-Host ""
    Write-Host "Essayez de redemarrer les services:" -ForegroundColor Yellow
    Write-Host "  .\stop_all_services.ps1"
    Write-Host "  .\start_all_services.ps1"
}