# diagnostic_autosci.ps1 - Script de diagnostic complet
Write-Host "=== DIAGNOSTIC AUTOSCI PIPELINE ===" -ForegroundColor Cyan

# 1. Vérifier la structure des fichiers
Write-Host "`n1. VERIFICATION DES FICHIERS:" -ForegroundColor Yellow
$requiredFiles = @(
    "autosci_main.py",
    "deployment_platform.py", 
    "streamlit_dashboard.py",
    "requirements.txt",
    "requirements-deployment.txt",
    "requirements-streamlit.txt"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  OK: $file" -ForegroundColor Green
    } else {
        Write-Host "  MANQUANT: $file" -ForegroundColor Red
        $missingFiles += $file
    }
}

# 2. Vérifier Python et pip
Write-Host "`n2. VERIFICATION DE PYTHON:" -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERREUR: Python non trouve" -ForegroundColor Red
}

try {
    $pipVersion = pip --version 2>&1
    Write-Host "  Pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERREUR: Pip non trouve" -ForegroundColor Red
}

# 3. Vérifier l'environnement virtuel
Write-Host "`n3. VERIFICATION ENVIRONNEMENT VIRTUEL:" -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  OK: Environnement virtuel existe" -ForegroundColor Green
    if (Test-Path "venv\Scripts\python.exe") {
        Write-Host "  OK: Python dans venv" -ForegroundColor Green
    } else {
        Write-Host "  ERREUR: Python manquant dans venv" -ForegroundColor Red
    }
} else {
    Write-Host "  MANQUANT: Environnement virtuel" -ForegroundColor Red
}

# 4. Vérifier les jobs PowerShell
Write-Host "`n4. VERIFICATION DES JOBS POWERSHELL:" -ForegroundColor Yellow
$jobs = Get-Job -ErrorAction SilentlyContinue
if ($jobs) {
    foreach ($job in $jobs) {
        Write-Host "  Job: $($job.Name) - Etat: $($job.State)" -ForegroundColor Cyan
        if ($job.State -eq "Failed") {
            Write-Host "    ERREUR dans le job:" -ForegroundColor Red
            Receive-Job -Id $job.Id | Write-Host -ForegroundColor Red
        }
    }
} else {
    Write-Host "  AUCUN job PowerShell actif" -ForegroundColor Yellow
}

# 5. Vérifier les ports
Write-Host "`n5. VERIFICATION DES PORTS:" -ForegroundColor Yellow
$ports = @(5000, 8000, 8001, 8002, 8501)
foreach ($port in $ports) {
    $connection = Test-NetConnection -ComputerName "localhost" -Port $port -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        Write-Host "  Port $port : OCCUPE" -ForegroundColor Red
    } else {
        Write-Host "  Port $port : LIBRE" -ForegroundColor Green
    }
}

# 6. Vérifier les processus Python
Write-Host "`n6. VERIFICATION DES PROCESSUS:" -ForegroundColor Yellow
$pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        Write-Host "  Processus Python: PID $($proc.Id) - $($proc.ProcessName)" -ForegroundColor Cyan
    }
} else {
    Write-Host "  AUCUN processus Python actif" -ForegroundColor Yellow
}

# 7. Proposer des solutions
Write-Host "`n=== SOLUTIONS PROPOSEES ===" -ForegroundColor Green

if ($missingFiles.Count -gt 0) {
    Write-Host "`nFICHIERS MANQUANTS DETECTES:" -ForegroundColor Red
    Write-Host "Vous devez creer ces fichiers avec le contenu des artefacts:"
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Yellow
    }
}

Write-Host "`nOPTIONS DE REPARATION:" -ForegroundColor Cyan
Write-Host "A) Nettoyer et recommencer"
Write-Host "B) Demarrage manuel service par service"
Write-Host "C) Mode Docker (si Docker est installe)"
Write-Host "D) Creer les fichiers manquants"

$choice = Read-Host "`nVotre choix (A/B/C/D)"

switch ($choice.ToUpper()) {
    "A" {
        Write-Host "Nettoyage complet..." -ForegroundColor Yellow
        
        # Arrêter tous les jobs
        Get-Job | Stop-Job -PassThru | Remove-Job
        
        # Tuer les processus Python
        Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
        
        # Supprimer et recréer l'environnement virtuel
        if (Test-Path "venv") {
            Remove-Item "venv" -Recurse -Force
        }
        
        Write-Host "Création nouvel environnement virtuel..." -ForegroundColor Green
        python -m venv venv
        
        Write-Host "Nettoyage terminé. Relancez le script de démarrage." -ForegroundColor Green
    }
    
    "B" {
        Write-Host "Demarrage manuel..." -ForegroundColor Yellow
        
        # S'assurer que l'environnement virtuel est activé
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & ".\venv\Scripts\Activate.ps1"
        }
        
        Write-Host "Installation des dependances..." -ForegroundColor Green
        pip install mlflow uvicorn fastapi streamlit pandas numpy scikit-learn
        
        Write-Host "`nOuvrez 4 terminaux PowerShell et executez ces commandes:"
        Write-Host "Terminal 1: python -m mlflow server --host 127.0.0.1 --port 5000" -ForegroundColor Cyan
        Write-Host "Terminal 2: python -m uvicorn autosci_main:app --port 8000" -ForegroundColor Cyan
        Write-Host "Terminal 3: python -m uvicorn deployment_platform:app --port 8002" -ForegroundColor Cyan
        Write-Host "Terminal 4: python -m streamlit run streamlit_dashboard.py --server.port 8501" -ForegroundColor Cyan
    }
    
    "C" {
        Write-Host "Tentative avec Docker..." -ForegroundColor Yellow
        
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            Write-Host "Demarrage avec Docker Compose..." -ForegroundColor Green
            docker-compose up -d
        } else {
            Write-Host "Docker n'est pas installe" -ForegroundColor Red
        }
    }
    
    "D" {
        Write-Host "Creation des fichiers manquants..." -ForegroundColor Yellow
        
        # Créer autosci_main.py minimal
        if (-not (Test-Path "autosci_main.py")) {
            @"
from fastapi import FastAPI
import mlflow

app = FastAPI(title="AutoSci Pipeline API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "AutoSci Pipeline API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {"experiments": len(experiments)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@ | Out-File -FilePath "autosci_main.py" -Encoding UTF8
            Write-Host "Cree: autosci_main.py (version minimale)" -ForegroundColor Green
        }
        
        # Créer deployment_platform.py minimal
        if (-not (Test-Path "deployment_platform.py")) {
            @"
from fastapi import FastAPI

app = FastAPI(title="Deployment Platform API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Deployment Platform API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
"@ | Out-File -FilePath "deployment_platform.py" -Encoding UTF8
            Write-Host "Cree: deployment_platform.py (version minimale)" -ForegroundColor Green
        }
        
        # Créer streamlit_dashboard.py minimal
        if (-not (Test-Path "streamlit_dashboard.py")) {
            @"
import streamlit as st

st.title("AutoSci Pipeline Dashboard")
st.write("Bienvenue dans AutoSci Pipeline!")

if st.button("Test connexion"):
    st.success("Dashboard fonctionne correctement!")

st.sidebar.title("Navigation")
st.sidebar.write("Version minimale du dashboard")
"@ | Out-File -FilePath "streamlit_dashboard.py" -Encoding UTF8
            Write-Host "Cree: streamlit_dashboard.py (version minimale)" -ForegroundColor Green
        }
        
        Write-Host "Fichiers crees. Vous pouvez maintenant les completer avec le contenu des artefacts." -ForegroundColor Green
    }
}

Write-Host "`n=== FIN DU DIAGNOSTIC ===" -ForegroundColor Cyan