# start_all_services_v2.ps1 - Script PowerShell mis à jour
Write-Host "Demarrage d'AutoSci Pipeline v2.0 avec Personal Data Platform" -ForegroundColor Green

# Créer les dossiers nécessaires
Write-Host "Creation des dossiers..." -ForegroundColor Yellow
$folders = @(
    "data\raw", "data\processed", "data\synthetic",
    "models", "logs", "downloaded_models", "deployed_models",
    "mlruns", "ssl", "collected_data", "processed_data",
    "anonymized_data", "temp"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
    }
}

Write-Host "Dossiers crees avec succes" -ForegroundColor Yellow

# Vérifier Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERREUR: Docker n'est pas installe" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "ERREUR: Docker Compose n'est pas installe" -ForegroundColor Red
    exit 1
}

# Menu de choix
Write-Host ""
Write-Host "Choisissez le mode de demarrage:" -ForegroundColor Cyan
Write-Host "1) Developpement (services locaux Python)"
Write-Host "2) Production (Docker Compose)"
Write-Host "3) Personal Data Platform uniquement"
Write-Host "4) AutoSci Pipeline uniquement"
$choice = Read-Host "Votre choix (1-4)"

switch ($choice) {
    "1" {
        Write-Host "Mode Developpement Complet" -ForegroundColor Green
        
        # Créer un environnement virtuel
        if (-not (Test-Path "venv")) {
            Write-Host "Creation de l'environnement virtuel..." -ForegroundColor Yellow
            python -m venv venv
        }
        
        # Activer l'environnement virtuel
        Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
        
        # Installer les dépendances
        Write-Host "Installation des dependances..." -ForegroundColor Yellow
        pip install -r requirements.txt
        pip install -r requirements-deployment.txt
        pip install -r requirements-streamlit.txt
        pip install psutil fastapi-users
        
        # Initialiser les bases de données
        Write-Host "Initialisation des bases de donnees..." -ForegroundColor Yellow
        python -c @"
import sqlite3
import os

# Base MLflow
if not os.path.exists('mlflow.db'):
    conn = sqlite3.connect('mlflow.db')
    conn.close()
    print('Base MLflow creee')

# Base plateforme de déploiement
if not os.path.exists('deployment_platform.db'):
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE deployed_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            model_name TEXT NOT NULL,
            framework TEXT,
            model_type TEXT,
            metrics TEXT,
            deployment_date TEXT,
            status TEXT DEFAULT 'active',
            model_path TEXT,
            pricing TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            purchase_type TEXT,
            price REAL,
            email TEXT,
            transaction_id TEXT,
            status TEXT,
            created_at TEXT,
            expires_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            model_id TEXT,
            subscription_type TEXT,
            status TEXT,
            created_at TEXT,
            expires_at TEXT,
            usage_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print('Base deployment_platform creee')

# Base Personal Data Platform
if not os.path.exists('personal_data_platform.db'):
    conn = sqlite3.connect('personal_data_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE data_collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            collection_id TEXT UNIQUE NOT NULL,
            config TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            completed_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE collected_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            file_name TEXT,
            file_path TEXT,
            data_type TEXT,
            source_type TEXT,
            size_bytes INTEGER,
            created_at TEXT,
            processed_at TEXT,
            metadata TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE data_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            analysis_type TEXT,
            results TEXT,
            visualizations TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE data_studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            problem_type TEXT,
            model_performance TEXT,
            cleaned_data_path TEXT,
            model_path TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE data_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            seller_id TEXT NOT NULL,
            price REAL,
            description TEXT,
            anonymized_file_path TEXT,
            status TEXT DEFAULT 'available',
            created_at TEXT,
            sold_at TEXT,
            buyer_id TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE data_donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donation_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            donor_id TEXT NOT NULL,
            recipient_organization TEXT,
            purpose TEXT,
            anonymized_file_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            completed_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE user_consents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            consent_given BOOLEAN,
            timestamp TEXT,
            expires_at TEXT,
            revoked_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print('Base personal_data_platform creee')

print('Toutes les bases de donnees initialisees')
"@
        
        # Démarrer les services
        Write-Host "Demarrage des services..." -ForegroundColor Green
        
        # MLflow
        Write-Host "Demarrage MLflow Server..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///mlflow.db" 
        } -Name "MLflowJob"
        
        # API AutoSci
        Write-Host "Demarrage AutoSci API..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn autosci_main:app --reload --port 8000 
        } -Name "APIJob"
        
        # Plateforme de Déploiement
        Write-Host "Demarrage Deployment Platform..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn deployment_platform:app --reload --port 8002 
        } -Name "DeploymentJob"
        
        # Personal Data Platform API
        Write-Host "Demarrage Personal Data Platform API..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn personal_data_platform:app --reload --port 8003 
        } -Name "PersonalDataJob"
        
        # Streamlit AutoSci Dashboard
        Write-Host "Demarrage AutoSci Dashboard..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m streamlit run streamlit_dashboard.py --server.port 8501 
        } -Name "StreamlitAutoSciJob"
        
        # Streamlit Personal Data Dashboard
        Write-Host "Demarrage Personal Data Dashboard..."
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m streamlit run personal_data_dashboard.py --server.port 8504 
        } -Name "StreamlitPersonalDataJob"
        
        Write-Host "Attente du demarrage des services (60s)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
    
    "2" {
        Write-Host "Mode Production (Docker Compose)" -ForegroundColor Green
        
        Write-Host "Construction des images Docker..." -ForegroundColor Yellow
        docker-compose -f docker-compose-v2.yml build
        
        Write-Host "Demarrage des services..." -ForegroundColor Yellow
        docker-compose -f docker-compose-v2.yml up -d
        
        Write-Host "Attente du demarrage des services (90s)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 90
    }
    
    "3" {
        Write-Host "Personal Data Platform uniquement" -ForegroundColor Green
        
        # Activer l'environnement virtuel
        & ".\venv\Scripts\Activate.ps1"
        
        # Démarrer seulement Personal Data Platform
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn personal_data_platform:app --reload --port 8003 
        } -Name "PersonalDataJob"
        
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m streamlit run personal_data_dashboard.py --server.port 8504 
        } -Name "StreamlitPersonalDataJob"
        
        Write-Host "Attente du demarrage (30s)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
    
    "4" {
        Write-Host "AutoSci Pipeline uniquement" -ForegroundColor Green
        
        # Activer l'environnement virtuel
        & ".\venv\Scripts\Activate.ps1"
        
        # Démarrer seulement AutoSci
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///mlflow.db" 
        } -Name "MLflowJob"
        
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn autosci_main:app --reload --port 8000 
        } -Name "APIJob"
        
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m uvicorn deployment_platform:app --reload --port 8002 
        } -Name "DeploymentJob"
        
        Start-Job -ScriptBlock { 
            & ".\venv\Scripts\python.exe" -m streamlit run streamlit_dashboard.py --server.port 8501 
        } -Name "StreamlitAutoSciJob"
        
        Write-Host "Attente du demarrage (45s)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 45
    }
    
    default {
        Write-Host "Choix invalide" -ForegroundColor Red
        exit 1
    }
}

# Vérification des services
Write-Host "Verification des services..." -ForegroundColor Green

$services = @()

# Services selon le choix
switch ($choice) {
    {$_ -in "1", "2"} {
        $services = @(
            @{Name="MLflow"; Url="http://localhost:5000"},
            @{Name="AutoSci API"; Url="http://localhost:8000/docs"},
            @{Name="Deployment Platform"; Url="http://localhost:8002/docs"},
            @{Name="Personal Data API"; Url="http://localhost:8003/docs"},
            @{Name="AutoSci Dashboard"; Url="http://localhost:8501"},
            @{Name="Personal Data Dashboard"; Url="http://localhost:8504"}
        )
    }
    "3" {
        $services = @(
            @{Name="Personal Data API"; Url="http://localhost:8003/docs"},
            @{Name="Personal Data Dashboard"; Url="http://localhost:8504"}
        )
    }
    "4" {
        $services = @(
            @{Name="MLflow"; Url="http://localhost:5000"},
            @{Name="AutoSci API"; Url="http://localhost:8000/docs"},
            @{Name="Deployment Platform"; Url="http://localhost:8002/docs"},
            @{Name="AutoSci Dashboard"; Url="http://localhost:8501"}
        )
    }
}

foreach ($service in $services) {
    $maxAttempts = 10
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -Method Head -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 404) {
                Write-Host "SUCCESS: $($service.Name) est accessible" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "Tentative $attempt/$maxAttempts pour $($service.Name)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 3
            $attempt++
        }
        
        if ($attempt -gt $maxAttempts) {
            Write-Host "ERREUR: $($service.Name) n'est pas accessible" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "AutoSci Pipeline v2.0 avec Personal Data Platform est maintenant operationnel!" -ForegroundColor Green
Write-Host ""
Write-Host "Acces aux services:" -ForegroundColor Cyan

switch ($choice) {
    {$_ -in "1", "2"} {
        Write-Host "  - AutoSci Dashboard       : http://localhost:8501"
        Write-Host "  - Personal Data Dashboard : http://localhost:8504"
        Write-Host "  - AutoSci API            : http://localhost:8000/docs"
        Write-Host "  - Personal Data API      : http://localhost:8003/docs"
        Write-Host "  - MLflow UI              : http://localhost:5000"
        Write-Host "  - Deployment Platform    : http://localhost:8002/docs"
    }
    "3" {
        Write-Host "  - Personal Data Dashboard : http://localhost:8504"
        Write-Host "  - Personal Data API      : http://localhost:8003/docs"
    }
    "4" {
        Write-Host "  - AutoSci Dashboard      : http://localhost:8501"
        Write-Host "  - AutoSci API           : http://localhost:8000/docs"
        Write-Host "  - MLflow UI             : http://localhost:5000"
        Write-Host "  - Deployment Platform   : http://localhost:8002/docs"
    }
}

Write-Host ""
Write-Host "Gestion:" -ForegroundColor Cyan
Write-Host "  - Arreter les services : .\stop_all_services_v2.ps1"
Write-Host "  - Voir les logs       : .\view_logs_v2.ps1"
Write-Host ""
Write-Host "Fonctionnalites Personal Data Platform:" -ForegroundColor Cyan
Write-Host "  1. Collecte securisee de donnees personnelles"
Write-Host "  2. Analyse automatique des donnees"
Write-Host "  3. Etudes data science automatisees" 
Write-Host "  4. Marketplace de donnees ethique"
Write-Host "  5. Gestion complete des consentements"
