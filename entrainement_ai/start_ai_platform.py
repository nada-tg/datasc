# start_ai_platform.py - Script de démarrage pour la plateforme d'entraînement IA

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'pandas', 'numpy', 'scikit-learn',
        'torch', 'tensorflow', 'xgboost', 'lightgbm', 'plotly', 'opencv-python',
        'librosa', 'whisper-openai', 'transformers', 'Pillow', 'requests',
        'websocket-client', 'psutil', 'GPUtil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Packages manquants:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Installez-les avec:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def create_directories():
    """Crée les dossiers nécessaires"""
    directories = [
        'trained_ai_models',
        'training_datasets', 
        'training_logs',
        'model_checkpoints',
        'uploaded_media',
        'processed_media_data',
        'trained_models',
        'marketplace_media'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Dossier créé: {directory}")

def start_api_server():
    """Démarre le serveur API"""
    print("🚀 Démarrage du serveur API d'entraînement...")
    
    try:
        # Démarrer le serveur API en arrière-plan
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "ai_training_platform:app",
            "--host", "0.0.0.0",
            "--port", "8006",
            "--reload"
        ])
        
        print("✅ Serveur API démarré sur http://localhost:8006")
        return api_process
        
    except Exception as e:
        print(f"❌ Erreur démarrage API: {e}")
        return None

def start_streamlit_app():
    """Démarre l'application Streamlit"""
    print("🎨 Démarrage de l'interface Streamlit...")
    
    # Attendre que l'API soit prête
    print("⏳ Attente du démarrage de l'API...")
    time.sleep(5)
    
    try:
        # Démarrer Streamlit
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "ai_training_dashboard.py",
            "--server.port", "8007",
            "--server.address", "localhost"
        ])
        
        print("✅ Interface Streamlit démarrée sur http://localhost:8007")
        return streamlit_process
        
    except Exception as e:
        print(f"❌ Erreur démarrage Streamlit: {e}")
        return None

def main():
    """Fonction principale de démarrage"""
    print("🤖 Démarrage de la Plateforme d'Entraînement IA")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_dependencies():
        return
    
    # Créer les dossiers
    create_directories()
    
    # Démarrer les services
    api_process = start_api_server()
    if not api_process:
        return
    
    streamlit_process = start_streamlit_app()
    if not streamlit_process:
        api_process.terminate()
        return
    
    print("\n🎉 Plateforme démarrée avec succès!")
    print("📊 API d'entraînement: http://localhost:8006")
    print("🎨 Interface utilisateur: http://localhost:8007")
    print("📚 Documentation API: http://localhost:8006/docs")
    print("\n⚠️  Appuyez sur Ctrl+C pour arrêter tous les services")
    
    try:
        # Attendre l'interruption
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt des services...")
        
        # Arrêter les processus
        if streamlit_process:
            streamlit_process.terminate()
        if api_process:
            api_process.terminate()
        
        print("✅ Services arrêtés avec succès")

if __name__ == "__main__":
    main()