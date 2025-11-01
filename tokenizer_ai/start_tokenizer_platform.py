# start_tokenizer_platform.py - Script de démarrage pour la plateforme de tokenizer universel

import subprocess
import sys
import time
import os
from pathlib import Path

def check_tokenizer_dependencies():
    """Vérifie les dépendances spécifiques au tokenizer"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'transformers', 'tokenizers', 
        'sentencepiece', 'spacy', 'polyglot', 'langdetect', 'textstat',
        'nltk', 'scikit-learn', 'fasttext', 'pandas', 'numpy', 'plotly',
        'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'polyglot':
                # Polyglot peut être problématique, le rendre optionnel
                continue
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Packages manquants pour la plateforme tokenizer:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\n📦 Installez-les avec:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Dépendances tokenizer installées")
    return True

def download_nltk_data():
    """Télécharge les données NLTK nécessaires"""
    try:
        import nltk
        print("📥 Téléchargement des données NLTK...")
        
        nltk_downloads = [
            'stopwords',
            'punkt', 
            'averaged_perceptron_tagger',
            'wordnet'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"⚠️  Impossible de télécharger {item}: {e}")
        
        print("✅ Données NLTK prêtes")
        return True
        
    except Exception as e:
        print(f"❌ Erreur téléchargement NLTK: {e}")
        return False

def setup_spacy_models():
    """Configure les modèles spaCy de base"""
    try:
        print("📥 Vérification des modèles spaCy...")
        
        # Essayer de charger un modèle multilingue de base
        try:
            import spacy
            from spacy.lang.xx import MultiLanguage
            nlp = MultiLanguage()
            print("✅ Modèle spaCy multilingue disponible")
        except Exception:
            print("⚠️  Modèle spaCy multilingue non disponible (optionnel)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Configuration spaCy: {e}")
        return True  # Non bloquant

def create_tokenizer_directories():
    """Crée les dossiers nécessaires pour le tokenizer"""
    directories = [
        'custom_tokenizers',
        'multilingual_corpus',
        'trained_tokenizer_models',
        'tokenizer_analysis'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Dossier tokenizer créé: {directory}")

def start_tokenizer_api():
    """Démarre l'API tokenizer"""
    print("🚀 Démarrage de l'API Universal Tokenizer...")
    
    try:
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "universal_tokenizer_api:app",
            "--host", "0.0.0.0", 
            "--port", "8008",
            "--reload"
        ])
        
        print("✅ API Tokenizer démarrée sur http://localhost:8008")
        return api_process
        
    except Exception as e:
        print(f"❌ Erreur démarrage API Tokenizer: {e}")
        return None

def start_tokenizer_dashboard():
    """Démarre l'interface Streamlit du tokenizer"""
    print("🎨 Démarrage de l'interface Tokenizer...")
    
    # Attendre que l'API soit prête
    print("⏳ Attente du démarrage de l'API Tokenizer...")
    time.sleep(8)
    
    try:
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "universal_tokenizer_dashboard.py",
            "--server.port", "8009",
            "--server.address", "localhost"
        ])
        
        print("✅ Interface Tokenizer démarrée sur http://localhost:8009")
        return streamlit_process
        
    except Exception as e:
        print(f"❌ Erreur démarrage interface Tokenizer: {e}")
        return None

def check_integration_with_ai_platform():
    """Vérifie la disponibilité de la plateforme IA pour intégration"""
    try:
        import requests
        response = requests.get("http://localhost:8006/health", timeout=2)
        if response.status_code == 200:
            print("🔗 AI Training Platform détectée - intégration disponible")
            return True
    except:
        pass
    
    print("ℹ️  AI Training Platform non disponible - fonctionnera en mode standalone")
    return False

def main():
    """Fonction principale de démarrage"""
    print("🔤 Démarrage de la Universal Tokenizer Platform")
    print("=" * 55)
    
    # Vérifications préliminaires
    if not check_tokenizer_dependencies():
        print("\n❌ Installez les dépendances manquantes avant de continuer")
        return
    
    # Configuration initiale
    download_nltk_data()
    setup_spacy_models()
    create_tokenizer_directories()
    
    # Vérifier l'intégration avec la plateforme IA
    ai_integration = check_integration_with_ai_platform()
    
    # Démarrer les services
    api_process = start_tokenizer_api()
    if not api_process:
        print("❌ Impossible de démarrer l'API Tokenizer")
        return
    
    dashboard_process = start_tokenizer_dashboard()
    if not dashboard_process:
        print("❌ Impossible de démarrer l'interface Tokenizer")
        api_process.terminate()
        return
    
    print("\n🎉 Universal Tokenizer Platform démarrée avec succès!")
    print("🔤 API Tokenizer: http://localhost:8008")
    print("🎨 Interface Tokenizer: http://localhost:8009")
    print("📚 Documentation API: http://localhost:8008/docs")
    
    if ai_integration:
        print("🤖 Intégration IA disponible: http://localhost:8006")
        print("🔗 Dashboard IA: http://localhost:8007")
    
    print("\n📋 Fonctionnalités disponibles:")
    print("   • Tokenisation multilingue universelle")
    print("   • Analyse linguistique avancée")
    print("   • Entraînement de tokenizers personnalisés")
    print("   • Gestion de corpus multilingues")
    print("   • Comparaison de performances")
    print("   • Analytics et statistiques")
    if ai_integration:
        print("   • Intégration avec AI Training Platform")
    
    print("\n⚠️  Appuyez sur Ctrl+C pour arrêter tous les services")
    
    try:
        # Boucle de monitoring
        while True:
            time.sleep(5)
            
            # Vérifier que les processus sont toujours vivants
            if api_process.poll() is not None:
                print("⚠️  API Tokenizer s'est arrêtée de manière inattendue")
                break
                
            if dashboard_process.poll() is not None:
                print("⚠️  Interface Tokenizer s'est arrêtée de manière inattendue")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Arrêt des services tokenizer...")
        
        # Arrêter les processus
        if dashboard_process:
            dashboard_process.terminate()
            print("✅ Interface Tokenizer arrêtée")
            
        if api_process:
            api_process.terminate() 
            print("✅ API Tokenizer arrêtée")
        
        print("✅ Universal Tokenizer Platform arrêtée avec succès")

if __name__ == "__main__":
    main()