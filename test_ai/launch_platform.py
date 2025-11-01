"""
launch_platform.py - Script de lancement complet de la plateforme
Lance automatiquement MLflow, l'API et l'interface Streamlit

Usage: python launch_platform.py
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path
import webbrowser

class PlatformLauncher:
    """Lanceur de la plateforme complète"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.processes = []
        
    def print_banner(self):
        """Affiche la bannière de démarrage"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🤖  AI MODEL BENCHMARK PLATFORM v2.0  🤖                 ║
║                                                               ║
║     Plateforme complète de test et évaluation                ║
║     de modèles d'Intelligence Artificielle                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_dependencies(self):
        """Vérifie les dépendances"""
        print("\n🔍 Vérification des dépendances...")
        
        required = {
            "fastapi": "FastAPI",
            "uvicorn": "Uvicorn",
            "streamlit": "Streamlit",
            "mlflow": "MLflow",
            "jose": "Python-Jose",
            "passlib": "Passlib",
            "requests": "Requests",
            "pandas": "Pandas",
            "plotly": "Plotly"
        }
        
        missing = []
        for module, name in required.items():
            try:
                __import__(module)
                print(f"  ✅ {name}")
            except ImportError:
                missing.append(name)
                print(f"  ❌ {name}")
        
        if missing:
            print(f"\n⚠️  Dépendances manquantes: {', '.join(missing)}")
            print("📦 Installation automatique...")
            
            packages = [
                "fastapi",
                "uvicorn[standard]",
                "streamlit",
                "mlflow",
                "python-jose[cryptography]",
                "passlib[bcrypt]",
                "requests",
                "pandas",
                "plotly",
                "pydantic[email]",
                "python-multipart",
                "numpy"
            ]
            
            subprocess.run([sys.executable, "-m", "pip", "install"] + packages)
            print("✅ Dépendances installées!")
        else:
            print("✅ Toutes les dépendances sont présentes")
    
    def create_directories(self):
        """Crée les dossiers nécessaires"""
        print("\n📁 Création des dossiers...")
        
        directories = [
            "uploaded_models",
            "logs",
            "mlruns"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
        
        print("✅ Dossiers créés")
    
    def check_files(self):
        """Vérifie que les fichiers nécessaires existent"""
        print("\n📄 Vérification des fichiers...")

        # Chaque service a une liste de fichiers possibles
        required_files = {
            "API FastAPI": ["api_complete.py", "test_api.py"],
            "Interface Streamlit": ["frontend_complete.py", "test_dashboard.py"]
        }

        missing = []
        for service, options in required_files.items():
            found = False
            for file in options:
                if (self.base_dir / file).exists():
                    found = True
                    break
            if not found:
                missing.append(f"{service} ({' ou '.join(options)})")

        if missing:
            print(f"❌ Fichiers manquants: {', '.join(missing)}")
            print("\n⚠️  Veuillez vous assurer que les fichiers suivants sont présents:")
            for service, options in required_files.items():
                print(f"   - {service} : {', '.join(options)}")
            sys.exit(1)

    print("✅ Tous les fichiers sont présents")
    
    def start_mlflow(self):
        """Démarre MLflow"""
        print("\n🚀 Démarrage de MLflow...")
        
        try:
            mlflow_process = subprocess.Popen(
                ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir)
            )
            
            self.processes.append(("MLflow", mlflow_process))
            
            # Attendre que MLflow soit prêt
            for i in range(30):
                try:
                    response = requests.get("http://localhost:5000", timeout=1)
                    if response.status_code == 200:
                        print("✅ MLflow démarré: http://localhost:5000")
                        return True
                except:
                    time.sleep(1)
            
            print("⚠️  MLflow met du temps à démarrer...")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de MLflow: {e}")
            return False
    
    def start_api(self):
        """Démarre l'API FastAPI"""
        print("\n🚀 Démarrage de l'API FastAPI...")
        
        try:
            api_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "test_api:app",
                    "--host", "0.0.0.0",
                    "--port", "8000",
                    "--reload"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir)
            )
            
            self.processes.append(("API", api_process))
            
            # Attendre que l'API soit prête
            for i in range(30):
                try:
                    response = requests.get("http://localhost:8000", timeout=1)
                    if response.status_code == 200:
                        print("✅ API démarrée: http://localhost:8000")
                        print("📖 Documentation: http://localhost:8000/docs")
                        return True
                except:
                    time.sleep(1)
            
            print("⚠️  L'API met du temps à démarrer...")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de l'API: {e}")
            return False
    
    def start_streamlit(self):
        """Démarre l'interface Streamlit"""
        print("\n🚀 Démarrage de Streamlit...")
        
        try:
            streamlit_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "test_dashboard.py",
                    "--server.port", "8501",
                    "--server.address", "0.0.0.0"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_dir)
            )
            
            self.processes.append(("Streamlit", streamlit_process))
            
            # Attendre que Streamlit soit prêt uvicorn test_api:app --reload --port 8008
            time.sleep(5)
            print("✅ Streamlit démarré: http://localhost:8001")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de Streamlit: {e}")
            return False
    
    def show_status(self):
        """Affiche le statut de la plateforme"""
        status = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║               🎉  PLATEFORME LANCÉE AVEC SUCCÈS!  🎉          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📍 Services disponibles:

  🌐 Interface Streamlit  : http://localhost:8501
  🔌 API FastAPI          : http://localhost:8000
  📖 Documentation API    : http://localhost:8000/docs
  📊 MLflow UI            : http://localhost:5000

═══════════════════════════════════════════════════════════════

🔐 Compte administrateur par défaut:
   Utilisateur: admin
   Mot de passe: admin123

💡 Conseils pour démarrer:
   1. Ouvrez http://localhost:8501 dans votre navigateur
   2. Connectez-vous avec le compte admin
   3. Uploadez votre premier modèle
   4. Lancez des tests de performance
   5. Consultez les résultats et le leaderboard

═══════════════════════════════════════════════════════════════

⚠️  Appuyez sur Ctrl+C pour arrêter tous les services

═══════════════════════════════════════════════════════════════
        """
        print(status)
    
    def open_browser(self):
        """Ouvre le navigateur"""
        print("\n🌐 Ouverture du navigateur...")
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass
    
    def run(self):
        """Lance toute la plateforme"""
        try:
            self.print_banner()
            
            # Vérifications préliminaires
            self.check_dependencies()
            self.check_files()
            self.create_directories()
            
            print("\n" + "="*65)
            print("\n🚀 LANCEMENT DES SERVICES\n")
            
            # Démarrer les services
            mlflow_ok = self.start_mlflow()
            time.sleep(3)
            
            api_ok = self.start_api()
            time.sleep(3)
            
            streamlit_ok = self.start_streamlit()
            time.sleep(2)
            
            if mlflow_ok and api_ok and streamlit_ok:
                self.show_status()
                self.open_browser()
                
                # Garder le script actif
                print("\n✅ Plateforme en cours d'exécution...\n")
                while True:
                    time.sleep(1)
            else:
                print("\n❌ Erreur lors du démarrage de certains services")
                self.cleanup()
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Arrêt de la plateforme...")
            self.cleanup()
            print("\n✅ Tous les services ont été arrêtés proprement")
            print("👋 À bientôt!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}")
            self.cleanup()
            sys.exit(1)
    
    def cleanup(self):
        """Arrête tous les processus"""
        print("\n🧹 Nettoyage des processus...")
        for name, process in self.processes:
            try:
                print(f"  Arrêt de {name}...")
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("✅ Nettoyage terminé")

# ============================================================
# COMMANDES RAPIDES
# ============================================================

def show_help():
    """Affiche l'aide"""
    help_text = """
╔═══════════════════════════════════════════════════════════════╗
║          AI MODEL BENCHMARK PLATFORM - AIDE                   ║
╚═══════════════════════════════════════════════════════════════╝

USAGE:
  python launch_platform.py [command]

COMMANDES:
  start     Lance toute la plateforme (défaut)
  stop      Arrête tous les services
  status    Affiche le statut des services
  help      Affiche cette aide

EXEMPLES:
  python launch_platform.py
  python launch_platform.py start
  python launch_platform.py status

SERVICES LANCÉS:
  - MLflow Server (port 5000)
  - FastAPI Backend (port 8000)
  - Streamlit Frontend (port 8501)

FICHIERS REQUIS:
  - test_api.py

  - test_dashboard.py

Pour plus d'informations, consultez la documentation.
    """
    print(help_text)

def check_status():
    """Vérifie le statut des services"""
    print("\n🔍 Vérification du statut des services...\n")
    
    services = {
        "MLflow": "http://localhost:5000",
        "API": "http://localhost:8000",
        "Streamlit": "http://localhost:8501"
    }
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✅ {name:15} : Running ({url})")
            else:
                print(f"  ⚠️  {name:15} : Responding with status {response.status_code}")
        except:
            print(f"  ❌ {name:15} : Not running")
    
    print()



# ============================================================
# POINT D'ENTRÉE
# ============================================================

PID_FILE = Path("platform_services.pid")

def save_pids(processes):
    """Sauvegarde les PID dans un fichier"""
    with open(PID_FILE, "w") as f:
        for name, process in processes:
            f.write(f"{name}:{process.pid}\n")

def load_pids():
    """Charge les PID depuis le fichier"""
    if not PID_FILE.exists():
        return []
    with open(PID_FILE, "r") as f:
        lines = f.readlines()
    pids = []
    for line in lines:
        try:
            name, pid = line.strip().split(":")
            pids.append((name, int(pid)))
        except:
            pass
    return pids

def stop_services():
    """Arrête les services en utilisant le fichier PID"""
    if not PID_FILE.exists():
        print("⚠️  Aucun service n’est en cours (fichier PID manquant)")
        return

    print("\n🛑 Arrêt des services via PID...")
    pids = load_pids()
    for name, pid in pids:
        try:
            print(f"  Arrêt de {name} (PID {pid})...")
            os.kill(pid, 9)  # SIGKILL forcé
        except Exception as e:
            print(f"  ⚠️  Impossible d’arrêter {name} (PID {pid}): {e}")
    PID_FILE.unlink(missing_ok=True)
    print("✅ Tous les services ont été arrêtés")

def restart_services():
    """Redémarre intelligemment la plateforme"""
    print("\n🔄 Redémarrage de la plateforme...")

    if PID_FILE.exists():
        print("🛑 Des services tournent déjà, arrêt en cours...")
        stop_services()
        time.sleep(2)
    else:
        print("ℹ️ Aucun service détecté, lancement direct...")

    launcher = PlatformLauncher()
    launcher.run()
    save_pids(launcher.processes)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "start":
        launcher = PlatformLauncher()
        launcher.run()
        save_pids(launch.processes)
    elif sys.argv[1] == "status":
        check_status()
    elif sys.argv[1] == "help":
        show_help()
    elif sys.argv[1] == "stop":
        stop_services()
    elif sys.argv[1] == "restart":
        restart_services()
    else:
        print("❌ Commande inconnue. Utilisez: start | stop | restart | status | help")

