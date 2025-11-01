"""
Script de lancement automatique des 47 plateformes NADA
Structure: 
- datasc/app.py (page d'accueil)
- datasc/{plateforme}/{plateforme}_app.py et {plateforme}_api.py
À placer dans le dossier datasc/
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# Configuration - Ce script doit être dans le dossier datasc
CURRENT_DIR = Path(__file__).parent  # Dossier datasc
BASE_DIR = CURRENT_DIR  # Les plateformes sont au même niveau que le script

# Liste des 47 plateformes avec leurs ports
PLATFORMS = [
    {"folder": "accelerateur_particules", "port": 8001},
    {"folder": "advanced_telescope_platform", "port": 8002},
    {"folder": "ai_connector", "port": 8003},
    {"folder": "ai_decision_platform", "port": 8004},
    {"folder": "ai_development_platform", "port": 8005},
    {"folder": "ai_lifecycle", "port": 8006},
    {"folder": "ai_quantique_biocomputing", "port": 8007},
    {"folder": "arvr_platform", "port": 8008},
    {"folder": "asi_platform", "port": 8009},
    {"folder": "autonomous_vehicle", "port": 8010},
    {"folder": "autosci_pipeline", "port": 8011},
    {"folder": "brain_organoid_platform", "port": 8012},
    {"folder": "business_tokenization", "port": 8013},
    {"folder": "collisionneur_particules", "port": 8014},
    {"folder": "conscience_artificielle", "port": 8015},
    {"folder": "conversation_director", "port": 8016},
    {"folder": "cosmic_intelligence", "port": 8017},
    {"folder": "cybersecurite", "port": 8018},
    {"folder": "cybersecurite_quantique_bio", "port": 8019},
    {"folder": "dark_matter_platform", "port": 8020},
    {"folder": "data_platform", "port": 8021},
    {"folder": "datacenter_platform", "port": 8022},
    {"folder": "energy_platform", "port": 8023},
    {"folder": "entrainement_ai", "port": 8024},
    {"folder": "fuse_plateform", "port": 8025},
    {"folder": "fusion_nuclear_lab", "port": 8026},
    {"folder": "holographic_multiverse", "port": 8027},
    {"folder": "intelligence_artificielle_generale", "port": 8028},
    {"folder": "intrication_quantique", "port": 8029},
    {"folder": "iso_certification", "port": 8030},
    {"folder": "media_intelligence_platform", "port": 8031},
    {"folder": "neuromorphic_exotic_matter", "port": 8032},
    {"folder": "nuclear_reactor", "port": 8033},
    {"folder": "optimisation", "port": 8034},
    {"folder": "optimisation_quantique_bio", "port": 8035},
    {"folder": "plateforme_test", "port": 8036},
    {"folder": "quantique_ia", "port": 8037},
    {"folder": "quantum_physics_platform", "port": 8038},
    {"folder": "robotique", "port": 8039},
    {"folder": "space_mechanics", "port": 8040},
    {"folder": "supercalculateur", "port": 8041},
    {"folder": "supraconducteur", "port": 8042},
    {"folder": "system_optimizer", "port": 8043},
    {"folder": "test_ai", "port": 8044},
    {"folder": "tokenizer_ai", "port": 8045},
    {"folder": "ultra_conservation_platform", "port": 8046},
]

processes = []

def launch_streamlit_app(platform):
    """Lance l'application Streamlit d'une plateforme"""
    folder_path = BASE_DIR / platform['folder']
    app_file = folder_path / f"{platform['folder']}_app.py"
    
    # Vérifier si le dossier et le fichier existent
    if not folder_path.exists():
        print(f"❌ Dossier non trouvé: {platform['folder']}")
        return None
    
    if not app_file.exists():
        print(f"❌ Fichier non trouvé: {platform['folder']}_app.py")
        return None
    
    # Commande Streamlit
    cmd = [
        "streamlit", "run",
        str(app_file),
        "--server.port", str(platform['port']),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(folder_path)
        )
        
        print(f"✅ {platform['folder']:50} | Port: {platform['port']} | PID: {process.pid}")
        return process
    
    except Exception as e:
        print(f"❌ Erreur: {platform['folder']} - {e}")
        return None

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    try:
        import streamlit
        print("✅ Streamlit installé")
        return True
    except ImportError:
        print("❌ Streamlit n'est pas installé!")
        print("   Installez-le avec: pip install streamlit")
        return False

def main():
    """Fonction principale"""
    print("=" * 90)
    print("🚀 LANCEMENT DES 47 PLATEFORMES NADA")
    print("=" * 90)
    print()
    
    # Vérifier les prérequis
    if not check_requirements():
        sys.exit(1)
    
    print(f"📁 Dossier de base: {BASE_DIR.absolute()}")
    print(f"📦 Nombre de plateformes à lancer: {len(PLATFORMS)}")
    print()
    
    # Scanner les dossiers disponibles
    print("🔍 Scan des dossiers disponibles...")
    available_folders = [d.name for d in BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"   Trouvés: {len(available_folders)} dossiers dans datasc/")
    print()
    
    print("🚀 Lancement des plateformes...")
    print("-" * 90)
    
    launched = 0
    failed = 0
    
    for i, platform in enumerate(PLATFORMS, 1):
        print(f"[{i:2d}/47] ", end="")
        
        process = launch_streamlit_app(platform)
        
        if process:
            processes.append(process)
            launched += 1
        else:
            failed += 1
        
        # Petit délai pour éviter la surcharge
        time.sleep(0.3)
    
    print("-" * 90)
    print()
    print("📊 RÉSUMÉ DU LANCEMENT")
    print("=" * 90)
    print(f"✅ Plateformes lancées avec succès: {launched}")
    print(f"❌ Plateformes échouées:           {failed}")
    print(f"📈 Total:                          {len(PLATFORMS)}")
    print()
    
    if launched > 0:
        print("🌐 ACCÈS AUX PLATEFORMES")
        print("=" * 90)
        print("Les plateformes sont accessibles sur:")
        print()
        
        # Afficher les 10 premières
        for i in range(min(10, launched)):
            platform = PLATFORMS[i]
            print(f"   • {platform['folder']:50} → http://localhost:{platform['port']}")
        
        if len(PLATFORMS) > 10:
            print(f"   ... et {len(PLATFORMS) - 10} autres plateformes")
        
        print()
        print("=" * 90)
        print("💡 COMMANDES UTILES")
        print("=" * 90)
        print("• Lancer la page d'accueil:")
        print("  cd datasc")
        print("  streamlit run app.py --server.port 8000")
        print()
        print("• Arrêter les plateformes: Appuyez sur Ctrl+C dans ce terminal")
        print()
        print("• Vérifier les ports ouverts:")
        print("  Linux/Mac: netstat -tulpn | grep LISTEN")
        print("  Windows:   netstat -an | findstr LISTEN")
        print()
        print("⏳ Plateformes en cours d'exécution... (Ctrl+C pour tout arrêter)")
        print()
        
        try:
            # Garder le script actif
            while True:
                # Vérifier si les processus sont toujours en vie
                alive = sum(1 for p in processes if p.poll() is None)
                
                if alive == 0:
                    print()
                    print("⚠️  Tous les processus se sont arrêtés!")
                    break
                
                # Afficher un point toutes les 5 secondes pour montrer que c'est actif
                print(".", end="", flush=True)
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n")
            print("=" * 90)
            print("🛑 ARRÊT DE TOUTES LES PLATEFORMES")
            print("=" * 90)
            print()
            
            for i, process in enumerate(processes, 1):
                try:
                    process.terminate()
                    print(f"[{i:2d}/{len(processes)}] Arrêt du processus PID {process.pid}...", end=" ")
                    process.wait(timeout=3)
                    print("✅")
                except subprocess.TimeoutExpired:
                    print("⏱️  Forçage...", end=" ")
                    process.kill()
                    print("✅")
                except Exception as e:
                    print(f"❌ {e}")
            
            print()
            print("✅ Toutes les plateformes ont été arrêtées.")
    
    else:
        print("❌ Aucune plateforme n'a pu être lancée!")
        print()
        print("🔍 VÉRIFICATIONS À FAIRE:")
        print("=" * 90)
        print("1. Vérifiez que vous êtes dans le bon dossier:")
        print(f"   Dossier actuel: {BASE_DIR.absolute()}")
        print()
        print("2. Vérifiez la structure:")
        print("   datasc/")
        print("   ├── app.py (page d'accueil)")
        print("   ├── launch_all.py (ce script)")
        print("   ├── accelerateur_particules/")
        print("   │   ├── accelerateur_particules_app.py")
        print("   │   └── accelerateur_particules_api.py")
        print("   ├── advanced_telescope_platform/")
        print("   │   ├── advanced_telescope_platform_app.py")
        print("   │   └── advanced_telescope_platform_api.py")
        print("   └── ... (45 autres dossiers)")
        print()
        print("3. Vérifiez que les fichiers *_app.py existent dans chaque dossier")
        print()
        print("4. Liste des dossiers trouvés:")
        for folder in available_folders[:10]:
            print(f"   - {folder}")
        if len(available_folders) > 10:
            print(f"   ... et {len(available_folders) - 10} autres")
    
    print()
    print("👋 Script terminé!")
    print("=" * 90)

if __name__ == "__main__":
    main()