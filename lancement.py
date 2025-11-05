"""
NADA - Lanceur Unifié
Lance automatiquement toutes les 47 plateformes et affiche la page d'accueil
Structure: datasc/{plateforme}/{plateforme}_app.py
"""

import streamlit as st
import psycopg2
from psycopg2 import sql
import hashlib
import secrets
from datetime import datetime
import subprocess
import os
import sys
import time
import atexit
from pathlib import Path
import threading
import webbrowser

# ==================== LANCEMENT DES PLATEFORMES EN ARRIÈRE-PLAN ====================

CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR

# Liste des 47 plateformes avec leurs ports pour le lancement automatique
PLATFORMS_LAUNCH = [
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
    {"folder": "brain_organoid_platform", "port": 8012},
    {"folder": "business_tokenization", "port": 8013},
    {"folder": "collisionneur_particules", "port": 8014},
    {"folder": "conscience_artificielle", "port": 8015},
    {"folder": "conversation_director", "port": 8016},
    {"folder": "cosmic_intelligence", "port": 8017},
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

if 'processes' not in st.session_state:
    st.session_state.processes = []
if 'platforms_launched' not in st.session_state:
    st.session_state.platforms_launched = False

def cleanup_processes():
    """Nettoie tous les processus au démarrage"""
    for process in st.session_state.processes:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            try:
                process.kill()
            except:
                pass
    st.session_state.processes = []

atexit.register(cleanup_processes)

def launch_streamlit_app_background(platform):
    """Lance l'application Streamlit d'une plateforme en arrière-plan"""
    folder_path = BASE_DIR / platform['folder']
    app_file = folder_path / f"{platform['folder']}_app.py"
    
    if not folder_path.exists() or not app_file.exists():
        return None
    
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
        return process
    except:
        return None

def launch_all_platforms_background():
    """Lance toutes les plateformes en arrière-plan au démarrage"""
    if st.session_state.platforms_launched:
        return
    
    st.session_state.platforms_launched = True
    
    for platform in PLATFORMS_LAUNCH:
        process = launch_streamlit_app_background(platform)
        if process:
            st.session_state.processes.append(process)
        time.sleep(0.2)

# Lancer les plateformes au démarrage (une seule fois)
if not st.session_state.platforms_launched:
    launch_all_platforms_background()

# ==================== CODE ORIGINAL DE L'INDEX (app.py) ====================

# Configuration de la page
st.set_page_config(
    page_title="NADA - Next-gen Automation for Data and Augmented Reality",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    /* Styles généraux */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* En-tête */
    .header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .header h1 {
        color: #00d4ff;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .header p {
        color: #ffffff;
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        text-decoration: none;
        transition: all 0.3s ease;
        border: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .nav-button:hover {
        background: rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
        transform: translateY(-3px);
    }
    
    /* Grille des plateformes */
    .platform-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .platform-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(5px);
    }
    
    .platform-card:hover {
        background: rgba(0, 212, 255, 0.15);
        border-color: #00d4ff;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    .platform-card h3 {
        color: #00d4ff;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .platform-card p {
        color: #ffffff;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Lien de plateforme */
    .platform-link {
        display: block;
        width: 100%;
        padding: 0.7rem 2rem;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white !important;
        text-decoration: none;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
    }
    
    .platform-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        text-decoration: none;
        color: white !important;
    }
    
    /* Formulaires */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Configuration PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'database': 'nada_platform_db',
    'user': 'postgres',
    'password': 'nadaprojet',
    'port': 5432
}

# Liste des 47 plateformes avec leurs ports

PLATFORMS = [
    {"name": "Accélérateur de Particules", "slug": "accelerateur_particules", "icon": "⚛️", "type": "streamlit"},
    {"name": "Télescope Avancé", "slug": "advanced_telescope_platform", "icon": "🔭", "type": "streamlit"},
    {"name": "Connecteur IA", "slug": "ai_connector", "icon": "🔗", "type": "streamlit"},
    {"name": "Plateforme de Décision IA", "slug": "ai_decision_platform", "icon": "🧠", "type": "streamlit"},
    {"name": "Développement IA", "slug": "ai_development_platform", "icon": "💻", "type": "streamlit"},
    {"name": "Cycle de Vie IA", "slug": "ai_lifecycle", "icon": "♻️", "type": "streamlit"},
    {"name": "IA Quantique Biocomputing", "slug": "ai_quantique_biocomputing", "icon": "🧬", "type": "streamlit"},
    {"name": "Plateforme AR/VR", "slug": "arvr_platform", "icon": "🥽", "type": "streamlit"},
    {"name": "Plateforme ASI", "slug": "asi_platform", "icon": "🤖", "type": "streamlit"},
    {"name": "Véhicule Autonome", "slug": "autonomous_vehicle", "icon": "🚗", "type": "streamlit"},
    {"name": "Organoïde Cérébral", "slug": "brain_organoid_platform", "icon": "🧠", "type": "streamlit"},
    {"name": "Tokenisation Business", "slug": "business_tokenization", "icon": "💰", "type": "streamlit"},
    {"name": "Collisionneur de Particules", "slug": "collisionneur_particules", "icon": "💥", "type": "streamlit"},
    {"name": "Conscience Artificielle", "slug": "conscience_artificielle", "icon": "🌟", "type": "streamlit"},
    {"name": "Directeur de Conversation", "slug": "conversation_director", "icon": "💬", "type": "streamlit"},
    {"name": "Intelligence Cosmique", "slug": "cosmic_intelligence", "icon": "🌌", "type": "streamlit"},
    {"name": "Cybersécurité Quantique Bio", "slug": "cybersecurite_quantique_bio", "icon": "🛡️", "type": "streamlit"},
    {"name": "Matière Noire", "slug": "dark_matter_platform", "icon": "🌑", "type": "streamlit"},
    {"name": "Plateforme Data", "slug": "data_platform", "icon": "📊", "type": "streamlit"},
    {"name": "Datacenter", "slug": "datacenter_platform", "icon": "🖥️", "type": "streamlit"},
    {"name": "Plateforme Énergie", "slug": "energy_platform", "icon": "⚡", "type": "streamlit"},
    {"name": "Entraînement IA", "slug": "entrainement_ai", "icon": "🎯", "type": "streamlit"},
    {"name": "Plateforme Fusion", "slug": "fuse_plateform", "icon": "🔥", "type": "streamlit"},
    {"name": "Laboratoire Fusion Nucléaire", "slug": "fusion_nuclear_lab", "icon": "☢️", "type": "streamlit"},
    {"name": "Multivers Holographique", "slug": "holographic_multiverse", "icon": "🌐", "type": "streamlit"},
    {"name": "IA Générale", "slug": "intelligence_artificielle_generale", "icon": "🧩", "type": "streamlit"},
    {"name": "Intrication Quantique", "slug": "intrication_quantique", "icon": "🔗", "type": "streamlit"},
    {"name": "Certification ISO", "slug": "iso_certification", "icon": "📜", "type": "streamlit"},
    {"name": "Intelligence Média", "slug": "media_intelligence_platform", "icon": "📺", "type": "streamlit"},
    {"name": "Matière Exotique Neuromorphique", "slug": "neuromorphic_exotic_matter", "icon": "🧪", "type": "streamlit"},
    {"name": "Réacteur Nucléaire", "slug": "nuclear_reactor", "icon": "⚛️", "type": "streamlit"},
    {"name": "Optimisation", "slug": "optimisation", "icon": "📈", "type": "streamlit"},
    {"name": "Optimisation Quantique Bio", "slug": "optimisation_quantique_bio", "icon": "🔬", "type": "streamlit"},
    {"name": "Plateforme de Test", "slug": "plateforme_test", "icon": "🧪", "type": "streamlit"},
    {"name": "IA Quantique", "slug": "quantique_ia", "icon": "⚛️", "type": "streamlit"},
    {"name": "Physique Quantique", "slug": "quantum_physics_platform", "icon": "🔮", "type": "streamlit"},
    {"name": "Robotique", "slug": "robotique", "icon": "🦾", "type": "streamlit"},
    {"name": "Mécanique Spatiale", "slug": "space_mechanics", "icon": "🚀", "type": "streamlit"},
    {"name": "Supercalculateur", "slug": "supercalculateur", "icon": "💻", "type": "streamlit"},
    {"name": "Supraconducteur", "slug": "supraconducteur", "icon": "⚡", "type": "streamlit"},
    {"name": "Optimiseur Système", "slug": "system_optimizer", "icon": "⚙️", "type": "streamlit"},
    {"name": "Test IA", "slug": "test_ai", "icon": "🧪", "type": "streamlit"},
    {"name": "Tokenizer IA", "slug": "tokenizer_ai", "icon": "🔤", "type": "streamlit"},
    {"name": "Conservation Ultra", "slug": "ultra_conservation_platform", "icon": "🌍", "type": "streamlit"},
]
# Fonctions de base de données
def init_db():
    """Initialise la base de données"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return False

def hash_password(password):
    """Hache le mot de passe"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    """Enregistre un nouvel utilisateur"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        password_hash = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except psycopg2.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Erreur lors de l'inscription: {e}")
        return False

def login_user(username, password):
    """Connecte un utilisateur"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        password_hash = hash_password(password)
        cur.execute(
            "SELECT id, username, email FROM users WHERE username = %s AND password_hash = %s",
            (username, password_hash)
        )
        
        user = cur.fetchone()
        
        if user:
            cur.execute(
                "UPDATE users SET last_login = %s WHERE id = %s",
                (datetime.now(), user[0])
            )
            conn.commit()
        
        cur.close()
        conn.close()
        return user
    except Exception as e:
        st.error(f"Erreur lors de la connexion: {e}")
        return None

# Initialiser la session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# Initialiser la base de données
init_db()

# Pages secondaires
def show_about_page():
    """Affiche la page À propos"""
    st.markdown("""
        <div class="header">
            <h1>🏢 À propos de NADA</h1>
            <p>Next-gen Automation for Data and Augmented Reality</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Retour à l'accueil"):
        st.session_state.current_page = "home"
        st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Notre Mission
        
        NADA est une plateforme révolutionnaire qui combine l'automatisation de pointe, 
        l'intelligence artificielle et la réalité augmentée pour créer la prochaine 
        génération de solutions technologiques.
        
        Nous repoussons les limites de l'innovation en intégrant:
        - 🧠 Intelligence Artificielle Avancée
        - ⚛️ Physique Quantique
        - 🔬 Biotechnologies
        - 🚀 Technologies Spatiales
        - 🤖 Robotique Autonome
        """)
    
    with col2:
        st.markdown("""
        ### 🌟 Notre Vision
        
        Créer un écosystème technologique unifié qui permet aux chercheurs, 
        développeurs et innovateurs de repousser les frontières de la science 
        et de la technologie.
        
        **47 plateformes spécialisées** couvrant:
        - Intelligence Artificielle & Machine Learning
        - Physique Quantique & Particules
        - Biotechnologie & Neurosciences
        - Énergie & Environnement
        - Cybersécurité & Optimisation
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques de la Plateforme")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Plateformes", "47", "En ligne")
    with col2:
        st.metric("Technologies", "150+", "Intégrées")
    with col3:
        st.metric("Chercheurs", "10,000+", "Actifs")
    with col4:
        st.metric("Projets", "5,000+", "En cours")


def show_documentation_page():
    """Affiche la page Documentation"""
    st.markdown("""
        <div class="header">
            <h1>📚 Documentation NADA</h1>
            <p>Guide complet de la plateforme</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Retour à l'accueil"):
        st.session_state.current_page = "home"
        st.rerun()
    
    st.markdown("---")
    
    doc_section = st.selectbox(
        "Sélectionnez une section",
        [
            "🚀 Guide de démarrage rapide",
            "🔧 Configuration",
            "🏗️ Architecture",
            "🔌 API & Intégrations",
            "❓ FAQ"
        ]
    )
    
    if doc_section == "🚀 Guide de démarrage rapide":
        st.markdown("""
        ## Guide de démarrage rapide
        
        ### 1. Inscription et connexion
        - Créez votre compte sur la page d'accueil
        - Connectez-vous avec vos identifiants
        - Accédez au tableau de bord principal
        
        ### 2. Explorer les plateformes
        - Parcourez les 47 plateformes disponibles
        - Utilisez la barre de recherche pour trouver rapidement
        - Cliquez sur "Accéder" pour ouvrir une plateforme dans un nouvel onglet
        
        ### 3. Premiers pas
        Chaque plateforme possède sa propre interface Streamlit sur un port dédié.
        """)
    
    elif doc_section == "🔧 Configuration":
        st.markdown("""
        ## Configuration de l'environnement
        
        ### Installation
        ```bash
        pip install streamlit psycopg2-binary
        
        # Créer la base de données
        createdb nada_platform_db
        
        # Lancer l'application
        streamlit run app.py --server.port 8000
        ```
        """)
    
    elif doc_section == "🏗️ Architecture":
        st.markdown("""
        ## Architecture de la plateforme
        
        NADA utilise une architecture microservices avec:
        - **Frontend**: Streamlit pour chaque plateforme (ports 8001-8046)
        - **Application principale**: Port 8000
        - **Base de données**: PostgreSQL pour l'authentification
        """)


def show_profile_page():
    """Affiche la page Profil"""
    st.markdown("""
        <div class="header">
            <h1>👤 Mon Profil</h1>
            <p>Gérez vos informations personnelles</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Retour à l'accueil"):
        st.session_state.current_page = "home"
        st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: rgba(0, 212, 255, 0.1); 
                    border-radius: 15px; border: 2px solid rgba(0, 212, 255, 0.3);'>
            <div style='font-size: 5rem;'>👤</div>
            <h2 style='color: #00d4ff; margin-top: 1rem;'>{st.session_state.user_info['username']}</h2>
            <p style='color: white; opacity: 0.8;'>{st.session_state.user_info['email']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Statistiques")
        st.metric("Plateformes utilisées", "12", "+3 ce mois")
        st.metric("Projets actifs", "5", "+1")
    
    with col2:
        tab1, tab2 = st.tabs(["📝 Informations", "🔐 Sécurité"])
        
        with tab1:
            st.markdown("### Informations personnelles")
            with st.form("update_profile"):
                new_username = st.text_input("Nom d'utilisateur", value=st.session_state.user_info['username'])
                new_email = st.text_input("Email", value=st.session_state.user_info['email'])
                phone = st.text_input("Téléphone", "")
                
                if st.form_submit_button("💾 Enregistrer"):
                    st.success("✅ Profil mis à jour!")
        
        with tab2:
            st.markdown("### Changer le mot de passe")
            with st.form("change_password"):
                current_password = st.text_input("Mot de passe actuel", type="password")
                new_password = st.text_input("Nouveau mot de passe", type="password")
                confirm_password = st.text_input("Confirmer", type="password")
                
                if st.form_submit_button("🔒 Changer"):
                    if new_password == confirm_password:
                        st.success("✅ Mot de passe modifié!")
                    else:
                        st.error("❌ Les mots de passe ne correspondent pas")


def show_login_page():
    """Affiche la page de connexion/inscription"""
    st.markdown("""
        <div class="header">
            <h1>🚀 NADA</h1>
            <p>Next-gen Automation for Data and Augmented Reality</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Connexion", "📝 Inscription"])
        
        with tab1:
            st.subheader("Connexion")
            login_username = st.text_input("Nom d'utilisateur", key="login_username")
            login_password = st.text_input("Mot de passe", type="password", key="login_password")
            
            if st.button("Se connecter", use_container_width=True):
                if login_username and login_password:
                    user = login_user(login_username, login_password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_info = {
                            'id': user[0],
                            'username': user[1],
                            'email': user[2]
                        }
                        st.success("Connexion réussie!")
                        st.rerun()
                    else:
                        st.error("Nom d'utilisateur ou mot de passe incorrect")
                else:
                    st.warning("Veuillez remplir tous les champs")
        
        with tab2:
            st.subheader("Inscription")
            reg_username = st.text_input("Nom d'utilisateur", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Mot de passe", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirmer le mot de passe", type="password", key="reg_confirm")
            
            if st.button("S'inscrire", use_container_width=True):
                if reg_username and reg_email and reg_password and reg_confirm:
                    if reg_password == reg_confirm:
                        if register_user(reg_username, reg_email, reg_password):
                            st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
                        else:
                            st.error("Ce nom d'utilisateur ou email existe déjà")
                    else:
                        st.error("Les mots de passe ne correspondent pas")
                else:
                    st.warning("Veuillez remplir tous les champs")


def show_home_page():
    """Affiche la page d'accueil avec les plateformes"""
    
    # En-tête
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="header">
                <h1>🚀 NADA Platform</h1>
                <p>Next-gen Automation for Data and Augmented Reality</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Barre de navigation
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏠 Accueil", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col2:
        if st.button("🏢 À propos", use_container_width=True):
            st.session_state.current_page = "about"
            st.rerun()
    with col3:
        if st.button("📚 Documentation", use_container_width=True):
            st.session_state.current_page = "documentation"
            st.rerun()
    with col4:
        if st.button("👤 Profil", use_container_width=True):
            st.session_state.current_page = "profile"
            st.rerun()
    with col5:
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.session_state.current_page = "home"
            st.rerun()
    
    st.markdown("---")
    
    # Afficher la page correspondante
    if st.session_state.current_page == "about":
        show_about_page()
    elif st.session_state.current_page == "documentation":
        show_documentation_page()
    elif st.session_state.current_page == "profile":
        show_profile_page()
    else:
        # Page d'accueil - TOUS LES BOUTONS IDENTIQUES
        st.markdown(f"### Bienvenue, {st.session_state.user_info['username']}! 👋")
        st.markdown("Sélectionnez une plateforme pour commencer:")
        
        # Barre de recherche
        search = st.text_input("🔍 Rechercher une plateforme", "")
        
        # Filtrer les plateformes
        filtered_platforms = [p for p in PLATFORMS if search.lower() in p['name'].lower()]
        
        # ✅ AFFICHAGE UNIFORME - TOUS PAREILS

        BASE_PORT = 8501

        # Attribuer un port unique à chaque plateforme
        for idx, platform in enumerate(filtered_platforms):
            platform['port'] = BASE_PORT + idx

        cols = st.columns(4)
        for idx, platform in enumerate(filtered_platforms):
            key = f"btn_{platform['slug']}"
            with cols[idx % 4]:
                st.markdown(f"""
                    <div class="platform-card">
                        <h3>{platform['icon']} {platform['name']}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Ici, on se sert du retour de st.button, pas du session_state
                clicked = st.button(f"🚀 Lancer {platform['name']}", key=key)

                if clicked:
                    folder = f"C:\\datasc\\{platform['slug']}"
                    app_file = os.path.join(folder, f"{platform['slug']}_app.py")
                    if os.path.exists(app_file):
                        st.toast(f"🧠 Démarrage de {platform['name']} sur le port {platform['port']}...")
                        subprocess.Popen(["streamlit", "run", app_file, "--server.port", str(platform['port'])])
                        webbrowser.open_new_tab(f"http://localhost:{platform['port']}")
                    else:
                        st.error(f"❌ Fichier Streamlit introuvable pour {platform['name']}")


# Affichage conditionnel
if st.session_state.logged_in:
    show_home_page()
else:
    show_login_page()
