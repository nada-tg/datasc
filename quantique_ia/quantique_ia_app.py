"""
quantum_engine_frontend.py - Interface Streamlit pour Quantum AI Engine

Installation:
pip install streamlit requests plotly pandas numpy

Lancement:
streamlit run quantique_ia_app.py
"""

import sys
import streamlit as st
import requests
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Configuration
API_BASE_URL = "http://localhost:8007"

# Configuration de la page
st.set_page_config(
    page_title="Quantum AI Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé Futuriste
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(255, 0, 110, 0.8)); }
    }
    
    .quantum-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .quantum-card-alt {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .quantum-card-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .success-quantum {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        border-left: 5px solid #2ecc71;
        margin: 1rem 0;
    }
    
    .warning-quantum {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        border-left: 5px solid #e67e22;
        margin: 1rem 0;
    }
    
    .info-quantum {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    
    .metric-quantum {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
    }
    
    .quantum-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions API
def create_project(data):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/project/create", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_project(project_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/project/{project_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def process_quantum_data(data):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/data/process", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_quantum_data(data_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/data/{data_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def train_model(data):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/model/train", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_model(model_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/{model_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_simulation(data):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/simulation/run", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_simulation(simulation_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/simulation/{simulation_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def create_quantum_computer(data):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/quantum-computer/create", json=data)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_quantum_computer(computer_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/quantum-computer/{computer_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def activate_quantum_computer(computer_id):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/quantum-computer/{computer_id}/activate")
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_on_quantum_computer(computer_id, num_qubits, algorithm, shots):
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/quantum-computer/{computer_id}/execute",
            params={"num_qubits": num_qubits, "algorithm": algorithm, "shots": shots}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_stats():
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats/overview")
        return response.json()
    except Exception as e:
        return {}

# Interface principale
def main():
    # Header
    st.markdown('<div class="main-header">⚛️ QUANTUM AI ENGINE</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Quantum+Engine", use_container_width=True)
        st.title("🌌 Navigation")
        
        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "🚀 Nouveau Projet",
                "📊 Données Quantiques",
                "🧠 Modèles IA",
                "🔬 Simulations",
                "💻 Ordinateurs Quantiques",
                "📈 Statistiques"
            ]
        )
        
        st.markdown("---")
        st.markdown("### ⚡ API Status")
        
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.success("✅ Connected")
            st.metric("Projets", health.get('projects', 0))
            st.metric("QC Virtuels", health.get('quantum_computers', 0))
            st.metric("Modèles", health.get('models', 0))
        except:
            st.error("❌ API Offline")
        
        st.markdown("---")
        st.info("💡 **Astuce**: Explorez toutes les possibilités du quantique!")
    
    # Pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🚀 Nouveau Projet":
        show_new_project()
    elif page == "📊 Données Quantiques":
        show_quantum_data()
    elif page == "🧠 Modèles IA":
        show_ai_models()
    elif page == "🔬 Simulations":
        show_simulations()
    elif page == "💻 Ordinateurs Quantiques":
        show_quantum_computers()
    elif page == "📈 Statistiques":
        show_statistics()

def show_dashboard():
    """Dashboard principal"""
    st.header("🌌 Tableau de Bord Quantique")
    
    # Statistiques
    stats = get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="quantum-card">
            <h2 style="text-align: center;">🚀</h2>
            <h3 style="text-align: center;">Projets</h3>
            <h1 style="text-align: center;">{}</h1>
        </div>
        """.format(stats.get("projects", {}).get("total", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="quantum-card-alt">
            <h2 style="text-align: center;">📊</h2>
            <h3 style="text-align: center;">Datasets</h3>
            <h1 style="text-align: center;">{}</h1>
        </div>
        """.format(stats.get("quantum_data", {}).get("total_datasets", 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="quantum-card-info">
            <h2 style="text-align: center;">🧠</h2>
            <h3 style="text-align: center;">Modèles IA</h3>
            <h1 style="text-align: center;">{}</h1>
        </div>
        """.format(stats.get("models", {}).get("total", 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="quantum-card">
            <h2 style="text-align: center;">💻</h2>
            <h3 style="text-align: center;">QC Virtuels</h3>
            <h1 style="text-align: center;">{}</h1>
        </div>
        """.format(stats.get("quantum_computers", {}).get("total", 0)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Capacités de la plateforme
    st.subheader("⚡ Capacités de la Plateforme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-quantum">
            <h3>🎯 Développement Quantique</h3>
            <ul>
                <li>✓ Logiciels & Applications</li>
                <li>✓ Sites Web & Plateformes Cloud</li>
                <li>✓ Applications Mobiles</li>
                <li>✓ Jeux Vidéo</li>
                <li>✓ IoT & Systèmes Embarqués</li>
                <li>✓ Agents IA & Plateformes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-quantum">
            <h3>📊 Traitement de Données</h3>
            <ul>
                <li>✓ Encodage Quantique</li>
                <li>✓ Data Science Quantique</li>
                <li>✓ Analyse Avancée</li>
                <li>✓ Stockage Optimisé</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-quantum">
            <h3>🔬 Simulation & Test</h3>
            <ul>
                <li>✓ Simulation d'Ordinateurs Quantiques</li>
                <li>✓ Tests de Performance</li>
                <li>✓ Évaluation de Scalabilité</li>
                <li>✓ Correction d'Erreurs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-quantum">
            <h3>💻 Ordinateurs Virtuels</h3>
            <ul>
                <li>✓ Création de QC Virtuels</li>
                <li>✓ Activation sur Machine Binaire</li>
                <li>✓ Performances Quantiques</li>
                <li>✓ Jusqu'à 128 Qubits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Actions Rapides")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Créer un Projet", use_container_width=True):
            st.session_state['active_page'] = "🚀 Nouveau Projet"
            st.rerun()
    
    with col2:
        if st.button("🧠 Entraîner un Modèle", use_container_width=True):
            st.session_state['active_page'] = "🧠 Modèles IA"
            st.rerun()
    
    with col3:
        if st.button("💻 Créer un QC", use_container_width=True):
            st.session_state['active_page'] = "💻 Ordinateurs Quantiques"
            st.rerun()

def show_new_project():
    """Page de création de projet"""
    st.header("🚀 Nouveau Projet Quantique")
    
    st.markdown('<div class="info-quantum">Développez n\'importe quel produit informatique avec la puissance du quantique</div>', unsafe_allow_html=True)
    
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("Nom du Projet*", placeholder="Mon Projet Quantique")
            
            product_type = st.selectbox(
                "Type de Produit*",
                [
                    "software", "website", "mobile_app", "cloud_platform",
                    "video_game", "iot_device", "embedded_system",
                    "ai_model", "ai_agent", "ai_agent_platform"
                ],
                format_func=lambda x: {
                    "software": "💻 Logiciel",
                    "website": "🌐 Site Web",
                    "mobile_app": "📱 Application Mobile",
                    "cloud_platform": "☁️ Plateforme Cloud",
                    "video_game": "🎮 Jeu Vidéo",
                    "iot_device": "🔌 Objet Connecté",
                    "embedded_system": "⚙️ Système Embarqué",
                    "ai_model": "🧠 Modèle d'IA",
                    "ai_agent": "🤖 Agent IA",
                    "ai_agent_platform": "🏢 Plateforme d'Agents"
                }[x]
            )
            
            description = st.text_area(
                "Description*",
                placeholder="Décrivez votre projet...",
                height=150
            )
        
        with col2:
            st.markdown("**Configuration Quantique**")
            
            target_qubits = st.slider(
                "Nombre de Qubits",
                min_value=2,
                max_value=100,
                value=8,
                help="Plus de qubits = plus de puissance quantique"
            )
            
            use_optimization = st.checkbox("Utiliser l'optimisation quantique", value=True)
            
            st.markdown("**Fonctionnalités Quantiques**")
            
            quantum_features = st.multiselect(
                "Sélectionnez les features",
                [
                    "Calcul Quantique Parallèle",
                    "Optimisation Quantique",
                    "Machine Learning Quantique",
                    "Cryptographie Quantique",
                    "Recherche Quantique (Grover)",
                    "Factorisation (Shor)",
                    "Simulation Quantique",
                    "Intrication Quantique"
                ],
                default=["Calcul Quantique Parallèle", "Optimisation Quantique"]
            )
            
            st.info(f"💡 Puissance estimée: **{2**(target_qubits/2):.0f}x** vs classique")
        
        submitted = st.form_submit_button("🚀 Créer le Projet", use_container_width=True)
        
        if submitted:
            if not project_name or not description:
                st.error("⚠️ Veuillez remplir tous les champs obligatoires")
            else:
                with st.spinner("🔄 Création du projet quantique..."):
                    result = create_project({
                        "product_type": product_type,
                        "project_name": project_name,
                        "description": description,
                        "quantum_features": quantum_features,
                        "target_qubits": target_qubits,
                        "use_quantum_optimization": use_optimization
                    })
                    
                    if result.get("success"):
                        project = result["project"]
                        st.markdown(f'<div class="success-quantum">✅ Projet créé avec succès!<br>ID: <b>{project["project_id"]}</b></div>', unsafe_allow_html=True)
                        
                        st.session_state['last_project_id'] = project["project_id"]
                        
                        # Afficher les étapes
                        st.subheader("📋 Étapes de Développement")
                        
                        phases = project.get("phases", [])
                        for phase in phases:
                            status_icon = "✅" if phase["status"] == "completed" else "🔄" if phase["status"] == "in_progress" else "⏳"
                            
                            with st.expander(f"{status_icon} {phase['phase']} - {phase['duration_days']} jours"):
                                st.write("**Tâches:**")
                                for task in phase["tasks"]:
                                    st.write(f"- {task}")
                        
                        st.balloons()
                    else:
                        st.error(f"❌ Erreur: {result.get('error')}")

def show_quantum_data():
    """Page de traitement des données quantiques"""
    st.header("📊 Données Quantiques")
    
    tab1, tab2 = st.tabs(["📤 Nouveau Dataset", "📥 Datasets Existants"])
    
    with tab1:
        st.markdown('<div class="info-quantum">Traitez vos données avec des algorithmes quantiques</div>', unsafe_allow_html=True)
        
        with st.form("data_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                data_name = st.text_input("Nom du Dataset*", placeholder="Mon Dataset Quantique")
                
                data_type = st.selectbox(
                    "Type de Données*",
                    ["structured", "unstructured", "time_series", "image", "text"],
                    format_func=lambda x: {
                        "structured": "📊 Structurées",
                        "unstructured": "📝 Non-structurées",
                        "time_series": "📈 Séries Temporelles",
                        "image": "🖼️ Images",
                        "text": "📄 Texte"
                    }[x]
                )
                
                size_mb = st.number_input("Taille (MB)", min_value=0.1, max_value=10000.0, value=100.0, step=10.0)
            
            with col2:
                quantum_encoding = st.selectbox(
                    "Encodage Quantique*",
                    ["amplitude", "basis", "angle", "iqp"],
                    format_func=lambda x: {
                        "amplitude": "🌊 Amplitude Encoding",
                        "basis": "📐 Basis Encoding",
                        "angle": "📐 Angle Encoding",
                        "iqp": "🔀 IQP Encoding"
                    }[x]
                )
                
                st.markdown("**Pipeline de Traitement**")
                st.info("""
                1. 📥 Collection
                2. 🧹 Nettoyage
                3. 🔄 Transformation
                4. ⚛️ Encodage Quantique
                5. 📊 Analyse
                6. 💾 Stockage
                """)
            
            submitted = st.form_submit_button("⚡ Lancer le Traitement", use_container_width=True)
            
            if submitted:
                if not data_name:
                    st.error("⚠️ Veuillez entrer un nom pour le dataset")
                else:
                    with st.spinner("🔄 Traitement des données quantiques en cours..."):
                        result = process_quantum_data({
                            "data_name": data_name,
                            "data_type": data_type,
                            "quantum_encoding": quantum_encoding,
                            "size_mb": size_mb
                        })
                        
                        if result.get("success"):
                            st.markdown(f'<div class="success-quantum">✅ Traitement lancé!<br>Data ID: <b>{result["data_id"]}</b></div>', unsafe_allow_html=True)
                            st.session_state['last_data_id'] = result['data_id']
                            
                            # Simulation de progression
                            progress_bar = st.progress(0)
                            status = st.empty()
                            
                            stages = ["📥 Collection", "🧹 Nettoyage", "🔄 Transformation", 
                                     "⚛️ Encodage", "📊 Analyse", "💾 Stockage"]
                            
                            for i, stage in enumerate(stages):
                                status.text(f"{stage}...")
                                for j in range(17):
                                    progress_value = (i * 17 + j) / 100
                                    progress_bar.progress(min(progress_value, 1.0))
                                    # progress_bar.progress((i * 17 + j) / 100)
                                    time.sleep(0.05)
                            
                            st.success("✨ Données traitées avec succès!")
                            st.balloons()
                        else:
                            st.error(f"❌ Erreur: {result.get('error')}")
    

    with tab2:
        data_id = st.text_input("ID du Dataset", value=st.session_state.get('last_data_id', ''))

        if st.button("🔍 Charger le Dataset", use_container_width=True):
            if data_id:
                data = get_quantum_data(data_id)

                if "error" not in data:
                    # 🧩 Sécurisation des métadonnées de base
                    data_name = data.get("data_name", "Dataset inconnu")
                    quantum_encoding = data.get("quantum_encoding", "Non spécifié")
                    data_type = data.get("data_type", "Inconnu")

                    # 🧮 Calcul automatique de la taille du dataset si absente
                    original_size_mb = data.get("original_size_mb")
                    if original_size_mb is None:
                        try:
                            # Si un DataFrame est présent
                            if isinstance(data.get("df"), pd.DataFrame):
                                original_size_mb = (
                                    data["df"].memory_usage(deep=True).sum() / (1024 * 1024)
                                )
                            # Si les features sont dans data['features']
                            elif "features" in data:
                                obj = data["features"]
                                if isinstance(obj, pd.DataFrame):
                                    original_size_mb = obj.memory_usage(deep=True).sum() / (1024 * 1024)
                                elif isinstance(obj, (list, np.ndarray)):
                                    original_size_mb = sys.getsizeof(obj) / (1024 * 1024)
                                else:
                                    original_size_mb = sys.getsizeof(data) / (1024 * 1024)
                            else:
                                original_size_mb = sys.getsizeof(data) / (1024 * 1024)
                        except Exception as e:
                            st.warning(f"Impossible de calculer la taille du dataset : {e}")
                            original_size_mb = 0.0

                    # ✅ Affichage résumé du dataset
                    st.success(f"✅ Dataset chargé: **{data_name}**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Taille originale", f"{original_size_mb:.2f} MB")
                    with col2:
                        st.metric("Encodage", quantum_encoding)
                    with col3:
                        st.metric("Type", data_type)

                    # 🔹 Pipeline de traitement
                    st.subheader("📊 Pipeline de Traitement")
                    stages = data.get("stages", [])
                    if stages:
                        for stage in stages:
                            stage_name = stage.get("stage", "Étape inconnue")
                            timestamp = stage.get("timestamp", "N/A")
                            with st.expander(f"✅ {stage_name} - {timestamp[:19]}"):
                                st.json(stage)
                    else:
                        st.info("Aucune étape de pipeline trouvée.")

                    # 🔹 Métadonnées
                    metadata = data.get("metadata", {})
                    if metadata:
                        st.subheader("📋 Métadonnées")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Échantillons", metadata.get("num_samples", "N/A"))
                        with col2:
                            st.metric("Features", metadata.get("num_features", "N/A"))
                        with col3:
                            st.metric("Dimension Quantique", metadata.get("quantum_state_dimension", "N/A"))
                    else:
                        st.info("Aucune métadonnée disponible.")
                else:
                    st.error(f"❌ {data.get('error', 'Erreur inconnue lors du chargement du dataset.')}")
    # with tab2:
    #     data_id = st.text_input("ID du Dataset", value=st.session_state.get('last_data_id', ''))
        
    #     if st.button("🔍 Charger le Dataset", use_container_width=True):
    #         if data_id:
    #             data = get_quantum_data(data_id)
                
    #             if "error" not in data:
    #                 st.success(f"✅ Dataset chargé: **{data['data_name']}**")
                    
    #                 col1, col2, col3 = st.columns(3)
    #                 with col1:
    #                     st.metric("Taille originale", f"{data['original_size_mb']:.1f} MB")
    #                 with col2:
    #                     st.metric("Encodage", data['quantum_encoding'])
    #                 with col3:
    #                     st.metric("Type", data['data_type'])
                    
    #                 # Pipeline stages
    #                 st.subheader("📊 Pipeline de Traitement")
    #                 for stage in data.get('stages', []):
    #                     with st.expander(f"✅ {stage['stage']} - {stage['timestamp'][:19]}"):
    #                         st.json(stage)
                    
    #                 # Metadata
    #                 if 'metadata' in data:
    #                     st.subheader("📋 Métadonnées")
    #                     col1, col2, col3 = st.columns(3)
    #                     with col1:
    #                         st.metric("Échantillons", data['metadata']['num_samples'])
    #                     with col2:
    #                         st.metric("Features", data['metadata']['num_features'])
    #                     with col3:
    #                         st.metric("Dimension Quantique", data['metadata']['quantum_state_dimension'])
    #             else:
    #                 st.error(f"❌ {data['error']}")

def show_ai_models():
    """Page des modèles d'IA quantiques"""
    st.header("🧠 Modèles d'IA Quantiques")
    
    tab1, tab2 = st.tabs(["🚀 Entraîner un Modèle", "📊 Modèles Existants"])
    
    with tab1:
        st.markdown('<div class="info-quantum">Entraînez des modèles d\'IA avec des algorithmes quantiques</div>', unsafe_allow_html=True)
        
        with st.form("model_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Nom du Modèle*", placeholder="Mon Modèle Quantique")
                
                data_source_id = st.text_input("ID du Dataset Source*", 
                    value=st.session_state.get('last_data_id', ''),
                    placeholder="data-id-123")
                
                algorithm = st.selectbox(
                    "Algorithme Quantique*",
                    ["grover", "shor", "vqe", "qaoa", "quantum_ml", "qgan"],
                    format_func=lambda x: {
                        "grover": "🔍 Grover (Recherche)",
                        "shor": "🔢 Shor (Factorisation)",
                        "vqe": "⚡ VQE (Variational)",
                        "qaoa": "🎯 QAOA (Optimisation)",
                        "quantum_ml": "🧠 Quantum ML",
                        "qgan": "🎨 QGAN (Génératif)"
                    }[x]
                )
            
            with col2:
                num_qubits = st.slider("Nombre de Qubits", 2, 50, 8)
                quantum_layers = st.slider("Couches Quantiques", 1, 20, 3)
                epochs = st.slider("Époques", 1, 1000, 100)
                
                st.info(f"""
                **Configuration:**
                - Puissance: **{2**(num_qubits/2):.0f}x** classique
                - Paramètres: ~**{quantum_layers * num_qubits * 3}** gates
                - Temps estimé: **{epochs * 0.05:.1f}s**
                """)
            
            submitted = st.form_submit_button("🚀 Lancer l'Entraînement", use_container_width=True)
            
            if submitted:
                if not model_name or not data_source_id:
                    st.error("⚠️ Veuillez remplir tous les champs obligatoires")
                else:
                    with st.spinner("🔄 Entraînement du modèle quantique..."):
                        result = train_model({
                            "model_name": model_name,
                            "data_source_id": data_source_id,
                            "algorithm": algorithm,
                            "num_qubits": num_qubits,
                            "epochs": epochs,
                            "quantum_layers": quantum_layers
                        })
                        
                        if result.get("success"):
                            st.markdown(f'<div class="success-quantum">✅ Entraînement lancé!<br>Model ID: <b>{result["model_id"]}</b></div>', unsafe_allow_html=True)
                            st.session_state['last_model_id'] = result['model_id']
                            
                            # Simulation d'entraînement
                            progress_bar = st.progress(0)
                            metrics_placeholder = st.empty()
                            
                            for i in range(epochs):
                                progress = (i + 1) / epochs
                                progress_bar.progress(progress)
                                
                                if i % max(1, epochs // 10) == 0:
                                    loss = 1.0 * np.exp(-i / epochs * 3)
                                    accuracy = 1.0 - loss
                                    metrics_placeholder.metric("Accuracy", f"{accuracy:.2%}", f"+{(accuracy-0.5)*100:.1f}%")
                                
                                time.sleep(0.01)
                            
                            st.success("✨ Modèle entraîné avec succès!")
                            st.balloons()
                        else:
                            st.error(f"❌ Erreur: {result.get('error')}")
    
    with tab2:
        model_id = st.text_input("ID du Modèle", value=st.session_state.get('last_model_id', ''))
        
        if st.button("🔍 Charger le Modèle", use_container_width=True):
            if model_id:
                model = get_model(model_id)
                
                if "error" not in model:
                    # st.success(f"✅ Modèle chargé: **{model['model_name']}**")
                    if "model_name" not in model:
                        model["model_name"] = f"quantum_model_{model_id[:6]}"

                    # Métriques finales
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = model.get('final_metrics', {})
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                    with col2:
                        st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
                    with col3:
                        st.metric("Fidelity", f"{metrics.get('quantum_fidelity', 0):.2%}")
                    with col4:
                        st.metric("Avantage Quantique", f"{metrics.get('quantum_advantage', 1):.2f}x")
                    
                    # Historique d'entraînement
                    if 'training_history' in model:
                        st.subheader("📈 Historique d'Entraînement")
                        
                        df = pd.DataFrame(model['training_history'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'], name='Accuracy', line=dict(color='#00d4ff', width=3)))
                        fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], name='Loss', line=dict(color='#ff006e', width=3)))
                        fig.add_trace(go.Scatter(x=df['epoch'], y=df['quantum_fidelity'], name='Fidelity', line=dict(color='#7b2cbf', width=3)))
                        
                        fig.update_layout(
                            title="Métriques d'Entraînement",
                            xaxis_title="Époque",
                            yaxis_title="Valeur",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Circuit quantique
                    if 'quantum_circuit' in model:
                        st.subheader("⚛️ Circuit Quantique")
                        circuit = model['quantum_circuit']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Qubits", circuit['num_qubits'])
                        with col2:
                            st.metric("Profondeur", circuit['depth'])
                        with col3:
                            st.metric("Intrication", circuit['entanglement'])
                else:
                    st.error(f"❌ {model['error']}")

def show_simulations():
    """Page des simulations quantiques"""
    st.header("🔬 Simulations Quantiques")
    
    tab1, tab2 = st.tabs(["🚀 Nouvelle Simulation", "📊 Résultats"])
    
    with tab1:
        st.markdown('<div class="info-quantum">Simulez le fonctionnement d\'ordinateurs quantiques</div>', unsafe_allow_html=True)
        
        with st.form("simulation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                project_id = st.text_input("ID du Projet*",
                    value=st.session_state.get('last_project_id', ''),
                    placeholder="project-id-123")
                
                simulation_type = st.selectbox(
                    "Type de Simulation*",
                    ["performance", "scalability", "error_correction", "full"],
                    format_func=lambda x: {
                        "performance": "⚡ Performance",
                        "scalability": "📈 Scalabilité",
                        "error_correction": "🛡️ Correction d'Erreurs",
                        "full": "🌟 Complète"
                    }[x]
                )
            
            with col2:
                num_qubits = st.slider("Nombre de Qubits", 2, 100, 16)
                shots = st.slider("Nombre de Shots", 100, 100000, 1000, step=100)
                
                st.info(f"""
                **Estimation:**
                - États possibles: **{2**num_qubits}**
                - Temps: ~**{num_qubits * 0.1 * shots / 1000:.1f}s**
                - Mémoire: **{num_qubits * 0.5:.1f} GB**
                """)
            
            submitted = st.form_submit_button("🔬 Lancer la Simulation", use_container_width=True)
            
            if submitted:
                if not project_id:
                    st.error("⚠️ Veuillez entrer un ID de projet")
                else:
                    with st.spinner("🔄 Simulation en cours..."):
                        result = run_simulation({
                            "project_id": project_id,
                            "simulation_type": simulation_type,
                            "num_qubits": num_qubits,
                            "shots": shots
                        })
                        
                        if result.get("success"):
                            st.markdown(f'<div class="success-quantum">✅ Simulation lancée!<br>Simulation ID: <b>{result["simulation_id"]}</b></div>', unsafe_allow_html=True)
                            st.session_state['last_simulation_id'] = result['simulation_id']
                            
                            # Animation de simulation
                            progress_bar = st.progress(0)
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                time.sleep(0.02)
                            
                            st.success("✨ Simulation terminée!")
                        else:
                            st.error(f"❌ Erreur: {result.get('error')}")
    
    with tab2:
        simulation_id = st.text_input("ID de la Simulation", value=st.session_state.get('last_simulation_id', ''))
        
        if st.button("🔍 Charger les Résultats", use_container_width=True):
            if simulation_id:
                sim = get_simulation(simulation_id)
                
                if "error" not in sim:
                    # st.success(f"✅ Simulation chargée - Type: **{sim['simulation_type']}**")
                    sim_type = sim.get('simulation_type', 'Type inconnu')
                    num_qubits = sim.get('num_qubits', 'N/A')
                    depth = sim.get('depth', 'N/A')
                    st.success(f"✅ Simulation chargée - Type: **{sim_type}**, Qubits: {num_qubits}, Profondeur: {depth}")

                    
                    # Résultats d'exécution
                    results = sim.get('results', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Shots Totaux", results.get('total_shots', 0))
                    with col2:
                        st.metric("Temps d'Exécution", f"{results.get('execution_time_ms', 0):.2f} ms")
                    with col3:
                        st.metric("Fidelity", f"{results.get('fidelity', 0):.2%}")
                    with col4:
                        st.metric("Probabilité de Succès", f"{results.get('success_probability', 0):.2%}")
                    
                    # Distribution des états
                    if 'counts' in results:
                        st.subheader("📊 Distribution des États Quantiques")
                        
                        counts = results['counts']
                        states = list(counts.keys())[:10]  # Top 10
                        values = [counts[s] for s in states]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=states, y=values, marker_color='#667eea')
                        ])
                        
                        fig.update_layout(
                            title="Top 10 États Mesurés",
                            xaxis_title="État Quantique",
                            yaxis_title="Occurrences",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse spécifique
                    st.subheader("📈 Analyse")
                    analysis = sim.get('analysis', {})
                    st.json(analysis)
                else:
                    st.error(f"❌ {sim['error']}")

def show_quantum_computers():
    """Page des ordinateurs quantiques virtuels"""
    st.header("💻 Ordinateurs Quantiques Virtuels")
    
    tab1, tab2, tab3 = st.tabs(["🆕 Créer un QC", "💻 Mes QC", "⚡ Exécuter"])
    
    with tab1:
        st.markdown('<div class="info-quantum">Créez votre propre ordinateur quantique virtuel</div>', unsafe_allow_html=True)
        
        with st.form("qc_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                qc_name = st.text_input("Nom de l'Ordinateur*", placeholder="Mon QC Virtuel")
                
                num_qubits = st.slider("Nombre de Qubits", 2, 128, 16)
                
                topology = st.selectbox(
                    "Topologie*",
                    ["linear", "grid", "all_to_all", "custom"],
                    format_func=lambda x: {
                        "linear": "📏 Linéaire",
                        "grid": "🔲 Grille",
                        "all_to_all": "🕸️ Tous connectés",
                        "custom": "🎨 Personnalisée"
                    }[x]
                )
            
            with col2:
                error_rate = st.slider("Taux d'Erreur", 0.0, 1.0, 0.01, 0.001, format="%.3f")
                
                enable_noise = st.checkbox("Activer le Modèle de Bruit", value=True)
                
                st.info(f"""
                **Spécifications:**
                - Volume Quantique: **{2**min(num_qubits, 10)}**
                - Fidelity: **{(1-error_rate)*100:.1f}%**
                - Mémoire: **{num_qubits * 2} GB**
                - Puissance: **{2**(num_qubits/3):.0f}x** classique
                """)
            
            submitted = st.form_submit_button("🚀 Créer l'Ordinateur", use_container_width=True)
            
            if submitted:
                if not qc_name:
                    st.error("⚠️ Veuillez entrer un nom")
                else:
                    result = create_quantum_computer({
                        "name": qc_name,
                        "num_qubits": num_qubits,
                        "topology": topology,
                        "error_rate": error_rate,
                        "enable_noise_model": enable_noise
                    })
                    
                    if result.get("success"):
                        computer = result['computer']
                        st.markdown(f'<div class="success-quantum">✅ Ordinateur quantique créé!<br>ID: <b>{computer["computer_id"]}</b></div>', unsafe_allow_html=True)
                        st.session_state['last_qc_id'] = computer['computer_id']
                        st.balloons()
                    else:
                        st.error(f"❌ Erreur: {result.get('error')}")
    
    with tab2:
        qc_id = st.text_input("ID de l'Ordinateur", value=st.session_state.get('last_qc_id', ''))
        
        if st.button("🔍 Charger l'Ordinateur", use_container_width=True):
            if qc_id:
                qc = get_quantum_computer(qc_id)
                
                if "error" not in qc:
                    st.success(f"✅ Ordinateur chargé: **{qc['name']}**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Qubits", qc['num_qubits'])
                    with col2:
                        st.metric("Topologie", qc['topology'])
                    with col3:
                        st.metric("Status", qc['status'])
                    
                    # Spécifications
                    st.subheader("⚙️ Spécifications")
                    specs = qc.get('specs', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Volume Quantique", specs.get('quantum_volume', 0))
                        st.metric("Fidelity Gate", f"{specs.get('gate_fidelity', 0):.2%}")
                        st.metric("Profondeur Max", specs.get('max_circuit_depth', 0))
                    with col2:
                        st.metric("Fidelity Mesure", f"{specs.get('measurement_fidelity', 0):.2%}")
                        st.metric("Mémoire", f"{specs.get('classical_memory_gb', 0)} GB")
                        st.metric("Shots/sec", specs.get('shots_per_second', 0))
                    
                    # Activation
                    st.markdown("---")
                    if st.button("⚡ Activer sur Machine Binaire", use_container_width=True):
                        with st.spinner("🔄 Activation en cours..."):
                            result = activate_quantum_computer(qc_id)
                            
                            if result.get("success"):
                                st.success("✅ Ordinateur quantique activé!")
                                
                                # Simulation des étapes
                                for step in ["Initialisation", "Calibration", "Activation"]:
                                    st.info(f"⚙️ {step}...")
                                    time.sleep(0.5)
                                
                                st.balloons()
                            else:
                                st.error(f"❌ Erreur: {result.get('error')}")
                else:
                    st.error(f"❌ {qc['error']}")
    
    with tab3:
        st.markdown('<div class="info-quantum">Exécutez des circuits sur votre ordinateur quantique</div>', unsafe_allow_html=True)
        
        qc_id_exec = st.text_input("ID de l'Ordinateur", value=st.session_state.get('last_qc_id', ''), key="exec_qc")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            exec_qubits = st.slider("Qubits à utiliser", 2, 20, 4)
        with col2:
            exec_algorithm = st.selectbox("Algorithme", ["custom", "grover", "shor", "vqe"])
        with col3:
            exec_shots = st.slider("Shots", 100, 10000, 1000)
        
        if st.button("⚡ Exécuter", use_container_width=True):
            if qc_id_exec:
                with st.spinner("🔄 Exécution en cours..."):
                    result = execute_on_quantum_computer(qc_id_exec, exec_qubits, exec_algorithm, exec_shots)
                    
                    if result.get("success"):
                        st.success("✅ Exécution terminée!")
                        
                        # Résultats
                        results = result['results']
                        advantage = result['quantum_advantage']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temps", f"{results['execution_time_ms']:.2f} ms")
                        with col2:
                            st.metric("Fidelity", f"{results['fidelity']:.2%}")
                        with col3:
                            st.metric("Speedup", f"{advantage['practical_speedup']:.1f}x")
                        
                        st.json(results)
                    else:
                        st.error(f"❌ {result.get('error')}")

def show_statistics():
    """Page des statistiques globales"""
    st.header("📈 Statistiques de la Plateforme")
    
    stats = get_stats()
    
    if stats:
        # Vue d'ensemble
        st.subheader("📊 Vue d'Ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Projets Totaux", stats.get("projects", {}).get("total", 0))
            st.metric("Projets Actifs", stats.get("projects", {}).get("active", 0))
        
        with col2:
            st.metric("Datasets", stats.get("quantum_data", {}).get("total_datasets", 0))
            st.metric("Taille Totale", f"{stats.get('quantum_data', {}).get('total_size_gb', 0):.1f} GB")
        
        with col3:
            st.metric("Modèles IA", stats.get("models", {}).get("total", 0))
            st.metric("Accuracy Moyenne", f"{stats.get('models', {}).get('average_accuracy', 0):.1%}")
        
        with col4:
            st.metric("Ordinateurs QC", stats.get("quantum_computers", {}).get("total", 0))
            st.metric("Qubits Totaux", stats.get("quantum_computers", {}).get("total_qubits", 0))
        
        st.markdown("---")
        
        # Graphiques
        st.subheader("📈 Visualisations")
        
        # Exemple de données pour les graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des types de projets
            fig = go.Figure(data=[go.Pie(
                labels=["Software", "Mobile", "Cloud", "AI", "IoT", "Autres"],
                values=[25, 20, 15, 30, 5, 5],
                marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
            )])
            fig.update_layout(title="Distribution des Projets", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance quantique vs classique
            categories = ['Performance', 'Efficacité', 'Scalabilité', 'Précision']
            quantum_scores = [85, 78, 92, 88]
            classical_scores = [45, 60, 55, 70]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=quantum_scores, theta=categories, fill='toself', name='Quantique', line_color='#00d4ff'))
            fig.add_trace(go.Scatterpolar(r=classical_scores, theta=categories, fill='toself', name='Classique', line_color='#ff006e'))
            fig.update_layout(title="Quantique vs Classique", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Impossible de charger les statistiques")

if __name__ == "__main__":
    # Initialisation session state
    if 'last_project_id' not in st.session_state:
        st.session_state['last_project_id'] = ''
    if 'last_data_id' not in st.session_state:
        st.session_state['last_data_id'] = ''
    if 'last_model_id' not in st.session_state:
        st.session_state['last_model_id'] = ''
    if 'last_simulation_id' not in st.session_state:
        st.session_state['last_simulation_id'] = ''
    if 'last_qc_id' not in st.session_state:
        st.session_state['last_qc_id'] = ''
    
    main()