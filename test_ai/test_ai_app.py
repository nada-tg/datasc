"""
frontend_complete.py - Interface Streamlit Complète pour le Benchmark de Modèles IA
Lancez avec: streamlit run test_ai_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Model Benchmark Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8009"

# Styles CSS  streamlit run frontend_complete.py
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GESTION DE L'AUTHENTIFICATION
# ============================================================

def init_session_state():
    """Initialise l'état de la session"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def get_auth_headers():
    """Retourne les headers d'authentification"""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def login(username: str, password: str):
    """Connexion à l'API"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/auth/login",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.authenticated = True
            st.session_state.token = data["access_token"]
            st.session_state.username = username
            return True, "Connexion réussie!"
        else:
            return False, "Identifiants incorrects"
    except Exception as e:
        return False, f"Erreur de connexion: {str(e)}"

def register(username: str, email: str, password: str, full_name: str = ""):
    """Enregistrement d'un nouvel utilisateur"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name
            }
        )
        if response.status_code == 200:
            return True, "Compte créé avec succès!"
        else:
            return False, response.json().get("detail", "Erreur lors de l'enregistrement")
    except Exception as e:
        return False, f"Erreur: {str(e)}"

def logout():
    """Déconnexion"""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def check_api_status():
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def make_api_request(method, endpoint, **kwargs):
    """Effectue une requête API avec gestion des erreurs"""
    try:
        headers = get_auth_headers()
        if 'headers' in kwargs:
            kwargs['headers'].update(headers)
        else:
            kwargs['headers'] = headers
        
        url = f"{API_URL}{endpoint}"
        response = requests.request(method, url, **kwargs)
        
        if response.status_code == 401:
            st.error("Session expirée. Veuillez vous reconnecter.")
            logout()
            st.rerun()
        
        return response
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return None

# ============================================================
# PAGE DE CONNEXION
# ============================================================

def show_login_page():
    """Affiche la page de connexion"""
    st.markdown('<h1 class="main-header"> AI Model Benchmark Platform</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        tab1, tab2 = st.tabs([" Connexion", " Inscription"])
        
        with tab1:
            st.subheader("Connexion")
            username = st.text_input("Nom d'utilisateur", key="login_username")
            password = st.text_input("Mot de passe", type="password", key="login_password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Se connecter", type="primary", use_container_width=True):
                    if username and password:
                        success, message = login(username, password)
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Veuillez remplir tous les champs")
            
            with col_b:
                st.info("**Compte test:**\nadmin / admin123")
        
        with tab2:
            st.subheader("Créer un compte")
            new_username = st.text_input("Nom d'utilisateur", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_full_name = st.text_input("Nom complet (optionnel)", key="reg_fullname")
            new_password = st.text_input("Mot de passe", type="password", key="reg_password")
            new_password_confirm = st.text_input("Confirmer le mot de passe", type="password", key="reg_password_confirm")
            
            if st.button("Créer le compte", type="primary", use_container_width=True):
                if not all([new_username, new_email, new_password]):
                    st.warning("Veuillez remplir tous les champs obligatoires")
                elif new_password != new_password_confirm:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_password) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères")
                else:
                    success, message = register(new_username, new_email, new_password, new_full_name)
                    if success:
                        st.success(message)
                        st.info("Vous pouvez maintenant vous connecter!")
                    else:
                        st.error(message)

# ============================================================
# PAGES PRINCIPALES
# ============================================================

def show_home_page():
    """Page d'accueil"""
    st.markdown('<h1 class="main-header"> AI Model Benchmark Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>Pour</h2>
            <h3>Upload</h3>
            <p>Uploadez vos modèles d'IA facilement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>Pour</h2>
            <h3>Test</h3>
            <p>Évaluez les performances complètes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Pour</h2>
            <h3>Analyse</h3>
            <p>Visualisez et comparez les résultats</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header(" Bienvenue!")
    
    st.markdown("""
    ### Cette plateforme vous permet de :
    
    - **Tester rigoureusement** vos modèles d'IA sur 9 aspects différents
    - **Comparer** les performances entre plusieurs modèles
    - **Analyser** en profondeur avec des visualisations avancées
    - **Suivre** l'évolution dans le temps avec MLflow
    - **Identifier** les forces et faiblesses de chaque modèle
    
    ###  Types de tests disponibles
    
    | Test | Description |
    |------|-------------|
    | **Reasoning** | Raisonnement logique et déduction |
    | **Language** | Compréhension et génération de texte |
    | **Math** | Capacités mathématiques |
    | **Speed** | Performance et latence |
    | **Creative** | Génération créative |
    | **Memory** | Gestion du contexte long |
    | **Logic** | Logique formelle |
    | **Comprehension** | Compréhension avancée |
    | **Coding** | Génération de code |
    """)
    
    st.success(" Commencez par uploader un modèle dans le menu de gauche!")

def show_upload_page():
    """Page d'upload de modèle"""
    st.header(" Upload de Modèle d'IA")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Informations du Modèle")
        
        model_name = st.text_input("Nom du modèle*", placeholder="Mon Super Modèle v1.0")
        
        model_type = st.selectbox(
            "Type de modèle*",
            ["NLP", "Vision", "Audio", "Multimodal", "Agent", "Code Generator", "Autre"]
        )
        
        framework = st.selectbox(
            "Framework utilisé*",
            ["PyTorch", "TensorFlow", "Transformers", "JAX", "ONNX", "Custom", "Autre"]
        )
        
        description = st.text_area(
            "Description",
            placeholder="Décrivez votre modèle, son architecture, ses spécificités..."
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            num_parameters = st.text_input("Nombre de paramètres", placeholder="ex: 7B, 13B")
        with col_b:
            architecture = st.text_input("Architecture", placeholder="ex: Transformer, CNN")
        
        st.subheader("Upload du fichier")
        uploaded_file = st.file_uploader(
            "Choisir le fichier du modèle",
            type=['pkl', 'pt', 'pth', 'h5', 'onnx', 'pb', 'safetensors'],
            help="Formats supportés: .pkl, .pt, .pth, .h5, .onnx, .pb, .safetensors"
        )
        
        if st.button(" Upload Modèle", type="primary"):
            if uploaded_file and model_name:
                with st.spinner("Upload en cours..."):
                    model_info = {
                        "name": model_name,
                        "type": model_type.lower(),
                        "framework": framework.lower(),
                        "description": description,
                        "parameters": num_parameters,
                        "architecture": architecture
                    }
                    
                    files = {"file": uploaded_file}
                    data = {"model_info": json.dumps(model_info)}
                    
                    response = make_api_request(
                        "POST",
                        "/api/v1/models/upload",
                        files=files,
                        data=data
                    )
                    
                    if response and response.status_code == 200:
                        result = response.json()
                        st.success(" Modèle uploadé avec succès!")
                        st.json(result)
                        st.balloons()
                    else:
                        st.error(" Erreur lors de l'upload")
            else:
                st.warning(" Veuillez remplir tous les champs obligatoires")
    
    with col2:
        st.info("""
        ###  Instructions
        
        1. Remplissez les informations
        2. Uploadez le fichier
        3. Cliquez sur "Upload Modèle"
        4. Votre modèle sera prêt pour les tests
        
        ###  Conseils
        
        - Nom descriptif et unique
        - Type de modèle précis
        - Description détaillée
        - Format de fichier valide
        """)

def show_tests_page():
    """Page de lancement des tests"""
    st.header(" Lancer des Tests de Performance")
    
    # Récupérer les modèles
    response = make_api_request("GET", "/api/v1/models")
    
    if not response or response.status_code != 200:
        st.error(" Impossible de récupérer la liste des modèles")
        return
    
    models_data = response.json()
    models_list = models_data.get("models", [])
    
    if not models_list:
        st.warning(" Aucun modèle disponible. Veuillez d'abord uploader un modèle.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration du Test")
        
        model_options = {m["name"]: m["id"] for m in models_list}
        selected_model_name = st.selectbox("Choisir le modèle", list(model_options.keys()))
        selected_model_id = model_options[selected_model_name]
        
        st.subheader("Types de Tests")
        test_types = st.multiselect(
            "Sélectionner les tests à effectuer",
            ["reasoning", "language", "math", "speed", "creative", "memory", "logic", "comprehension", "coding"],
            default=["reasoning", "language", "math"]
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            difficulty = st.select_slider(
                "Niveau de difficulté",
                options=["easy", "medium", "hard", "expert"],
                value="medium"
            )
        
        with col_b:
            num_samples = st.slider("Nombre d'échantillons", 10, 500, 100, 10)
        
        with st.expander(" Options Avancées"):
            enable_mlflow = st.checkbox("Activer le tracking MLflow", value=True)
            enable_detailed = st.checkbox("Rapport détaillé", value=True)
        
        if st.button(" Lancer les Tests", type="primary"):
            with st.spinner("Tests en cours... Cela peut prendre plusieurs minutes."):
                test_config = {
                    "model_id": selected_model_id,
                    "test_types": test_types,
                    "difficulty_level": difficulty,
                    "num_samples": num_samples
                }
                
                response = make_api_request(
                    "POST",
                    "/api/v1/tests/run",
                    json=test_config
                )
                
                if response and response.status_code == 200:
                    result = response.json()
                    test_id = result["test_id"]
                    
                    st.success(" Tests lancés avec succès!")
                    st.info(f" Test ID: {test_id}")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        time.sleep(2)
                        status_response = make_api_request(
                            "GET",
                            f"/api/v1/tests/status/{test_id}"
                        )
                        
                        if status_response and status_response.status_code == 200:
                            status_data = status_response.json()
                            status = status_data.get("status")
                            
                            progress_bar.progress((i + 1) / 100)
                            status_text.text(f"Status: {status}")
                            
                            if status == "completed":
                                st.success(" Tests terminés!")
                                st.json(status_data.get("final_score"))
                                st.balloons()
                                break
                            elif status == "failed":
                                st.error(f" Tests échoués: {status_data.get('error')}")
                                break
                else:
                    st.error(" Erreur lors du lancement des tests")
    
    with col2:
        st.info("""
        ###  Guide des Tests
        
        **Reasoning**: Logique et raisonnement
        
        **Language**: Compréhension de texte
        
        **Math**: Capacités mathématiques
        
        **Speed**: Performance et vitesse
        
        **Creative**: Tests de créativité
        
        **Memory**: Gestion du contexte
        
        **Logic**: Logique formelle
        
        **Comprehension**: Compréhension avancée
        
        **Coding**: Génération de code
        """)

def show_results_page():
    """Page de visualisation des résultats"""
    st.header(" Résultats des Tests")
    
    response = make_api_request("GET", "/api/v1/models")
    
    if not response or response.status_code != 200:
        st.error(" Impossible de récupérer les modèles")
        return
    
    models_data = response.json()
    models_list = models_data.get("models", [])
    
    if not models_list:
        st.warning(" Aucun modèle disponible")
        return
    
    model_options = {m["name"]: m["id"] for m in models_list}
    selected_model_name = st.selectbox("Choisir le modèle", list(model_options.keys()))
    selected_model_id = model_options[selected_model_name]
    
    results_response = make_api_request("GET", f"/api/v1/results/{selected_model_id}")
    
    if not results_response or results_response.status_code != 200:
        st.error(" Erreur lors de la récupération des résultats")
        return
    
    results_data = results_response.json()
    tests = results_data.get("tests", [])
    
    if not tests:
        st.info(" Aucun test effectué pour ce modèle")
        return
    
    latest_test = tests[-1] if tests else None
    
    if latest_test and latest_test.get("status") == "completed":
        st.subheader(" Derniers Résultats")
        
        final_score = latest_test.get("final_score", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score Global", f"{final_score.get('overall_score', 0):.1f}/100")
        with col2:
            st.metric("Grade", final_score.get('grade', 'N/A'))
        with col3:
            st.metric("Tests Effectués", len(latest_test.get("results", {}).get("tests", {})))
        with col4:
            st.metric("Statut", " Terminé")
        
        st.markdown("---")
        
        test_results = latest_test.get("results", {}).get("tests", {})
        
        if test_results:
            categories = list(test_results.keys())
            scores = [test_results[cat].get("score", 0) for cat in categories]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name=selected_model_name
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Performance par Catégorie",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(" Détails des Tests")
        
        for test_name, test_data in test_results.items():
            with st.expander(f"{test_name.capitalize()} - Score: {test_data.get('score', 0):.1f}/100"):
                st.json(test_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Appréciation")
            st.info(final_score.get("appreciation", ""))
            
            st.subheader(" Points Forts")
            for strength in final_score.get("strengths", []):
                st.success(f"✓ {strength}")
        
        with col2:
            st.subheader(" Recommandations")
            for rec in final_score.get("recommendations", []):
                st.warning(f"→ {rec}")
            
            st.subheader(" Points à Améliorer")
            for weakness in final_score.get("weaknesses", []):
                st.error(f"✗ {weakness}")
    else:
        st.info(" Tests en cours ou non terminés")

def show_comparison_page():
    """Page de comparaison de modèles"""
    st.header(" Comparaison de Modèles")
    
    response = make_api_request("GET", "/api/v1/models")
    
    if not response or response.status_code != 200:
        st.error(" Impossible de récupérer les modèles")
        return
    
    models_data = response.json()
    models_list = models_data.get("models", [])
    
    if len(models_list) < 2:
        st.warning(" Vous devez avoir au moins 2 modèles pour effectuer une comparaison")
        return
    
    st.subheader("Sélection des Modèles")
    
    model_options = {m["name"]: m["id"] for m in models_list}
    selected_models = st.multiselect(
        "Choisir les modèles à comparer (2-5 modèles)",
        list(model_options.keys()),
        max_selections=5
    )
    
    if len(selected_models) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            test_suite = st.selectbox(
                "Suite de tests",
                ["complete", "reasoning", "language", "performance", "creative"]
            )
        
        with col2:
            metrics = st.multiselect(
                "Métriques à comparer",
                ["accuracy", "speed", "score", "efficiency"],
                default=["accuracy", "score"]
            )
        
        if st.button(" Comparer les Modèles", type="primary"):
            with st.spinner("Comparaison en cours..."):
                model_ids = [model_options[name] for name in selected_models]
                
                comparison_request = {
                    "model_ids": model_ids,
                    "test_suite": test_suite,
                    "metrics": metrics
                }
                
                response = make_api_request(
                    "POST",
                    "/api/v1/compare",
                    json=comparison_request
                )
                
                if response and response.status_code == 200:
                    comparison_data = response.json()
                    comparison = comparison_data.get("comparison", {})
                    
                    st.success(" Comparaison terminée!")
                    
                    st.subheader("🏆 Classement")
                    ranking = comparison.get("ranking", [])
                    
                    for i, model in enumerate(ranking, 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        col_a, col_b, col_c = st.columns([1, 3, 1])
                        with col_a:
                            st.markdown(f"### {medal}")
                        with col_b:
                            st.markdown(f"**{model['name']}**")
                        with col_c:
                            st.metric("Score", f"{model['average_score']:.2f}")
                    
                    st.markdown("---")
                    st.subheader(" Comparaison Visuelle")
                    
                    comparison_df = pd.DataFrame(ranking)
                    
                    fig = px.bar(
                        comparison_df,
                        x='name',
                        y='average_score',
                        title='Scores de Performance',
                        labels={'name': 'Modèle', 'average_score': 'Score'},
                        color='average_score',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander(" Détails Complets"):
                        st.json(comparison)
                else:
                    st.error(" Erreur lors de la comparaison")
    else:
        st.info(" Veuillez sélectionner au moins 2 modèles pour la comparaison")

def show_leaderboard_page():
    """Page du leaderboard"""
    st.header("🏆 Classement des Modèles")
    
    response = make_api_request("GET", "/api/v1/leaderboard?limit=20")
    
    if not response or response.status_code != 200:
        st.error(" Erreur lors de la récupération du leaderboard")
        return
    
    leaderboard_data = response.json()
    leaderboard = leaderboard_data.get("leaderboard", [])
    
    if not leaderboard:
        st.info(" Aucun modèle testé pour le moment")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox(
            "Filtrer par type",
            ["Tous"] + list(set([m["model_type"] for m in leaderboard]))
        )
    
    with col2:
        sort_by = st.selectbox("Trier par", ["Score", "Nombre de tests", "Nom"])
    
    with col3:
        order = st.radio("Ordre", ["Décroissant", "Croissant"])
    
    filtered_leaderboard = leaderboard
    if filter_type != "Tous":
        filtered_leaderboard = [m for m in leaderboard if m["model_type"] == filter_type]
    
    st.markdown("---")
    
    for i, model in enumerate(filtered_leaderboard, 1):
        col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
        
        with col1:
            if i == 1:
                st.markdown("# 🥇")
            elif i == 2:
                st.markdown("# 🥈")
            elif i == 3:
                st.markdown("# 🥉")
            else:
                st.markdown(f"### {i}")
        
        with col2:
            st.markdown(f"**{model['model_name']}**")
            st.caption(f"Type: {model['model_type']}")
        
        with col3:
            st.metric("Score Moyen", f"{model['average_score']:.2f}")
        
        with col4:
            st.metric("Tests", model['total_tests'])
        
        st.markdown("---")
    
    st.subheader(" Top 10 Modèles")
    
    top_10 = filtered_leaderboard[:10]
    df = pd.DataFrame(top_10)
    
    fig = px.bar(
        df,
        x='model_name',
        y='average_score',
        color='average_score',
        title='Top 10 des Meilleurs Modèles',
        labels={'model_name': 'Modèle', 'average_score': 'Score Moyen'},
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_analytics_page():
    """Page d'analytics"""
    st.header(" Analytics et Statistiques Avancées")
    
    stats_response = make_api_request("GET", "/api/v1/stats")
    
    if not stats_response or stats_response.status_code != 200:
        st.error(" Erreur lors de la récupération des statistiques")
        return
    
    stats = stats_response.json()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Modèles", stats.get("total_models", 0))
    with col2:
        st.metric("Tests Effectués", stats.get("completed_tests", 0))
    with col3:
        st.metric("Score Moyen", f"{stats.get('average_score', 0):.1f}")
    with col4:
        st.metric("Utilisateurs", stats.get("total_users", 0))
    
    st.markdown("---")
    
    models_response = make_api_request("GET", "/api/v1/models")
    
    if models_response and models_response.status_code == 200:
        models_data = models_response.json()
        models_list = models_data.get("models", [])
        
        if models_list:
            st.subheader(" Distribution des Types de Modèles")
            
            types_count = {}
            for model in models_list:
                model_type = model.get("type", "unknown")
                types_count[model_type] = types_count.get(model_type, 0) + 1
            
            fig_pie = px.pie(
                values=list(types_count.values()),
                names=list(types_count.keys()),
                title="Répartition par Type"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader(" Évolution des Performances (Simulé)")
            
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
            scores = np.random.uniform(60, 95, len(dates))
            
            df_time = pd.DataFrame({
                'Date': dates,
                'Score Moyen': scores
            })
            
            fig_line = px.line(
                df_time,
                x='Date',
                y='Score Moyen',
                title='Évolution des Scores dans le Temps',
                markers=True
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            st.subheader(" Heatmap des Performances (Simulé)")
            
            categories = ["Reasoning", "Language", "Math", "Speed", "Creative"]
            models_sample = [m["name"][:15] for m in models_list[:5]]
            
            heatmap_data = np.random.uniform(50, 100, (len(models_sample), len(categories)))
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=categories,
                y=models_sample,
                colorscale='RdYlGn'
            ))
            
            fig_heatmap.update_layout(
                title="Performances par Catégorie et Modèle",
                xaxis_title="Catégorie",
                yaxis_title="Modèle"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

def show_settings_page():
    """Page des paramètres"""
    st.header(" Paramètres")
    
    tab1, tab2, tab3 = st.tabs([" Profil", " Sécurité", "ℹ À propos"])
    
    with tab1:
        st.subheader("Informations du Profil")
        
        if st.session_state.username:
            st.info(f"**Utilisateur:** {st.session_state.username}")
            
            with st.form("profile_form"):
                new_full_name = st.text_input("Nom complet")
                new_email = st.text_input("Email")
                
                if st.form_submit_button(" Mettre à jour le profil"):
                    st.success(" Profil mis à jour!")
    
    with tab2:
        st.subheader("Sécurité")
        
        with st.form("password_form"):
            current_password = st.text_input("Mot de passe actuel", type="password")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            
            if st.form_submit_button(" Changer le mot de passe"):
                if new_password == confirm_password:
                    st.success(" Mot de passe changé!")
                else:
                    st.error(" Les mots de passe ne correspondent pas")
        
        st.markdown("---")
        
        if st.button(" Se Déconnecter", type="secondary"):
            logout()
            st.rerun()
    
    with tab3:
        st.subheader("À propos de la plateforme")
        
        st.markdown("""
        ###  AI Model Benchmark Platform v2.0
        
        Plateforme complète pour tester, évaluer et comparer les performances 
        des modèles d'Intelligence Artificielle.
        
        **Fonctionnalités:**
        - 9 types de tests de performance
        - Authentification sécurisée (JWT)
        - Tracking MLflow intégré
        - Comparaison multi-modèles
        - Analytics avancées
        - Interface intuitive
        
        **Technologies:**
        - Backend: FastAPI
        - Frontend: Streamlit
        - ML Tracking: MLflow
        - Auth: JWT + bcrypt
        
        **Développé apar NADA dépuis 2024**
        
        ---
        
        ###  Statistiques de l'API
        """)
        
        if check_api_status():
            st.success(" API connectée")
            st.code(f"URL: {API_URL}")
        else:
            st.error(" API non accessible")

# ============================================================
# NAVIGATION ET MAIN
# ============================================================

def main():
    """Fonction principale"""
    init_session_state()
    
    # Vérifier la connexion à l'API
    api_status = check_api_status()
    
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AI+Benchmark", use_container_width=True)
        st.title("Navigation")
        
        # Afficher l'utilisateur connecté
        st.success(f"👤 Connecté: **{st.session_state.username}**")
        
        page = st.radio(
            "Choisir une page",
            [" Accueil", " Upload Modèle", " Lancer Tests", " Résultats", 
             " Comparer Modèles", " Leaderboard", " Analytics", " Paramètres"]
        )
        
        st.markdown("---")
        
        # Status API
        if api_status:
            st.success(" API: Connecté")
        else:
            st.error(" API: Déconnecté")
        
        st.markdown("---")
        
        # Statistiques rapides
        st.markdown("###  Statistiques")
        try:
            stats_response = make_api_request("GET", "/api/v1/stats")
            if stats_response and stats_response.status_code == 200:
                stats = stats_response.json()
                st.metric("Modèles", stats.get("total_models", 0))
                st.metric("Tests", stats.get("completed_tests", 0))
        except:
            st.metric("Modèles", "N/A")
        
        st.markdown("---")
        
        # Liens rapides
        st.markdown("###  Liens Utiles")
        st.markdown("- [Documentation API](http://127.0.0.1:8009/docs)")
        st.markdown("- [MLflow UI](http://localhost:5000)")
        
        st.markdown("---")
        
        if st.button("🚪 Déconnexion", use_container_width=True):
            logout()
            st.rerun()
    
    # Afficher la page sélectionnée
    if page == " Accueil":
        show_home_page()
    elif page == " Upload Modèle":
        show_upload_page()
    elif page == " Lancer Tests":
        show_tests_page()
    elif page == " Résultats":
        show_results_page()
    elif page == " Comparer Modèles":
        show_comparison_page()
    elif page == " Leaderboard":
        show_leaderboard_page()
    elif page == " Analytics":
        show_analytics_page()
    elif page == " Paramètres":
        show_settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p> AI Model Benchmark Platform v2.0.0</p>
        <p>Développé par NADA pour l'évaluation des modèle</p>
        <p>© 2024 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()