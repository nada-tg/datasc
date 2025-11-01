"""
ai_connector_frontend.py - Interface Streamlit

Lancement:
streamlit run ai_connector_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

st.set_page_config(
    page_title="AI Connector Platform",
    page_icon="🔗",
    layout="wide"
)

API_URL = "http://localhost:8003"

def init_session():
    if 'registered_models' not in st.session_state:
        st.session_state.registered_models = []
    if 'current_connection' not in st.session_state:
        st.session_state.current_connection = None

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# PAGE: Dashboard
def page_dashboard():
    st.title("AI Connector Platform")
    st.write("Connectez et benchmarkez plusieurs modèles d'IA")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/statistics")
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Modèles Enregistrés", stats['total_models'])
            with col2:
                st.metric("Connexions Actives", stats['total_connections'])
            with col3:
                st.metric("Requêtes Traitées", stats['total_queries'])
            with col4:
                st.metric("Benchmarks", stats['total_benchmarks'])
            
            st.write("---")
            
            # Distribution par type
            if stats['models_by_type']:
                st.subheader("Modèles par Type")
                df = pd.DataFrame(list(stats['models_by_type'].items()), columns=['Type', 'Nombre'])
                fig = px.bar(df, x='Type', y='Nombre')
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Enregistrer un Modèle", type="primary", use_container_width=True):
            st.session_state.active_view = 'register_model'
            st.rerun()
    
    with col2:
        if st.button("Créer une Connexion", use_container_width=True):
            st.session_state.active_view = 'create_connection'
            st.rerun()
    
    with col3:
        if st.button("Lancer un Benchmark", use_container_width=True):
            st.session_state.active_view = 'benchmark'
            st.rerun()

# PAGE: Enregistrer Modèle
def page_register_model():
    st.title("Enregistrer un Modèle IA")
    
    with st.form("register_model_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom du Modèle *")
            model_type = st.selectbox("Type", ["chatgpt", "claude", "llama", "gemini", "mistral", "palm", "custom"])
            model_version = st.text_input("Version", "latest")
        
        with col2:
            api_key = st.text_input("Clé API (optionnel)", type="password")
            endpoint = st.text_input("Endpoint (optionnel)")
        
        st.write("---")
        st.subheader("Paramètres")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        with col2:
            max_tokens = st.number_input("Max Tokens", 100, 8000, 2000)
        with col3:
            st.write("")  # Spacing
        
        custom_params = st.text_area("Paramètres Personnalisés (JSON)")
        
        submitted = st.form_submit_button("Enregistrer", type="primary")
        
        if submitted:
            if not name:
                st.error("Le nom est requis")
            else:
                payload = {
                    "name": name,
                    "model_type": model_type,
                    "api_key": api_key if api_key else None,
                    "endpoint": endpoint if endpoint else None,
                    "model_version": model_version,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "custom_params": json.loads(custom_params) if custom_params else {}
                }
                
                try:
                    response = requests.post(f"{API_URL}/api/v1/models/register", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Modèle enregistré avec succès! ID: {result['model_id']}")
                        time.sleep(2)
                        st.session_state.active_view = 'models_list'
                        st.rerun()
                    else:
                        st.error(f"Erreur: {response.text}")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

# PAGE: Liste des Modèles
def page_models_list():
    st.title("Modèles Enregistrés")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/models")
        
        if response.status_code == 200:
            data = response.json()
            models = data['models']
            
            if not models:
                st.info("Aucun modèle enregistré")
                return
            
            # Tableau
            df = pd.DataFrame(models)
            display_cols = ['name', 'model_type', 'model_version', 'temperature', 'max_tokens', 'status']
            st.dataframe(df[display_cols], use_container_width=True)
            
            st.write("---")
            
            # Détails par modèle
            for model in models:
                with st.expander(f"{model['name']} ({model['model_type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**ID:** {model['id']}")
                        st.write(f"**Version:** {model['model_version']}")
                        st.write(f"**Status:** {model['status']}")
                    
                    with col2:
                        st.write(f"**Temperature:** {model['temperature']}")
                        st.write(f"**Max Tokens:** {model['max_tokens']}")
                        st.write(f"**Enregistré:** {model['registered_at'][:10]}")
                    
                    with col3:
                        if model.get('endpoint'):
                            st.write(f"**Endpoint:** {model['endpoint']}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Créer Connexion
def page_create_connection():
    st.title("Créer une Connexion")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/models")
        
        if response.status_code == 200:
            models = response.json()['models']
            
            if len(models) < 2:
                st.warning("Vous devez enregistrer au moins 2 modèles pour créer une connexion")
                return
            
            with st.form("connection_form"):
                name = st.text_input("Nom de la Connexion *")
                
                # Sélection des modèles
                st.write("Sélectionnez les modèles à connecter")
                selected_models = []
                
                for model in models:
                    if st.checkbox(f"{model['name']} ({model['model_type']})", key=f"model_{model['id']}"):
                        selected_models.append(model['id'])
                
                connection_type = st.selectbox(
                    "Type de Connexion",
                    ["parallel", "sequential", "voting", "hierarchical"]
                )
                
                synthesis_strategy = st.selectbox(
                    "Stratégie de Synthèse",
                    ["best_response", "consensus", "fusion", "voting"]
                )
                
                description = st.text_area("Description")
                
                submitted = st.form_submit_button("Créer Connexion", type="primary")
                
                if submitted:
                    if not name or len(selected_models) < 2:
                        st.error("Nom requis et au moins 2 modèles")
                    else:
                        payload = {
                            "name": name,
                            "model_ids": selected_models,
                            "connection_type": connection_type,
                            "synthesis_strategy": synthesis_strategy,
                            "description": description
                        }
                        
                        try:
                            resp = requests.post(f"{API_URL}/api/v1/connections/create", json=payload)
                            
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success("Connexion créée!")
                                st.session_state.current_connection = result['connection_id']
                                time.sleep(1)
                                st.session_state.active_view = 'query'
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Exécuter Requête
def page_query():
    st.title("Exécuter une Requête")
    
    connection_id = st.session_state.current_connection
    
    if not connection_id:
        st.warning("Aucune connexion sélectionnée")
        return
    
    query = st.text_area("Votre Requête", height=150, placeholder="Posez une question complexe...")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Envoyer", type="primary", use_container_width=True):
            if query:
                with st.spinner("Traitement en cours..."):
                    payload = {
                        "connection_id": connection_id,
                        "query": query
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/api/v1/query", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("Requête traitée!")
                            
                            # Réponse synthétisée
                            st.write("---")
                            st.subheader("Réponse Synthétisée")
                            
                            synthesis = result['synthesis']
                            
                            st.info(synthesis['synthesized_response'])
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Source", synthesis['source_model'])
                            with col2:
                                st.metric("Confiance", f"{synthesis['confidence']:.1f}%")
                            with col3:
                                st.metric("Méthode", synthesis['synthesis_method'])
                            
                            # Réponses individuelles
                            st.write("---")
                            st.subheader("Réponses Individuelles")
                            
                            for resp in result['individual_responses']:
                                with st.expander(f"{resp['model_name']} - {resp['model_type']}"):
                                    st.write(resp['response'])
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Confiance", f"{resp['confidence']:.1f}%")
                                    with col2:
                                        st.metric("Temps", f"{resp['response_time']:.2f}s")
                                    with col3:
                                        st.metric("Tokens", resp['tokens_used'])
                    
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
            else:
                st.warning("Veuillez entrer une requête")

# PAGE: Benchmark
def page_benchmark():
    st.title("Créer un Benchmark")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/models")
        
        if response.status_code == 200:
            models = response.json()['models']
            
            if not models:
                st.warning("Aucun modèle disponible")
                return
            
            with st.form("benchmark_form"):
                name = st.text_input("Nom du Benchmark *")
                
                st.write("Sélectionnez les modèles à tester")
                selected_models = []
                
                for model in models:
                    if st.checkbox(f"{model['name']}", key=f"bench_{model['id']}"):
                        selected_models.append(model['id'])
                
                benchmark_type = st.selectbox(
                    "Type de Test",
                    ["reasoning", "coding", "math", "creative", "factual", "multilingual", "comprehensive"]
                )
                
                st.write("---")
                st.subheader("Cas de Test")
                
                num_tests = st.number_input("Nombre de tests", 1, 20, 5)
                
                test_cases = []
                for i in range(num_tests):
                    test_cases.append({
                        "name": f"Test {i+1}",
                        "difficulty": "medium",
                        "category": benchmark_type
                    })
                
                submitted = st.form_submit_button("Lancer Benchmark", type="primary")
                
                if submitted:
                    if not name or not selected_models:
                        st.error("Nom et modèles requis")
                    else:
                        payload = {
                            "name": name,
                            "model_ids": selected_models,
                            "benchmark_type": benchmark_type,
                            "test_cases": test_cases
                        }
                        
                        try:
                            resp = requests.post(f"{API_URL}/api/v1/benchmark/create", json=payload)
                            
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success("Benchmark lancé!")
                                st.session_state.current_benchmark = result['benchmark_id']
                                time.sleep(2)
                                st.session_state.active_view = 'benchmark_results'
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Résultats Benchmark
def page_benchmark_results():
    st.title("Résultats du Benchmark")
    
    benchmark_id = st.session_state.get('current_benchmark')
    
    if not benchmark_id:
        st.warning("Aucun benchmark sélectionné")
        return
    
    try:
        response = requests.get(f"{API_URL}/api/v1/benchmark/{benchmark_id}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == 'pending':
                st.info("Benchmark en attente...")
                if st.button("Rafraîchir"):
                    st.rerun()
                return
            
            if data['status'] == 'running':
                st.info("Benchmark en cours...")
                if st.button("Rafraîchir"):
                    st.rerun()
                return
            
            results = data['results']
            
            # Classement
            st.subheader("Classement")
            
            rankings = results['rankings']
            
            for rank in rankings[:3]:
                medal = ["🥇", "🥈", "🥉"][rank['rank']-1] if rank['rank'] <= 3 else f"{rank['rank']}."
                st.write(f"{medal} **{rank['model_name']}** - Score: {rank['average_score']:.2f} ({rank['passed_tests']}/{rank['total_tests']} tests réussis)")
            
            st.write("---")
            
            # Graphique de comparaison
            st.subheader("Comparaison des Scores")
            
            df_rankings = pd.DataFrame(rankings)
            fig = px.bar(df_rankings, x='model_name', y='average_score', color='average_score', 
                        title="Scores Moyens par Modèle")
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails par modèle
            st.write("---")
            st.subheader("Détails par Modèle")
            
            for model_id, result in results['model_results'].items():
                with st.expander(f"{result['model_name']} - Score: {result['average_score']:.2f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Tests Réussis", f"{result['passed']}/{result['total_tests']}")
                    with col2:
                        st.metric("Score Moyen", f"{result['average_score']:.2f}")
                    with col3:
                        st.metric("Écart-Type", f"{result['metrics']['std']:.2f}")
                    
                    # Graphique des scores
                    scores = [detail['score'] for detail in result['details']]
                    fig = px.line(x=range(1, len(scores)+1), y=scores, 
                                 title="Scores par Test", markers=True)
                    fig.update_xaxes(title="Test #")
                    fig.update_yaxes(title="Score")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métriques détaillées
                    st.write("**Métriques Statistiques:**")
                    metrics_df = pd.DataFrame([result['metrics']])
                    st.dataframe(metrics_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Architecture
def page_architecture():
    st.title("Designer l'Architecture de Test")
    
    st.info("Créez une architecture visuelle pour organiser vos tests")
    
    with st.form("architecture_form"):
        name = st.text_input("Nom de l'Architecture *")
        description = st.text_area("Description")
        
        st.write("---")
        st.subheader("Définir les Nœuds")
        
        num_nodes = st.number_input("Nombre de nœuds", 1, 10, 3)
        
        nodes = []
        for i in range(num_nodes):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                node_name = st.text_input(f"Nœud {i+1} - Nom", value=f"Node_{i+1}", key=f"node_name_{i}")
            with col2:
                node_type = st.selectbox(f"Type", ["input", "processor", "output"], key=f"node_type_{i}")
            with col3:
                node_x = st.number_input("Position X", 0, 1000, min(i*30, 100), key=f"node_x_{i}")
            
            nodes.append({
                "id": f"node_{i}",
                "name": node_name,
                "type": node_type,
                "x": node_x,
                "y": 50
            })
        
        st.write("---")
        st.subheader("Définir les Connexions")
        
        num_connections = st.number_input("Nombre de connexions", 0, 10, 2)
        
        connections = []
        for i in range(num_connections):
            col1, col2 = st.columns(2)
            
            with col1:
                source = st.selectbox(f"Depuis", [n['name'] for n in nodes], key=f"conn_src_{i}")
            with col2:
                target = st.selectbox(f"Vers", [n['name'] for n in nodes], key=f"conn_tgt_{i}")
            
            connections.append({
                "source": source,
                "target": target
            })
        
        st.write("---")
        submitted = st.form_submit_button("Créer Architecture", type="primary")
        
        if submitted:
            if not name:
                st.error("Nom requis")
            else:
                payload = {
                    "name": name,
                    "description": description,
                    "nodes": nodes,
                    "connections": connections
                }
                
                try:
                    resp = requests.post(f"{API_URL}/api/v1/architecture/create", json=payload)
                    
                    if resp.status_code == 200:
                        st.success("Architecture créée!")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

# PAGE: Historique
def page_history():
    st.title("Historique des Requêtes")
    
    connection_id = st.session_state.get('current_connection')
    
    if not connection_id:
        st.warning("Sélectionnez une connexion")
        return
    
    try:
        response = requests.get(f"{API_URL}/api/v1/history/{connection_id}")
        
        if response.status_code == 200:
            data = response.json()
            history = data['history']
            
            if not history:
                st.info("Aucun historique")
                return
            
            st.write(f"**Total:** {data['total']} requêtes")
            
            for item in history:
                with st.expander(f"{item['query'][:100]}... - {item['timestamp'][:19]}"):
                    st.write(f"**Requête:** {item['query']}")
                    st.write(f"**Réponse:** {item['synthesis']['synthesized_response'][:200]}...")
                    
                    st.write(f"**Modèles utilisés:** {len(item['responses'])}")
                    
                    for resp in item['responses']:
                        st.write(f"- {resp['model_name']}: {resp['confidence']:.1f}% confiance")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Paramètres
def page_settings():
    st.title("Paramètres Avancés")
    
    tab1, tab2, tab3 = st.tabs(["Général", "Synthèse", "Performance"])
    
    with tab1:
        st.subheader("Configuration Générale")
        
        api_url = st.text_input("URL API", value=API_URL)
        timeout = st.number_input("Timeout (s)", 1, 60, 30)
        
        st.write("---")
        
        default_temp = st.slider("Temperature par défaut", 0.0, 2.0, 0.7, 0.1)
        default_tokens = st.number_input("Tokens par défaut", 100, 8000, 2000)
    
    with tab2:
        st.subheader("Stratégies de Synthèse")
        
        st.write("**best_response:** Sélectionne la meilleure réponse")
        st.write("**consensus:** Construit un consensus")
        st.write("**fusion:** Fusionne toutes les réponses")
        st.write("**voting:** Vote majoritaire")
        
        st.write("---")
        
        confidence_threshold = st.slider("Seuil de confiance minimum", 0, 100, 70)
    
    with tab3:
        st.subheader("Optimisation")
        
        cache_enabled = st.checkbox("Activer le cache", value=True)
        parallel_requests = st.checkbox("Requêtes parallèles", value=True)
        
        max_retries = st.number_input("Tentatives max", 1, 5, 3)
    
    if st.button("Sauvegarder", type="primary"):
        st.success("Paramètres sauvegardés")

# Navigation
def main():
    init_session()
    
    with st.sidebar:
        st.title("AI Connector")
        
        menu = {
            "Dashboard": "dashboard",
            "Enregistrer Modèle": "register_model",
            "Liste Modèles": "models_list",
            "Créer Connexion": "create_connection",
            "Exécuter Requête": "query",
            "Benchmark": "benchmark",
            "Résultats Benchmark": "benchmark_results",
            "Architecture": "architecture",
            "Historique": "history",
            "Paramètres": "settings"
        }
        
        for label, view in menu.items():
            if st.button(label, use_container_width=True):
                st.session_state.active_view = view
                st.rerun()
        
        st.write("---")
        
        if check_api():
            st.success("API Connectée")
        else:
            st.error("API Déconnectée")
        
        st.caption("AI Connector Platform v1.0")
    
    view = st.session_state.get('active_view', 'dashboard')
    
    if view == 'dashboard':
        page_dashboard()
    elif view == 'register_model':
        page_register_model()
    elif view == 'models_list':
        page_models_list()
    elif view == 'create_connection':
        page_create_connection()
    elif view == 'query':
        page_query()
    elif view == 'benchmark':
        page_benchmark()
    elif view == 'benchmark_results':
        page_benchmark_results()
    elif view == 'architecture':
        page_architecture()
    elif view == 'history':
        page_history()
    elif view == 'settings':
        page_settings()

if __name__ == "__main__":
    main()
