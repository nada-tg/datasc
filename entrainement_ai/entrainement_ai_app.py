# ai_training_dashboard.py - Interface Streamlit pour Plateforme d'Entraînement IA
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta
import uuid
import asyncio
import websocket
import threading
from typing import Dict, List, Any
import base64
import io

# from universal_tokenizer_dashboard import call_tokenizer_api


# Configuration de la page  streamlit run entrainement_ai_app.py
st.set_page_config(
    page_title="AI Training Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration API
AI_TRAINING_API_URL = "http://localhost:8025"
WEBSOCKET_URL = "ws://localhost:8006"

# Style CSS avancé
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
    font-weight: 700;
}

.training-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.model-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border-left: 6px solid #1E88E5;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.model-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.metric-card {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.status-running {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.status-completed {
    background: linear-gradient(45deg, #2196F3, #1976D2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.status-failed {
    background: linear-gradient(45deg, #f44336, #d32f2f);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.progress-container {
    background: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.resource-monitor {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.dataset-preview {
    background: #f8f9fa;
    border: 2px dashed #1E88E5;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.hyperparameter-section {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# État de session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'
if 'ws_connection' not in st.session_state:
    st.session_state.ws_connection = None
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = {}

# Fonctions utilitaires
def call_ai_api(endpoint, method="GET", data=None):
    """Appel API AI Training Platform"""
    url = f"{AI_TRAINING_API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def setup_websocket():
    """Configure la connexion WebSocket pour les mises à jour en temps réel"""
    try:
        client_id = f"{st.session_state.user_id}_{uuid.uuid4().hex[:8]}"
        ws_url = f"{WEBSOCKET_URL}/ws/{client_id}"
        
        def on_message(ws, message):
            data = json.loads(message)
            if data.get("type") == "training_update":
                job_id = data.get("job_id")
                if job_id not in st.session_state.training_logs:
                    st.session_state.training_logs[job_id] = []
                st.session_state.training_logs[job_id].append(data)
        
        def on_error(ws, error):
            st.error(f"Erreur WebSocket: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            st.info("Connexion WebSocket fermée")
        
        def on_open(ws):
            st.success("Connexion temps réel établie")
        
        ws = websocket.WebSocketApp(ws_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        # Démarrer la connexion en arrière-plan
        def run_websocket():
            ws.run_forever()
        
        if not st.session_state.ws_connection:
            threading.Thread(target=run_websocket, daemon=True).start()
            st.session_state.ws_connection = ws
            
    except Exception as e:
        st.error(f"Impossible d'établir la connexion WebSocket: {e}")

def get_system_resources():
    """Récupère les ressources système"""
    resources, error = call_ai_api("/system/resources")
    if resources:
        return resources
    return {
        "cpu_percent": 0,
        "memory_percent": 0,
        "memory_used_gb": 0,
        "memory_total_gb": 0,
        "gpu_info": []
    }

def display_resource_monitor():
    """Affiche le monitoring des ressources système"""
    resources = get_system_resources()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="resource-monitor">
            <h4>CPU</h4>
            <h2>{resources.get('cpu_percent', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        memory_percent = resources.get('memory_percent', 0)
        memory_used = resources.get('memory_used_gb', 0)
        memory_total = resources.get('memory_total_gb', 0)
        st.markdown(f"""
        <div class="resource-monitor">
            <h4>RAM</h4>
            <h2>{memory_percent:.1f}%</h2>
            <p>{memory_used:.1f} / {memory_total:.1f} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gpu_info = resources.get('gpu_info', [])
        if gpu_info:
            gpu = gpu_info[0]
            st.markdown(f"""
            <div class="resource-monitor">
                <h4>GPU</h4>
                <h2>{gpu.get('load', 0):.1f}%</h2>
                <p>{gpu.get('name', 'GPU')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="resource-monitor">
                <h4>GPU</h4>
                <h2>N/A</h2>
                <p>Non disponible</p>
            </div>
            """, unsafe_allow_html=True)

def get_training_jobs(user_id):
    """Récupère les jobs d'entraînement d'un utilisateur"""
    jobs, error = call_ai_api(f"/training/jobs/{user_id}")
    if jobs:
        return jobs.get("jobs", [])
    return []

def get_available_datasets():
    """Récupère les datasets disponibles"""
    datasets, error = call_ai_api("/datasets/available")
    if datasets:
        return datasets.get("datasets", [])
    return []

def get_available_algorithms():
    """Récupère les algorithmes disponibles"""
    algorithms, error = call_ai_api("/training/algorithms")
    return algorithms if algorithms else {}

def main():
    st.markdown('<h1 class="main-header">🤖 AI Training Platform</h1>', unsafe_allow_html=True)
    
    # Configuration WebSocket
    setup_websocket()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Sections:",
            ["Dashboard", "Nouveau Modèle", "Mes Modèles", "Monitoring", "Datasets", "Modèles Déployés"]
        )
        
        st.divider()
        
        # Monitoring en temps réel des ressources
        st.subheader("Ressources Système")
        display_resource_monitor()
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID: `{st.session_state.user_id[:8]}...`")
        
        # Statut API
        health, error = call_ai_api("/health")
        if health:
            st.success("🟢 API Training en ligne")
            active_jobs = health.get("active_jobs", 0)
            st.metric("Jobs actifs", active_jobs)
        else:
            st.error("🔴 API Training hors ligne")
    
    # Pages principales
    if page == "Dashboard":
        show_dashboard()
    elif page == "Nouveau Modèle":
        show_new_model_page()
    elif page == "Mes Modèles":
        show_my_models_page()
    elif page == "Monitoring":
        show_monitoring_page()
    elif page == "Datasets":
        show_datasets_page()
    elif page == "Modèles Déployés":
        show_deployed_models_page()

def show_dashboard():
    """Dashboard principal"""
    st.title("Dashboard d'Entraînement IA")
    
    # Récupérer les jobs de l'utilisateur
    jobs = get_training_jobs(st.session_state.user_id)
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    total_jobs = len(jobs)
    running_jobs = len([j for j in jobs if j["status"] == "running"])
    completed_jobs = len([j for j in jobs if j["status"] == "completed"])
    failed_jobs = len([j for j in jobs if j["status"] == "failed"])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_jobs}</h3>
            <p>Total Modèles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{running_jobs}</h3>
            <p>En Cours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{completed_jobs}</h3>
            <p>Terminés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{failed_jobs}</h3>
            <p>Échoués</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Jobs récents avec statuts en temps réel
    st.subheader("Jobs d'Entraînement Récents")
    
    if jobs:
        for job in jobs[:5]:  # Afficher les 5 plus récents
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>{job['job_name']}</h4>
                        <p><strong>Algorithme:</strong> {job['algorithm']}</p>
                        <p><strong>Type:</strong> {job['model_type']} - {job['task_type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    status = job["status"]
                    if status == "running":
                        st.markdown('<span class="status-running">En cours</span>', unsafe_allow_html=True)
                    elif status == "completed":
                        st.markdown('<span class="status-completed">Terminé</span>', unsafe_allow_html=True)
                    elif status == "failed":
                        st.markdown('<span class="status-failed">Échoué</span>', unsafe_allow_html=True)
                    else:
                        st.write(f"Status: {status}")
                
                with col3:
                    progress = job.get("progress", 0)
                    st.progress(progress / 100.0)
                    st.write(f"{progress:.1f}%")
                
                with col4:
                    if st.button("Détails", key=f"details_{job['job_id']}"):
                        st.session_state.selected_job = job['job_id']
                        show_job_details(job['job_id'])
                
                st.divider()
    else:
        st.info("Aucun modèle d'entraînement. Créez votre premier modèle!")



def show_live_training_logs(job_id):
    """Affiche les logs d'entraînement en temps réel"""
    try:
        # Récupérer le statut avec les logs
        status_data, error = call_ai_api(f"/training/job/{job_id}/status")
        
        if error:
            st.error(f"Erreur récupération logs: {error}")
            return
        
        logs = status_data.get("logs", []) if status_data else []
        
        if logs:
            # Afficher les logs dans une zone de texte
            log_container = st.container()
            with log_container:
                st.subheader("Logs d'Entraînement")
                
                # Créer un texte scrollable avec les derniers logs
                recent_logs = logs[-20:]  # Derniers 20 logs
                log_text = "\n".join(recent_logs)
                
                st.text_area(
                    "",
                    value=log_text,
                    height=300,
                    disabled=True,
                    key=f"logs_{job_id}"
                )
        else:
            st.info("Aucun log disponible pour ce job.")
            
    except Exception as e:
        st.error(f"Erreur affichage logs: {e}")

def calculate_model_complexity_score(algorithm, hyperparams, dataset_size):
    """Calcule un score de complexité du modèle"""
    try:
        base_complexity = {
            "logistic_regression": 1,
            "linear_regression": 1,
            "random_forest": 3,
            "svm": 4,
            "neural_network": 6,
            "xgboost": 5,
            "gradient_boosting": 4
        }
        
        complexity = base_complexity.get(algorithm, 3)
        
        # Ajustements basés sur les hyperparamètres
        if "n_estimators" in hyperparams:
            complexity += min(hyperparams["n_estimators"] / 100, 3)
        
        if "max_depth" in hyperparams and hyperparams["max_depth"]:
            complexity += min(hyperparams["max_depth"] / 10, 2)
        
        if "epochs" in hyperparams:
            complexity += min(hyperparams["epochs"] / 100, 4)
        
        if "hidden_sizes" in hyperparams:
            total_neurons = sum(hyperparams["hidden_sizes"])
            complexity += min(total_neurons / 1000, 3)
        
        # Ajustement selon la taille du dataset
        size_factor = min(dataset_size / 100000, 2)
        complexity += size_factor
        
        return min(complexity, 10)  # Score max de 10
        
    except Exception:
        return 5  # Score par défaut

def format_model_performance(metrics):
    """Formate les métriques de performance pour l'affichage"""
    try:
        formatted = {}
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ["accuracy", "precision", "recall", "f1_score"]:
                    formatted[key.replace("_", " ").title()] = f"{value:.1%}"
                elif key in ["loss", "mse", "rmse"]:
                    formatted[key.upper()] = f"{value:.4f}"
                elif key == "r2_score":
                    formatted["R² Score"] = f"{value:.4f}"
                else:
                    formatted[key.replace("_", " ").title()] = f"{value:.4f}"
            else:
                formatted[key.replace("_", " ").title()] = str(value)
        
        return formatted
        
    except Exception:
        return {"Error": "Format invalide"}

def create_model_comparison_chart(jobs):
    """Crée un graphique de comparaison des modèles"""
    try:
        if not jobs or len(jobs) < 2:
            return None
        
        # Extraire les données de performance
        comparison_data = []
        
        for job in jobs:
            if job["status"] == "completed" and job.get("final_metrics"):
                metrics = job["final_metrics"]
                
                row = {
                    "model_name": f"{job['job_name'][:15]}...",
                    "algorithm": job["algorithm"],
                    "model_type": job["model_type"]
                }
                
                # Ajouter les métriques principales
                if "accuracy" in metrics:
                    row["accuracy"] = metrics["accuracy"]
                if "f1_score" in metrics:
                    row["f1_score"] = metrics["f1_score"]
                if "mse" in metrics:
                    row["mse"] = metrics["mse"]
                if "r2_score" in metrics:
                    row["r2_score"] = metrics["r2_score"]
                
                comparison_data.append(row)
        
        if not comparison_data:
            return None
        
        df = pd.DataFrame(comparison_data)
        
        # Créer le graphique selon les métriques disponibles
        if "accuracy" in df.columns:
            fig = px.bar(
                df, 
                x="model_name", 
                y="accuracy",
                color="algorithm",
                title="Comparaison de Précision des Modèles",
                labels={"accuracy": "Précision", "model_name": "Modèle"}
            )
        elif "r2_score" in df.columns:
            fig = px.bar(
                df,
                x="model_name",
                y="r2_score", 
                color="algorithm",
                title="Comparaison R² des Modèles",
                labels={"r2_score": "Score R²", "model_name": "Modèle"}
            )
        else:
            return None
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        st.error(f"Erreur création graphique comparaison: {e}")
        return None

def show_model_insights(job_data):
    """Affiche des insights avancés sur le modèle"""
    try:
        st.subheader("Insights du Modèle")
        
        # Calculer des métriques dérivées
        final_metrics = job_data.get("final_metrics", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Analyse de Performance:**")
            
            if "accuracy" in final_metrics:
                accuracy = final_metrics["accuracy"]
                if accuracy > 0.9:
                    st.success("Excellente performance (>90%)")
                elif accuracy > 0.8:
                    st.info("Bonne performance (80-90%)")
                elif accuracy > 0.7:
                    st.warning("Performance correcte (70-80%)")
                else:
                    st.error("Performance faible (<70%)")
            
            # Recommandations d'amélioration
            st.write("**Recommandations:**")
            recommendations = generate_model_recommendations(job_data)
            for rec in recommendations:
                st.write(f"• {rec}")
        
        with col2:
            st.write("**Caractéristiques du Modèle:**")
            
            # Score de complexité
            dataset_size = 1000  # Valeur par défaut, à récupérer des métadonnées
            complexity = calculate_model_complexity_score(
                job_data["algorithm"],
                {},  # hyperparams pas disponibles ici
                dataset_size
            )
            
            complexity_color = "red" if complexity > 7 else "orange" if complexity > 4 else "green"
            st.markdown(f"**Complexité:** <span style='color:{complexity_color}'>{complexity}/10</span>", 
                       unsafe_allow_html=True)
            
            # Temps d'entraînement
            if job_data.get("started_at") and job_data.get("completed_at"):
                duration = format_job_duration(job_data["started_at"], job_data["completed_at"])
                st.write(f"**Durée d'entraînement:** {duration}")
            
            # Type de tâche et algorithme
            st.write(f"**Tâche:** {job_data['task_type'].title()}")
            st.write(f"**Framework:** {job_data['model_type'].title()}")
            
    except Exception as e:
        st.error(f"Erreur génération insights: {e}")

def generate_model_recommendations(job_data):
    """Génère des recommandations pour améliorer le modèle"""
    recommendations = []
    
    try:
        final_metrics = job_data.get("final_metrics", {})
        algorithm = job_data.get("algorithm", "")
        
        # Recommandations basées sur la performance
        if "accuracy" in final_metrics:
            accuracy = final_metrics["accuracy"]
            
            if accuracy < 0.7:
                recommendations.append("Essayez d'augmenter la taille du dataset")
                recommendations.append("Considérez un préprocessing plus poussé")
                
                if algorithm == "logistic_regression":
                    recommendations.append("Testez Random Forest ou XGBoost")
                elif algorithm == "svm":
                    recommendations.append("Ajustez le paramètre C et le noyau")
            
            elif accuracy > 0.95:
                recommendations.append("Attention au surapprentissage - vérifiez la validation croisée")
                recommendations.append("Considérez la régularisation")
        
        # Recommandations spécifiques par algorithme
        if algorithm == "neural_network":
            recommendations.append("Surveillez le surapprentissage avec early stopping")
            recommendations.append("Expérimentez avec différentes architectures")
        elif algorithm == "random_forest":
            recommendations.append("Optimisez n_estimators et max_depth")
            recommendations.append("Analysez l'importance des features")
        elif algorithm == "xgboost":
            recommendations.append("Tunez learning_rate et max_depth")
            recommendations.append("Utilisez la validation croisée pour éviter le surapprentissage")
        
        # Recommandations générales
        recommendations.append("Analysez la matrice de confusion pour identifier les classes problématiques")
        recommendations.append("Considérez l'ensemble de modèles pour améliorer la robustesse")
        
        return recommendations[:4]  # Limiter à 4 recommandations
        
    except Exception:
        return ["Analysez les métriques de performance", 
                "Optimisez les hyperparamètres",
                "Vérifiez la qualité des données"]

def refresh_data_automatically():
    """Actualise automatiquement les données si l'option est activée"""
    if st.session_state.get("auto_refresh", False):
        refresh_interval = st.session_state.get("refresh_interval", 5)
        
        # Utiliser un placeholder pour le timer
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh > refresh_interval:
            st.session_state.last_refresh = current_time
            st.rerun()

# Corriger la fonction principale pour inclure l'auto-refresh
def main():
    st.markdown('<h1 class="main-header">🤖 AI Training Platform</h1>', unsafe_allow_html=True)
    
    # Auto-refresh si activé
    refresh_data_automatically()
    
    # Configuration WebSocket
    setup_websocket()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Sections:",
            ["Dashboard", "Nouveau Modèle", "Mes Modèles", "Monitoring", "Datasets", "Modèles Déployés"]
        )
        
        st.divider()
        
        # Paramètres avancés
        show_advanced_settings()
        
        st.divider()
        
        # Monitoring en temps réel des ressources
        if st.session_state.get("show_system_metrics", True):
            st.subheader("Ressources Système")
            display_resource_monitor()
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID: `{st.session_state.user_id[:8]}...`")
        
        # Statut API
        health, error = call_ai_api("/health")
        if health:
            st.success("🟢 API Training en ligne")
            active_jobs = health.get("active_jobs", 0)
            st.metric("Jobs actifs", active_jobs)
            system_load = health.get("system_load", 0)
            st.metric("Charge système", f"{system_load:.1f}%")
        else:
            st.error("🔴 API Training hors ligne")
        
        # Graphique de performance système
        if st.session_state.get("show_system_metrics", True):
            st.subheader("Performance")
            perf_chart = create_resource_usage_chart()
            st.plotly_chart(perf_chart, use_container_width=True)
    
    # Pages principales
    if page == "Dashboard":
        show_dashboard()
    elif page == "Nouveau Modèle":
        show_new_model_page()
    elif page == "Mes Modèles":
        show_my_models_page()
    elif page == "Monitoring":
        show_monitoring_page()
    elif page == "Datasets":
        show_datasets_page()
    elif page == "Modèles Déployés":
        show_deployed_models_page()
    
    # Footer avec informations de la plateforme
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**AI Training Platform**")
        st.markdown("Entraînement d'IA de pointe")
    
    with col2:
        st.markdown("**Frameworks supportés:**")
        st.markdown("- Scikit-learn")
        st.markdown("- PyTorch")
        st.markdown("- TensorFlow") 
        st.markdown("- XGBoost")
    
    with col3:
        st.markdown("**Support:**")
        st.markdown("Version 1.0.0")
        st.markdown(f"Dernière MAJ: {datetime.now().strftime('%Y-%m-%d')}")
        if health:
            st.markdown("API Status: ✅ En ligne")

def get_job_details_realtime(job_id):
    """Récupère les détails d'un job en temps réel"""
    try:
        # Récupérer le statut
        status, error = call_ai_api(f"/training/job/{job_id}/status")
        if error:
            return None, error
        
        # Récupérer les métriques
        metrics_data, _ = call_ai_api(f"/training/job/{job_id}/metrics")
        
        return {
            "status": status,
            "metrics": metrics_data.get("metrics", []) if metrics_data else []
        }, None
        
    except Exception as e:
        return None, str(e)

def format_job_duration(start_time, end_time=None):
    """Formate la durée d'un job"""
    try:
        if not start_time:
            return "N/A"
        
        from datetime import datetime
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        if end_time:
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        else:
            end = datetime.now()
        
        duration = end - start
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        if duration.days > 0:
            return f"{duration.days}j {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    except Exception:
        return "N/A"

def create_training_progress_chart(metrics_data):
    """Crée un graphique de progression d'entraînement"""
    if not metrics_data or len(metrics_data) < 2:
        return None
    
    df = pd.DataFrame(metrics_data)
    
    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Accuracy", "Learning Rate", "Métriques Additionnelles"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss
    if 'train_loss' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_loss'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
    
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_loss'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
    
    # Accuracy
    if 'train_accuracy' in df.columns and df['train_accuracy'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_accuracy'], 
                      name='Train Accuracy', line=dict(color='green')),
            row=1, col=2
        )
    
    if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_accuracy'], 
                      name='Val Accuracy', line=dict(color='orange')),
            row=1, col=2
        )
    
    # Learning Rate
    if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['learning_rate'], 
                      name='Learning Rate', line=dict(color='purple')),
            row=2, col=1
        )
    
    # Métriques additionnelles (gradient norm, etc.)
    if len(df.columns) > 6:  # Si on a des métriques supplémentaires
        additional_cols = [col for col in df.columns if col not in 
                          ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rate', 'timestamp']]
        
        for i, col in enumerate(additional_cols[:3]):  # Limiter à 3
            if df[col].notna().any():
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df[col], 
                              name=col, line=dict(color=px.colors.qualitative.Set1[i])),
                    row=2, col=2
                )
    
    fig.update_layout(height=600, title_text="Progression de l'Entraînement")
    return fig

def export_training_results(job_data):
    """Exporte les résultats d'entraînement"""
    try:
        # Préparer les données pour export
        export_data = {
            "job_info": {
                "job_id": job_data.get("job_id"),
                "job_name": job_data.get("job_name"),
                "algorithm": job_data.get("algorithm"),
                "model_type": job_data.get("model_type"),
                "status": job_data.get("status"),
                "created_at": job_data.get("created_at"),
                "completed_at": job_data.get("completed_at")
            },
            "final_metrics": job_data.get("final_metrics", {}),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Ajouter les métriques d'entraînement si disponibles
        if "training_metrics" in job_data:
            export_data["training_metrics"] = job_data["training_metrics"]
        
        return json.dumps(export_data, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Erreur export: {str(e)}"}, indent=2)

def validate_hyperparameters(algorithm, hyperparams):
    """Valide les hyperparamètres selon l'algorithme"""
    validation_errors = []
    
    try:
        if algorithm == "random_forest":
            if "n_estimators" in hyperparams:
                if not isinstance(hyperparams["n_estimators"], int) or hyperparams["n_estimators"] < 1:
                    validation_errors.append("n_estimators doit être un entier positif")
            
            if "max_depth" in hyperparams:
                if hyperparams["max_depth"] is not None and hyperparams["max_depth"] < 1:
                    validation_errors.append("max_depth doit être positif ou None")
        
        elif algorithm == "neural_network":
            if "hidden_sizes" in hyperparams:
                if not isinstance(hyperparams["hidden_sizes"], list):
                    validation_errors.append("hidden_sizes doit être une liste")
                elif any(size < 1 for size in hyperparams["hidden_sizes"]):
                    validation_errors.append("Toutes les tailles de couches doivent être positives")
            
            if "learning_rate" in hyperparams:
                if not 0 < hyperparams["learning_rate"] <= 1:
                    validation_errors.append("learning_rate doit être entre 0 et 1")
        
        elif algorithm == "xgboost":
            if "n_estimators" in hyperparams:
                if hyperparams["n_estimators"] < 1:
                    validation_errors.append("n_estimators doit être positif")
            
            if "learning_rate" in hyperparams:
                if not 0 < hyperparams["learning_rate"] <= 1:
                    validation_errors.append("learning_rate doit être entre 0 et 1")
        
        return validation_errors
        
    except Exception as e:
        return [f"Erreur validation: {str(e)}"]

def estimate_training_time(dataset_size, algorithm, hyperparams):
    """Estime le temps d'entraînement"""
    try:
        # Estimations basiques (en minutes)
        base_times = {
            "sklearn": {
                "random_forest": 0.1,
                "logistic_regression": 0.05,
                "svm": 0.5,
                "neural_network": 2.0
            },
            "pytorch": {
                "neural_network": 5.0
            },
            "tensorflow": {
                "neural_network": 4.0
            },
            "xgboost": {
                "xgboost": 1.0
            }
        }
        
        # Facteurs multiplicateurs
        size_factor = max(1, dataset_size / 10000)  # Base: 10k échantillons
        
        # Facteur algorithme
        algo_category = "sklearn"
        if "pytorch" in str(algorithm).lower():
            algo_category = "pytorch"
        elif "tensorflow" in str(algorithm).lower():
            algo_category = "tensorflow"
        elif "xgboost" in str(algorithm).lower():
            algo_category = "xgboost"
        
        base_time = base_times.get(algo_category, {}).get(algorithm, 2.0)
        
        # Facteur hyperparamètres
        hyperparams_factor = 1.0
        if "epochs" in hyperparams:
            hyperparams_factor *= hyperparams["epochs"] / 100
        if "n_estimators" in hyperparams:
            hyperparams_factor *= hyperparams["n_estimators"] / 100
        
        estimated_time = base_time * size_factor * hyperparams_factor
        
        # Convertir en format lisible
        if estimated_time < 1:
            return f"{int(estimated_time * 60)} secondes"
        elif estimated_time < 60:
            return f"{int(estimated_time)} minutes"
        else:
            hours = int(estimated_time // 60)
            minutes = int(estimated_time % 60)
            return f"{hours}h {minutes}m"
            
    except Exception:
        return "Estimation non disponible"


def create_resource_usage_chart():
    """Crée un graphique d'utilisation des ressources"""
    # Générer des données de test pour les ressources système
    timestamps = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                              end=datetime.now(), 
                              freq='30s')
    
    # Simuler des données de charge système
    cpu_data = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
    memory_data = 45 + 15 * np.cos(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))
    
    # S'assurer que les valeurs sont dans des limites réalistes
    cpu_data = np.clip(cpu_data, 0, 100)
    memory_data = np.clip(memory_data, 0, 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_data,
        mode='lines',
        name='CPU %',
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_data,
        mode='lines',
        name='Memory %',
        line=dict(color='#FF7043', width=2)
    ))
    
    fig.update_layout(
        title='Utilisation Ressources (30min)',
        xaxis_title='Temps',
        yaxis_title='Utilisation %',
        height=300,
        showlegend=True,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def debug_dataset_structure(datasets):
    """Fonction de debug pour vérifier la structure des datasets"""
    st.subheader("🔍 Debug - Structure des Datasets")
    
    if not datasets:
        st.warning("Aucun dataset disponible")
        return False
    
    st.write(f"Nombre de datasets trouvés: {len(datasets)}")
    
    for i, dataset in enumerate(datasets):
        with st.expander(f"Dataset {i+1}: {dataset.get('name', 'Sans nom')}"):
            st.write("**Clés disponibles:**")
            for key in dataset.keys():
                st.write(f"- {key}: {type(dataset[key])}")
            
            st.write("**Contenu complet:**")
            st.json(dataset)
    
    # Vérifier si file_path existe dans tous les datasets
    missing_file_path = [d.get('name', f'Dataset {i}') for i, d in enumerate(datasets) if 'file_path' not in d]
    
    if missing_file_path:
        st.error(f"⚠️ Datasets sans 'file_path': {', '.join(missing_file_path)}")
        return False
    else:
        """Affiche les statistiques détaillées d'un dataset"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informations Générales")
                st.metric("Lignes", f"{dataset['rows_count']:,}")
                st.metric("Colonnes", dataset['columns_count'])
                st.metric("Taille", f"{dataset['size_mb']:.1f} MB")
                
                if dataset.get('target_columns'):
                    st.write("**Colonnes cibles possibles:**")
                    for col in dataset['target_columns']:
                        st.code(col)
            
            with col2:
                st.subheader("Métadonnées")
                st.write(f"**Type:** {dataset['dataset_type']}")
                st.write(f"**Créé le:** {dataset['created_at'][:10]}")
                st.write(f"**Public:** {'Oui' if dataset['is_public'] else 'Non'}")
                
                # Estimation de la complexité
                complexity_score = min(dataset['rows_count'] * dataset['columns_count'] / 1000000, 10)
                if complexity_score < 1:
                    complexity = "Faible"
                    color = "green"
                elif complexity_score < 5:
                    complexity = "Modérée"
                    color = "orange"
                else:
                    complexity = "Élevée"
                    color = "red"
                
                st.markdown(f"**Complexité:** <span style='color: {color}'>{complexity}</span>", 
                        unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Erreur affichage statistiques: {e}")
        st.success("✅ Tous les datasets ont une clé 'file_path'")
        return True
    
def show_new_model_page():
    """Page de création d'un nouveau modèle"""
    st.title("Créer un Nouveau Modèle IA")
    
    st.markdown("""
    <div class="training-card">
        <h3>🚀 Configuration de l'Entraînement</h3>
        <p>Configurez votre modèle d'IA avec nos algorithmes de pointe</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Récupérer les données nécessaires
    datasets = get_available_datasets()
    algorithms = get_available_algorithms()
    
    # Debug temporaire - à retirer une fois le problème résolu
    if st.checkbox("🔍 Mode Debug - Afficher structure datasets", value=False):
        debug_dataset_structure(datasets)
    
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration Générale")
            
            job_name = st.text_input("Nom du Modèle:", placeholder="Mon super modèle IA")
            
            # Sélection du dataset
            if datasets:
                dataset_options = {f"{d['name']} ({d['rows_count']} lignes)": d for d in datasets}
                selected_dataset_name = st.selectbox("Dataset:", list(dataset_options.keys()))
                selected_dataset = dataset_options[selected_dataset_name]
            else:
                st.error("Aucun dataset disponible. Ajoutez-en dans la section Datasets.")
                return
            
            # Type de tâche
            task_type = st.selectbox("Type de Tâche:", ["classification", "regression", "clustering"])
            
            # Sélection de la colonne cible
            if task_type != "clustering" and selected_dataset:
                target_columns = selected_dataset.get("target_columns", [])
                if target_columns:
                    target_column = st.selectbox("Colonne Cible:", target_columns)
                else:
                    target_column = st.text_input("Colonne Cible:", placeholder="target")
            else:
                target_column = None
        
        with col2:
            st.subheader("Algorithme et Framework")
            
            # Sélection du framework
            model_type = st.selectbox("Framework:", ["sklearn", "pytorch", "tensorflow", "xgboost"])
            
            # Sélection de l'algorithme selon le framework et la tâche
            if model_type in algorithms and task_type in algorithms[model_type]:
                algorithm_options = algorithms[model_type][task_type]
                algorithm_names = [alg["display_name"] for alg in algorithm_options]
                selected_algo_name = st.selectbox("Algorithme:", algorithm_names)
                
                # Trouver l'algorithme sélectionné
                selected_algo = next(alg for alg in algorithm_options if alg["display_name"] == selected_algo_name)
                algorithm = selected_algo["name"]
            else:
                st.error(f"Aucun algorithme disponible pour {model_type} - {task_type}")
                return
        
        st.divider()
        st.subheader("Hyperparamètres")
        
        # Configuration des hyperparamètres dynamique
        hyperparams = {}
        
        if selected_algo:
            hyperparams_list = selected_algo.get("hyperparams", [])
            
            if hyperparams_list:
                cols = st.columns(min(3, len(hyperparams_list)))
                
                for i, param in enumerate(hyperparams_list):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="hyperparameter-section">
                            <h4>{param}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if param in ["n_estimators", "max_depth", "epochs"]:
                            hyperparams[param] = st.number_input(
                                f"{param}:", 
                                min_value=1, 
                                max_value=1000, 
                                value=100 if param != "max_depth" else 10,
                                key=f"hp_{param}"
                            )
                        elif param in ["learning_rate", "C"]:
                            hyperparams[param] = st.number_input(
                                f"{param}:", 
                                min_value=0.0001, 
                                max_value=10.0, 
                                value=0.001 if param == "learning_rate" else 1.0,
                                step=0.0001,
                                format="%.4f",
                                key=f"hp_{param}"
                            )
                        elif param == "batch_size":
                            hyperparams[param] = st.selectbox(
                                f"{param}:", 
                                [16, 32, 64, 128, 256],
                                index=1,
                                key=f"hp_{param}"
                            )
                        elif param in ["kernel", "penalty", "solver"]:
                            if param == "kernel":
                                hyperparams[param] = st.selectbox(
                                    f"{param}:", 
                                    ["rbf", "linear", "poly"],
                                    key=f"hp_{param}"
                                )
                            elif param == "penalty":
                                hyperparams[param] = st.selectbox(
                                    f"{param}:", 
                                    ["l2", "l1", "elasticnet"],
                                    key=f"hp_{param}"
                                )
                            elif param == "solver":
                                hyperparams[param] = st.selectbox(
                                    f"{param}:", 
                                    ["lbfgs", "saga", "adam"],
                                    key=f"hp_{param}"
                                )
                        elif param == "hidden_sizes":
                            sizes_text = st.text_input(
                                "Hidden Sizes (séparés par des virgules):", 
                                value="128,64",
                                key=f"hp_{param}"
                            )
                            hyperparams[param] = [int(x.strip()) for x in sizes_text.split(",")]
                        elif param == "hidden_layers":
                            hyperparams[param] = [st.number_input(
                                f"Couche {i+1}:", 
                                min_value=1, 
                                max_value=1000, 
                                value=128 if i == 0 else 64,
                                key=f"hp_layer_{i}"
                            ) for i in range(2)]
                        else:
                            # Paramètre générique
                            hyperparams[param] = st.text_input(
                                f"{param}:", 
                                value="auto",
                                key=f"hp_{param}"
                            )
        
        st.divider()
        st.subheader("Configuration Avancée")
        
        col1, col2 = st.columns(2)
        with col1:
            validation_split = st.slider("Split Validation:", 0.1, 0.5, 0.2, 0.05)
            early_stopping = st.checkbox("Early Stopping", value=True)
        with col2:
            save_checkpoints = st.checkbox("Sauvegarder Checkpoints", value=True)
            enable_monitoring = st.checkbox("Monitoring Temps Réel", value=True)
        
        training_config = {
            "validation_split": validation_split,
            "early_stopping": early_stopping,
            "save_checkpoints": save_checkpoints,
            "enable_monitoring": enable_monitoring
        }
        
        # Bouton de soumission
        submitted = st.form_submit_button(
            "🚀 Lancer l'Entraînement", 
            type="primary",
            use_container_width=True
        )

        if submitted:
            if not job_name:
                st.error("Veuillez donner un nom à votre modèle")
                return
            
            # Préparer la requête
            training_request = {
                "job_name": job_name,
                "user_id": st.session_state.user_id,
                "model_type": model_type,
                "algorithm": algorithm,
                "dataset_path": selected_dataset.get("file_path", ""),
                "target_column": target_column,
                "task_type": task_type,
                "hyperparameters": hyperparams,
                "training_config": training_config
            }
            
            # Vérifier que le dataset path existe
            if not training_request["dataset_path"]:
                st.error("Erreur: Chemin du dataset manquant. Veuillez réessayer ou sélectionner un autre dataset.")
                st.write("Debug - Structure du dataset sélectionné:")
                st.json(selected_dataset)
                return
            
            # Lancer l'entraînement
            with st.spinner("Lancement de l'entraînement..."):
                result, error = call_ai_api("/training/start", method="POST", data=training_request)
                
                if result:
                    st.success(f"✅ Entraînement lancé! Job ID: {result['job_id']}")
                    st.balloons()
                    st.info("Consultez la section 'Monitoring' pour suivre le progrès en temps réel.")
                    
                    # Auto-redirect vers monitoring
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Erreur: {error}")
                    
def show_my_models_page():
    """Page de gestion des modèles"""
    st.title("Mes Modèles d'IA")
    
    jobs = get_training_jobs(st.session_state.user_id)
    
    if not jobs:
        st.info("Aucun modèle créé. Créez votre premier modèle!")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Filtrer par statut:", ["Tous", "running", "completed", "failed", "queued"])
    with col2:
        model_type_filter = st.selectbox("Filtrer par framework:", ["Tous"] + list(set([j["model_type"] for j in jobs])))
    with col3:
        task_type_filter = st.selectbox("Filtrer par tâche:", ["Tous"] + list(set([j["task_type"] for j in jobs])))
    
    # Appliquer les filtres
    filtered_jobs = jobs
    if status_filter != "Tous":
        filtered_jobs = [j for j in filtered_jobs if j["status"] == status_filter]
    if model_type_filter != "Tous":
        filtered_jobs = [j for j in filtered_jobs if j["model_type"] == model_type_filter]
    if task_type_filter != "Tous":
        filtered_jobs = [j for j in filtered_jobs if j["task_type"] == task_type_filter]
    
    # Affichage des modèles
    for job in filtered_jobs:
        with st.expander(f"🤖 {job['job_name']} ({job['status']})", expanded=job["status"] == "running"):
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Framework:** {job['model_type']}")
                st.write(f"**Algorithme:** {job['algorithm']}")
                st.write(f"**Type de tâche:** {job['task_type']}")
                st.write(f"**Créé le:** {job['created_at']}")
                if job.get('completed_at'):
                    st.write(f"**Terminé le:** {job['completed_at']}")
            
            with col2:
                # Statut et progrès
                status = job["status"]
                progress = job.get("progress", 0)
                
                if status == "running":
                    st.markdown('<span class="status-running">🔄 En cours</span>', unsafe_allow_html=True)
                    st.progress(progress / 100.0)
                    st.write(f"Progrès: {progress:.1f}%")
                elif status == "completed":
                    st.markdown('<span class="status-completed">✅ Terminé</span>', unsafe_allow_html=True)
                    st.progress(1.0)
                elif status == "failed":
                    st.markdown('<span class="status-failed">❌ Échoué</span>', unsafe_allow_html=True)
                else:
                    st.write(f"Status: {status}")
            
            with col3:
                # Actions
                if status == "completed":
                    if st.button("📊 Métriques", key=f"metrics_{job['job_id']}"):
                        show_model_metrics(job['job_id'], job['final_metrics'])
                    
                    if st.button("🚀 Déployer", key=f"deploy_{job['job_id']}"):
                        show_deployment_form(job['job_id'])
                        
                elif status == "running":
                    if st.button("⏹️ Arrêter", key=f"stop_{job['job_id']}"):
                        stop_training_job(job['job_id'])
                
                if st.button("👁️ Monitoring", key=f"monitor_{job['job_id']}"):
                    show_detailed_monitoring(job['job_id'])

def show_model_metrics(job_id, final_metrics):
    """Affiche les métriques d'un modèle"""
    st.subheader(f"Métriques du Modèle {job_id[:8]}...")
    
    if final_metrics:
        cols = st.columns(len(final_metrics))
        for i, (metric, value) in enumerate(final_metrics.items()):
            with cols[i % len(cols)]:
                st.metric(metric.replace("_", " ").title(), f"{value:.4f}")
    
    # Récupérer l'historique des métriques
    metrics_history, error = call_ai_api(f"/training/job/{job_id}/metrics")
    
    if metrics_history and metrics_history.get("metrics"):
        metrics_data = pd.DataFrame(metrics_history["metrics"])
        
        if not metrics_data.empty:
            # Graphique des losses
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Loss", "Accuracy", "Learning Rate", "Résumé"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # Loss
            if 'train_loss' in metrics_data.columns:
                fig.add_trace(
                    go.Scatter(x=metrics_data['epoch'], y=metrics_data['train_loss'], 
                              name='Train Loss', line=dict(color='blue')),
                    row=1, col=1
                )
            
            if 'val_loss' in metrics_data.columns and metrics_data['val_loss'].notna().any():
                fig.add_trace(
                    go.Scatter(x=metrics_data['epoch'], y=metrics_data['val_loss'], 
                              name='Val Loss', line=dict(color='red')),
                    row=1, col=1
                )
            
            # Accuracy
            if 'train_accuracy' in metrics_data.columns and metrics_data['train_accuracy'].notna().any():
                fig.add_trace(
                    go.Scatter(x=metrics_data['epoch'], y=metrics_data['train_accuracy'], 
                              name='Train Accuracy', line=dict(color='green')),
                    row=1, col=2
                )
            
            if 'val_accuracy' in metrics_data.columns and metrics_data['val_accuracy'].notna().any():
                fig.add_trace(
                    go.Scatter(x=metrics_data['epoch'], y=metrics_data['val_accuracy'], 
                              name='Val Accuracy', line=dict(color='orange')),
                    row=1, col=2
                )
            
            # Learning Rate
            if 'learning_rate' in metrics_data.columns and metrics_data['learning_rate'].notna().any():
                fig.add_trace(
                    go.Scatter(x=metrics_data['epoch'], y=metrics_data['learning_rate'], 
                              name='Learning Rate', line=dict(color='purple')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title_text="Évolution des Métriques d'Entraînement")
            st.plotly_chart(fig, use_container_width=True)

def show_deployment_form(job_id):
    """Formulaire de déploiement d'un modèle"""
    st.subheader("Déployer le Modèle")
    
    with st.form("deploy_form"):
        deployment_name = st.text_input("Nom du déploiement:", placeholder=f"deployment_{job_id[:8]}")
        
        col1, col2 = st.columns(2)
        with col1:
            auto_scaling = st.checkbox("Auto-scaling", value=True)
            max_requests = st.number_input("Max requêtes/min:", min_value=1, max_value=10000, value=100)
        
        with col2:
            monitoring_enabled = st.checkbox("Monitoring activé", value=True)
            cache_enabled = st.checkbox("Cache des prédictions", value=True)
        
        endpoint_config = {
            "auto_scaling": auto_scaling,
            "max_requests": max_requests,
            "monitoring_enabled": monitoring_enabled,
            "cache_enabled": cache_enabled
        }
        
        if st.form_submit_button("🚀 Déployer", type="primary"):
            deployment_request = {
                "model_id": job_id,
                "user_id": st.session_state.user_id,
                "deployment_name": deployment_name,
                "endpoint_config": endpoint_config
            }
            
            with st.spinner("Déploiement en cours..."):
                result, error = call_ai_api("/models/deploy", method="POST", data=deployment_request)
                
                if result:
                    st.success(f"✅ Modèle déployé! Endpoint: {result['endpoint_url']}")
                    st.info("Votre modèle est maintenant accessible via API.")
                else:
                    st.error(f"❌ Erreur de déploiement: {error}")

def stop_training_job(job_id):
    """Arrête un job d'entraînement"""
    with st.spinner("Arrêt de l'entraînement..."):
        result, error = call_ai_api(
            f"/training/job/{job_id}/stop", 
            method="POST", 
            data={"user_id": st.session_state.user_id}
        )
        
        if result:
            st.success("✅ Entraînement arrêté")
        else:
            st.error(f"❌ Erreur: {error}")

def show_monitoring_page():
    """Page de monitoring en temps réel"""
    st.title("Monitoring d'Entraînement")
    
    # Jobs en cours
    jobs = get_training_jobs(st.session_state.user_id)
    running_jobs = [j for j in jobs if j["status"] == "running"]
    
    if not running_jobs:
        st.info("Aucun entraînement en cours.")
        return
    
    # Sélection du job à monitorer
    job_names = {f"{j['job_name']} ({j['job_id'][:8]}...)": j['job_id'] for j in running_jobs}
    selected_job_name = st.selectbox("Sélectionner un job à monitorer:", list(job_names.keys()))
    selected_job_id = job_names[selected_job_name]
    
    # Monitoring en temps réel
    st.subheader(f"Monitoring: {selected_job_name}")
    
    # Placeholder pour les mises à jour en temps réel
    metrics_placeholder = st.empty()
    logs_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (5s)", value=True):
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        if time.time() - st.session_state.last_refresh > 5:
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    show_detailed_monitoring(selected_job_id, metrics_placeholder, logs_placeholder, chart_placeholder)

def show_detailed_monitoring(job_id, metrics_placeholder=None, logs_placeholder=None, chart_placeholder=None):
    """Affichage détaillé du monitoring"""
    
    # Récupérer le statut
    status, error = call_ai_api(f"/training/job/{job_id}/status")
    
    if status:
        # Métriques actuelles
        if metrics_placeholder:
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", status.get("status", "N/A"))
                with col2:
                    progress = status.get("progress", 0)
                    st.metric("Progrès", f"{progress:.1f}%")
                with col3:
                    current_epoch = status.get("current_epoch", 0)
                    total_epochs = status.get("total_epochs", 0)
                    st.metric("Époque", f"{current_epoch}/{total_epochs}" if total_epochs > 0 else "N/A")
        
        # Logs en temps réel
        if logs_placeholder:
            logs = status.get("logs", [])
            if logs:
                with logs_placeholder.container():
                    st.subheader("Logs d'Entraînement")
                    log_text = "\n".join(logs[-10:])  # Derniers 10 logs
                    st.text_area("", value=log_text, height=200, disabled=True)
        
        # Graphique des métriques
        if chart_placeholder:
            metrics_history, _ = call_ai_api(f"/training/job/{job_id}/metrics")
            
            if metrics_history and metrics_history.get("metrics"):
                metrics_data = pd.DataFrame(metrics_history["metrics"])
                
                if not metrics_data.empty and len(metrics_data) > 1:
                    with chart_placeholder.container():
                        st.subheader("Évolution des Métriques")
                        
                        # Graphique interactif
                        fig = go.Figure()
                        
                        if 'train_loss' in metrics_data.columns:
                            fig.add_trace(go.Scatter(
                                x=metrics_data['epoch'],
                                y=metrics_data['train_loss'],
                                mode='lines+markers',
                                name='Train Loss',
                                line=dict(color='blue')
                            ))
                        
                        if 'val_loss' in metrics_data.columns and metrics_data['val_loss'].notna().any():
                            fig.add_trace(go.Scatter(
                                x=metrics_data['epoch'],
                                y=metrics_data['val_loss'],
                                mode='lines+markers',
                                name='Val Loss',
                                line=dict(color='red')
                            ))
                        
                        fig.update_layout(
                            title='Loss en Temps Réel',
                            xaxis_title='Époque',
                            yaxis_title='Loss',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def show_datasets_page():
    """Page de gestion des datasets"""
    st.title("Gestion des Datasets")
    
    tab1, tab2 = st.tabs(["Datasets Disponibles", "Ajouter Dataset"])
    
    with tab1:
        st.subheader("Datasets Publics")
        
        datasets = get_available_datasets()
        
        if datasets:
            for dataset in datasets:
                with st.expander(f"📊 {dataset['name']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {dataset['description']}")
                        st.write(f"**Type:** {dataset['dataset_type']}")
                        st.write(f"**Lignes:** {dataset['rows_count']:,}")
                        st.write(f"**Colonnes:** {dataset['columns_count']}")
                        st.write(f"**Taille:** {dataset['size_mb']:.1f} MB")
                    
                    with col2:
                        st.write(f"**Créé le:** {dataset['created_at'][:10]}")
                        st.write(f"**Public:** {'Oui' if dataset['is_public'] else 'Non'}")
                        
                        # Colonnes cibles potentielles
                        target_cols = dataset.get('target_columns', [])
                        if target_cols:
                            st.write("**Cibles possibles:**")
                            for col in target_cols[:3]:  # Limiter à 3
                                st.code(col)
                    
                    # Bouton de prévisualisation
                    if st.button(f"Prévisualiser", key=f"preview_{dataset['dataset_id']}"):
                        show_dataset_preview(dataset)
        else:
            st.info("Chargement des datasets par défaut...")
            st.info("Si aucun dataset n'apparaît après quelques secondes, redémarrez l'API.")
    
    with tab2:
        st.subheader("Ajouter un Nouveau Dataset")
        
        # Créer le formulaire avec st.form
        with st.form("upload_dataset_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_name = st.text_input("Nom du Dataset:", placeholder="Mon Dataset")
                description = st.text_area("Description:", placeholder="Description du dataset...")
                
            with col2:
                uploaded_file = st.file_uploader("Fichier Dataset:", type=['csv', 'json', 'pkl'])
                is_public = st.checkbox("Rendre public", value=True)
            
            # IMPORTANT: Bouton submit à l'intérieur du formulaire
            submitted = st.form_submit_button("Ajouter Dataset", type="primary")
            
            if submitted:
                if uploaded_file and dataset_name:
                    # Sauvegarder le fichier temporairement
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Appeler l'API pour ajouter le dataset
                    result, error = call_ai_api("/datasets/upload", method="POST", data={
                        "dataset_name": dataset_name,
                        "description": description,
                        "user_id": st.session_state.user_id,
                        "file_path": temp_path,
                        "is_public": is_public
                    })
                    
                    if result:
                        st.success("Dataset ajouté avec succès!")
                        st.json(result)
                        # Attendre un peu puis recharger
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Erreur: {error}")
                    
                    # Nettoyer le fichier temporaire
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                else:
                    st.error("Veuillez fournir un nom et un fichier.")

def show_dataset_preview(dataset):
    """Affiche une prévisualisation du dataset"""
    st.subheader(f"Prévisualisation: {dataset['name']}")
    
    try:
        # Dans un cas réel, on chargerait le dataset via l'API
        # Ici on simule avec des données d'exemple
        st.info("Aperçu des données (simulation)")
        
        # Créer des données d'exemple basées sur les métadonnées
        rows = min(dataset['rows_count'], 100)  # Limiter l'aperçu
        cols = dataset['columns_count']
        
        # Générer des données simulées
        sample_data = {}
        for i in range(min(cols, 10)):  # Limiter à 10 colonnes pour l'aperçu
            col_name = f"column_{i+1}"
            if col_name in dataset.get('target_columns', []):
                col_name = f"target_{i+1}"
            
            # Générer des données selon le type de colonne
            if 'target' in col_name:
                sample_data[col_name] = np.random.choice(['A', 'B', 'C'], rows)
            else:
                sample_data[col_name] = np.random.randn(rows)
        
        df_preview = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_preview.head(20), use_container_width=True)
        
        with col2:
            st.subheader("Statistiques")
            st.write(f"**Shape:** {df_preview.shape}")
            st.write("**Types de données:**")
            for col, dtype in df_preview.dtypes.items():
                st.write(f"- {col}: {dtype}")
            
            # Graphique de distribution pour colonnes numériques
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Colonne à visualiser:", numeric_cols)
                
                fig = px.histogram(df_preview, x=selected_col, title=f"Distribution de {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur lors de la prévisualisation: {e}")

def show_deployed_models_page():
    """Page des modèles déployés"""
    st.title("Modèles Déployés")
    
    # Récupérer les modèles déployés
    deployed_models, error = call_ai_api(f"/models/deployed/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur: {error}")
        return
    
    models = deployed_models.get("deployed_models", []) if deployed_models else []
    
    if not models:
        st.info("Aucun modèle déployé. Déployez vos modèles depuis 'Mes Modèles'.")
        return
    
    # Métriques des déploiements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modèles Déployés", len(models))
    
    with col2:
        total_usage = sum([m.get("usage_count", 0) for m in models])
        st.metric("Utilisations Totales", total_usage)
    
    with col3:
        active_models = len([m for m in models if m.get("status", "active") == "active"])
        st.metric("Modèles Actifs", active_models)
    
    st.divider()
    
    # Liste des modèles déployés
    for model in models:
        with st.expander(f"🚀 {model['deployment_name']}", expanded=False):
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Nom Original:** {model['original_job_name']}")
                st.write(f"**Algorithme:** {model['algorithm']}")
                st.write(f"**Endpoint:** `{model['endpoint_url']}`")
                st.write(f"**Déployé le:** {model['deployed_at']}")
            
            with col2:
                st.metric("Utilisations", model['usage_count'])
                if model.get('last_used'):
                    st.write(f"**Dernière utilisation:** {model['last_used'][:16]}")
                else:
                    st.write("**Jamais utilisé**")
            
            with col3:
                # Testeur de modèle
                if st.button(f"Tester", key=f"test_{model['model_id']}"):
                    show_model_tester(model)
                
                # Arrêter le déploiement
                if st.button(f"Arrêter", key=f"stop_deploy_{model['model_id']}", type="secondary"):
                    if st.confirm("Êtes-vous sûr de vouloir arrêter ce déploiement?"):
                        stop_deployment(model['model_id'])

def show_model_tester(model):
    """Interface de test pour un modèle déployé"""
    st.subheader(f"Testeur: {model['deployment_name']}")
    
    # Interface simple pour tester les prédictions
    st.write("Entrez des valeurs pour tester votre modèle:")
    
    # Simulation d'interface de test
    # Dans la réalité, on récupérerait la signature du modèle
    
    with st.form(f"test_form_{model['model_id']}"):
        st.write("Format des données d'entrée (exemple):")
        
        # Interface générique pour les features
        num_features = st.number_input("Nombre de features:", min_value=1, max_value=100, value=10)
        
        features = []
        cols = st.columns(min(5, num_features))
        
        for i in range(num_features):
            with cols[i % len(cols)]:
                feature_value = st.number_input(
                    f"Feature {i+1}:", 
                    value=0.0,
                    key=f"feature_{i}_{model['model_id']}"
                )
                features.append(feature_value)
        
        if st.form_submit_button("Prédire", type="primary"):
            # Appeler l'API de prédiction
            prediction_data = {"features": features}
            
            with st.spinner("Prédiction en cours..."):
                result, error = call_ai_api(
                    f"/api/predict/{model['model_id']}", 
                    method="POST", 
                    data=prediction_data
                )
                
                if result:
                    st.success("Prédiction réussie!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Résultat")
                        prediction = result['prediction']
                        if isinstance(prediction[0], (list, np.ndarray)):
                            st.write(f"Prédiction: {prediction[0]}")
                        else:
                            st.write(f"Prédiction: {prediction}")
                    
                    with col2:
                        if 'probabilities' in result:
                            st.subheader("Probabilités")
                            probs = result['probabilities'][0]
                            for i, prob in enumerate(probs):
                                st.write(f"Classe {i}: {prob:.3f}")
                    
                    # Afficher la réponse complète
                    with st.expander("Réponse complète"):
                        st.json(result)
                        
                else:
                    st.error(f"Erreur de prédiction: {error}")

def stop_deployment(model_id):
    """Arrête un déploiement"""
    # Ici on implémenterait l'arrêt du déploiement
    st.success(f"Déploiement {model_id[:8]}... arrêté")
    st.rerun()

def show_job_details(job_id):
    """Affiche les détails complets d'un job"""
    st.subheader(f"Détails du Job {job_id[:8]}...")
    
    # Récupérer les détails
    status, error = call_ai_api(f"/training/job/{job_id}/status")
    
    if status:
        tabs = st.tabs(["Vue d'ensemble", "Métriques", "Logs", "Configuration"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Status:** {status['status']}")
                st.write(f"**Progrès:** {status['progress']:.1f}%")
            with col2:
                if status.get('current_epoch'):
                    st.write(f"**Époque actuelle:** {status['current_epoch']}")
                if status.get('total_epochs'):
                    st.write(f"**Total époques:** {status['total_epochs']}")
        
        with tabs[1]:
            show_model_metrics(job_id, status.get('metrics', {}))
        
        with tabs[2]:
            logs = status.get('logs', [])
            if logs:
                st.text_area("Logs d'entraînement:", value='\n'.join(logs), height=300)
            else:
                st.info("Aucun log disponible")
        
        with tabs[3]:
            st.info("Configuration du modèle (à implémenter)")

# Fonction pour créer des graphiques de performance système
def create_system_performance_chart():
    """Crée un graphique de performance système"""
    # Simuler des données de performance
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                              end=datetime.now(), 
                              freq='1min')
    
    cpu_data = np.random.normal(45, 15, len(timestamps))
    memory_data = np.random.normal(60, 10, len(timestamps))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_data,
        mode='lines',
        name='CPU %',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_data,
        mode='lines',
        name='Memory %',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Performance Système (1h)',
        xaxis_title='Temps',
        yaxis_title='Utilisation %',
        height=400
    )
    
    return fig

# Interface de configuration avancée
def show_advanced_settings():
    """Affiche les paramètres avancés"""
    with st.sidebar.expander("⚙️ Paramètres Avancés"):
        
        st.subheader("API")
        api_timeout = st.slider("Timeout API (s)", 10, 120, 30)
        st.session_state.api_timeout = api_timeout
        
        st.subheader("Interface")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        st.session_state.auto_refresh = auto_refresh
        
        refresh_interval = st.slider("Intervalle refresh (s)", 1, 30, 5)
        st.session_state.refresh_interval = refresh_interval
        
        st.subheader("Monitoring")
        enable_websocket = st.checkbox("WebSocket temps réel", value=True)
        st.session_state.enable_websocket = enable_websocket
        
        show_system_metrics = st.checkbox("Métriques système", value=True)
        st.session_state.show_system_metrics = show_system_metrics
        
        if st.button("Réinitialiser Cache"):
            st.cache_data.clear()
            st.success("Cache vidé!")

# Interface principale mise à jour
def main():
    st.markdown('<h1 class="main-header">🤖 AI Training Platform</h1>', unsafe_allow_html=True)
    
    # Configuration WebSocket
    setup_websocket()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Sections:",
            ["Dashboard", "Nouveau Modèle", "Mes Modèles", "Monitoring", "Datasets", "Modèles Déployés"]
        )
        
        st.divider()
        
        # Paramètres avancés
        show_advanced_settings()
        
        st.divider()
        
        # Monitoring en temps réel des ressources
        if st.session_state.get("show_system_metrics", True):
            st.subheader("Ressources Système")
            display_resource_monitor()
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID: `{st.session_state.user_id[:8]}...`")
        
        # Statut API
        health, error = call_ai_api("/health")
        if health:
            st.success("🟢 API Training en ligne")
            active_jobs = health.get("active_jobs", 0)
            st.metric("Jobs actifs", active_jobs)
            system_load = health.get("system_load", 0)
            st.metric("Charge système", f"{system_load:.1f}%")
        else:
            st.error("🔴 API Training hors ligne")
        
        # Graphique de performance système
        if st.session_state.get("show_system_metrics", True):
            st.subheader("Performance")
            perf_chart = create_system_performance_chart()
            st.plotly_chart(perf_chart, use_container_width=True)
    
    # Pages principales
    if page == "Dashboard":
        show_dashboard()
    elif page == "Nouveau Modèle":
        show_new_model_page()
    elif page == "Mes Modèles":
        show_my_models_page()
    elif page == "Monitoring":
        show_monitoring_page()
    elif page == "Datasets":
        show_datasets_page()
    elif page == "Modèles Déployés":
        show_deployed_models_page()
    
    # Footer avec informations de la plateforme
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**AI Training Platform**")
        st.markdown("Entraînement d'IA de pointe")
    
    with col2:
        st.markdown("**Frameworks supportés:**")
        st.markdown("- Scikit-learn")
        st.markdown("- PyTorch")
        st.markdown("- TensorFlow") 
        st.markdown("- XGBoost")
    
    with col3:
        st.markdown("**Support:**")
        st.markdown("Version 1.0.0")
        st.markdown(f"Dernière MAJ: {datetime.now().strftime('%Y-%m-%d')}")
        if health:
            uptime = health.get("uptime", "N/A")
            st.markdown(f"Uptime: {uptime}")

if __name__ == "__main__":
    main()
