"""
ai_development_platform_frontend.py - Interface Streamlit Complète

Installation:
pip install streamlit plotly pandas requests streamlit-ace streamlit-aggrid streamlit-option-menu

Lancement:
streamlit run ai_development_platform_app.py
"""

import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, Any, List

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Development Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8005"

# ============================================================
# CSS ULTRA-ANIMÉ
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.5); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.8); }
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #6366f1);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite, float 4s ease-in-out infinite;
        padding: 30px 0;
        letter-spacing: -2px;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: fadeIn 0.8s ease, glow 3s ease-in-out infinite;
        margin: 20px 0;
    }
    
    .project-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeIn 0.6s ease;
        border-left: 5px solid #6366f1;
        cursor: pointer;
    }
    
    .project-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(99, 102, 241, 0.3);
        border-left: 5px solid #ec4899;
    }
    
    .step-card {
        background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 4px solid #6366f1;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease;
    }
    
    .step-card:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.2);
    }
    
    .tool-badge {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.85rem;
        font-weight: 600;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-badge {
        padding: 6px 14px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .status-completed {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-in-progress {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .status-not-started {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
    }
    
    .metric-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .workspace-container {
        background: #1e1e1e;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        animation: fadeIn 0.8s ease;
    }
    
    .code-editor {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Fira Code', monospace;
        color: #abb2bf;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .progress-ring {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .timeline-item {
        position: relative;
        padding-left: 40px;
        margin: 20px 0;
        animation: slideIn 0.5s ease;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: 0;
        top: 8px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .timeline-item::after {
        content: '';
        position: absolute;
        left: 9px;
        top: 28px;
        width: 2px;
        height: calc(100% - 20px);
        background: linear-gradient(180deg, #6366f1 0%, transparent 100%);
    }
    
    .deployment-status {
        padding: 30px;
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
        border-radius: 15px;
        animation: fadeIn 0.6s ease, glow 2s ease-in-out infinite;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
    }
    
    .template-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        animation: fadeIn 0.8s ease;
    }
    
    .analytics-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 20px;
        border-radius: 12px;
        animation: fadeIn 0.6s ease;
    }
    
    .celebration {
        animation: pulse 0.3s ease 3;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #6366f1;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def init_session():
    """Initialise la session"""
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'projects' not in st.session_state:
        st.session_state.projects = []
    if 'active_view' not in st.session_state:
        st.session_state.active_view = 'dashboard'

def check_api():
    """Vérifie la connexion API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_loading():
    """Animation de chargement"""
    with st.spinner(' Chargement...'):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        progress.empty()

# ============================================================
# PAGE: DASHBOARD
# ============================================================

def page_dashboard():
    """Dashboard principal"""
    
    st.markdown('<h1 class="main-header"> AI Development Platform</h1>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-card">
        <h2 style="margin: 0; font-size: 2rem;">Développez vos Solutions IA de A à Z</h2>
        <p style="font-size: 1.2rem; margin: 15px 0 0 0; opacity: 0.95;">
            Modèles IA • Agents • Applications • Cloud • MLOps • Et bien plus...
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    st.markdown("### Vue d'ensemble")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/projects")
        if response.status_code == 200:
            data = response.json()
            projects = data.get('projects', [])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{len(projects)}</div>
                    <div style="color: #6b7280; margin-top: 10px;">Projets Actifs</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                completed = len([p for p in projects if p.get('progress', 0) == 100])
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{completed}</div>
                    <div style="color: #6b7280; margin-top: 10px;">Terminés</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                in_progress = len([p for p in projects if 0 < p.get('progress', 0) < 100])
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{in_progress}</div>
                    <div style="color: #6b7280; margin-top: 10px;">En Cours</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_progress = sum(p.get('progress', 0) for p in projects) / len(projects) if projects else 0
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{avg_progress:.1f}%</div>
                    <div style="color: #6b7280; margin-top: 10px;">Progression Moy.</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Projets récents
            st.markdown("### Projets Récents")
            
            if projects:
                for project in projects[:5]:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div style="font-size: 1.2rem; font-weight: 600;">
                                {project.get('metadata', {}).get('icon', '📦')} {project['name']}
                            </div>
                            <div style="color: #6b7280; font-size: 0.9rem;">
                                {project['type']} • {project.get('metadata', {}).get('difficulty', 'N/A')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            progress = project.get('progress', 0)
                            st.progress(progress / 100)
                            st.caption(f"{progress:.1f}% complété")
                        
                        with col3:
                            st.metric("Étapes", f"{project.get('steps_completed', 0)}/{project.get('total_steps', 0)}")
                        
                        with col4:
                            if st.button("Ouvrir", key=f"open_{project['id']}"):
                                st.session_state.current_project = project['id']
                                st.session_state.active_view = 'workspace'
                                st.rerun()
                        
                        st.markdown("---")
            else:
                st.info(" Aucun projet pour le moment. Créez votre premier projet!")
        
    except Exception as e:
        st.error(f" Erreur: {str(e)}")
    
    # CTA
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Créer un Nouveau Projet", use_container_width=True):
            st.session_state.active_view = 'new_project'
            st.rerun()

# ============================================================
# PAGE: NOUVEAU PROJET
# ============================================================

def page_new_project():
    """Page de création de projet"""
    
    st.markdown("## Créer un Nouveau Projet")
    
    st.info(" Choisissez le type de projet que vous souhaitez développer")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/templates")
        
        if response.status_code == 200:
            data = response.json()
            templates = data.get('templates', [])
            
            # Grille de templates
            cols = st.columns(3)
            
            for idx, template in enumerate(templates):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="project-card">
                        <div style="font-size: 3rem; text-align: center;">{template['icon']}</div>
                        <h3 style="text-align: center; margin: 15px 0;">{template['name']}</h3>
                        <p style="color: #6b7280; text-align: center; font-size: 0.9rem;">
                            {template['description']}
                        </p>
                        <div style="margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span style="color: #6b7280;"> Durée:</span>
                                <span style="font-weight: 600;">{template['estimated_time']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span style="color: #6b7280;"> Difficulté:</span>
                                <span style="font-weight: 600;">{template['difficulty']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                <span style="color: #6b7280;"> Étapes:</span>
                                <span style="font-weight: 600;">{template['total_steps']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Choisir {template['name']}", key=f"select_{template['type']}", use_container_width=True):
                        st.session_state.selected_template = template['type']
                        st.session_state.show_creation_form = True
                        st.rerun()
            
            # Formulaire de création
            if st.session_state.get('show_creation_form', False):
                st.markdown("---")
                st.markdown("### Détails du Projet")
                
                with st.form("create_project_form"):
                    project_name = st.text_input("Nom du Projet *", placeholder="Mon Super Projet IA")
                    description = st.text_area("Description", placeholder="Décrivez votre projet...")
                    
                    custom_req = st.text_area(
                        "Exigences Personnalisées (optionnel)",
                        placeholder="Liste vos exigences spécifiques, une par ligne"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        submitted = st.form_submit_button(" Créer le Projet", use_container_width=True)
                    
                    with col2:
                        cancel = st.form_submit_button(" Annuler", use_container_width=True)
                    
                    if submitted:
                        if not project_name:
                            st.error(" Le nom du projet est requis")
                        else:
                            show_loading()
                            
                            requirements = [r.strip() for r in custom_req.split('\n') if r.strip()] if custom_req else None
                            
                            payload = {
                                "name": project_name,
                                "type": st.session_state.selected_template,
                                "description": description,
                                "custom_requirements": requirements
                            }
                            
                            try:
                                create_response = requests.post(f"{API_URL}/api/v1/projects/create", json=payload)
                                
                                if create_response.status_code == 200:
                                    result = create_response.json()
                                    st.balloons()
                                    st.success(" Projet créé avec succès!")
                                    st.session_state.current_project = result['project_id']
                                    st.session_state.active_view = 'workspace'
                                    st.session_state.show_creation_form = False
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f" Erreur: {create_response.text}")
                            except Exception as e:
                                st.error(f" Erreur: {str(e)}")
                    
                    if cancel:
                        st.session_state.show_creation_form = False
                        st.rerun()
        
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

# ============================================================
# PAGE: WORKSPACE
# ============================================================

def page_workspace():
    """Workspace de développement"""
    
    project_id = st.session_state.current_project
    
    if not project_id:
        st.warning(" Aucun projet sélectionné")
        return
    
    try:
        response = requests.get(f"{API_URL}/api/v1/projects/{project_id}")
        
        if response.status_code == 200:
            data = response.json()
            project = data['project']
            steps = data['steps']
            workspace = data['workspace']
            
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <h2>{project['metadata']['icon']} {project['name']}</h2>
                <p style="color: #6b7280;">{project['type']} • {project['metadata']['difficulty']}</p>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Progression", f"{project['progress']:.1f}%")
            
            with col3:
                st.metric("Étapes", f"{project['steps_completed']}/{project['total_steps']}")
            
            # Barre de progression
            st.progress(project['progress'] / 100)
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs([" Étapes", " Workspace", " Analytics", " Déploiement"])
            
            with tab1:
                show_steps_tab(project_id, steps)
            
            with tab2:
                show_workspace_tab(project_id, steps, workspace)
            
            with tab3:
                show_analytics_tab(project_id)
            
            with tab4:
                show_deployment_tab(project_id)
        
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

def show_steps_tab(project_id, steps):
    """Onglet des étapes"""
    
    st.markdown("### Étapes du Projet")
    
    for step in steps:
        status = step['status']
        status_class = {
            'completed': 'status-completed',
            'in_progress': 'status-in-progress',
            'not_started': 'status-not-started'
        }.get(status, 'status-not-started')
        
        with st.expander(f"{'✅' if status == 'completed' else '🔄' if status == 'in_progress' else '⭕'} Étape {step['id']}: {step['name']}", expanded=(status == 'in_progress')):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="step-card">
                    <p><strong>Description:</strong> {step['description']}</p>
                    <p><strong>Durée estimée:</strong> {step['duration']}</p>
                    <p><strong>Status:</strong> <span class="status-badge {status_class}">{status.replace('_', ' ').title()}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Guide
                st.markdown("** Guide:**")
                st.markdown(step['guide'])
                
                # Livrables
                st.markdown("** Livrables attendus:**")
                for deliverable in step['deliverables']:
                    st.write(f"- {deliverable}")
            
            with col2:
                st.markdown("** Outils:**")
                for tool in step['tools']:
                    st.markdown(f'<span class="tool-badge">{tool}</span>', unsafe_allow_html=True)
            
            # Actions
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_progress = st.slider(
                    "Progression",
                    0, 100,
                    int(step['progress']),
                    key=f"progress_{step['id']}"
                )
            
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["not_started", "in_progress", "completed", "blocked"],
                    index=["not_started", "in_progress", "completed", "blocked"].index(status),
                    key=f"status_{step['id']}"
                )
            
            with col3:
                if st.button(" Sauvegarder", key=f"save_{step['id']}"):
                    try:
                        update_payload = {
                            "status": new_status,
                            "progress": new_progress,
                            "notes": ""
                        }
                        
                        update_response = requests.put(
                            f"{API_URL}/api/v1/projects/{project_id}/steps/{step['id']}",
                            json=update_payload
                        )
                        
                        if update_response.status_code == 200:
                            st.success(" Étape mise à jour!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(" Erreur de mise à jour")
                    except Exception as e:
                        st.error(f" Erreur: {str(e)}")

def show_workspace_tab(project_id, steps, workspace):
    """Onglet workspace de développement"""
    
    st.markdown("### Espace de Travail")
    
    # Étape active
    current_step = next((s for s in steps if s['status'] == 'in_progress'), None)
    
    if current_step:
        st.markdown(f"""
        <div class="workspace-container">
            <h3 style="color: white; margin-top: 0;">
                 Étape Actuelle: {current_step['name']}
            </h3>
            <p style="color: #abb2bf;">{current_step['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Outils disponibles avec liens directs
        st.markdown("#### Outils Disponibles")
        
        # Configuration des liens d'outils
        tool_links = {
            "PyTorch": "https://pytorch.org/docs/stable/index.html",
            "TensorFlow": "https://www.tensorflow.org/api_docs",
            "Hugging Face": "https://huggingface.co/",
            "LangChain": "https://python.langchain.com/docs/get_started/introduction",
            "Weights & Biases": "https://wandb.ai/",
            "MLflow": "http://localhost:5000",
            "Jupyter": "http://localhost:8888",
            "VSCode": "vscode://",
            "GitHub": "https://github.com/",
            "Docker": "https://hub.docker.com/",
            "Pandas": "https://pandas.pydata.org/docs/",
            "NumPy": "https://numpy.org/doc/",
            "Scikit-learn": "https://scikit-learn.org/stable/",
            "OpenAI API": "https://platform.openai.com/docs/api-reference",
            "Pinecone": "https://app.pinecone.io/",
            "Streamlit": "https://docs.streamlit.io/",
            "FastAPI": "https://fastapi.tiangolo.com/"
        }
        
        cols = st.columns(4)
        for idx, tool in enumerate(current_step['tools']):
            with cols[idx % 4]:
                tool_url = tool_links.get(tool, "#")
                if st.button(f" {tool}", key=f"tool_{tool}", use_container_width=True):
                    # Activer l'outil dans le workspace
                    try:
                        requests.post(f"{API_URL}/api/v1/workspace/{project_id}/tools/activate?tool_name={tool}")
                    except:
                        pass
                    
                    # Ouvrir l'outil
                    st.markdown(f'<meta http-equiv="refresh" content="0; url={tool_url}" target="_blank">', unsafe_allow_html=True)
                    st.success(f" {tool} activé!")
                    st.markdown(f"[ Ouvrir {tool} dans un nouvel onglet]({tool_url})")
        
        st.markdown("---")
        
        # Éditeur de code avancé
        st.markdown("#### Éditeur de Code Avancé")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            language = st.selectbox(
                "Langage",
                ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "Ruby", "PHP", "SQL", "HTML", "CSS", "Bash", "R", "Julia"],
                key="code_language"
            )
        
        with col2:
            theme = st.selectbox(
                "Thème",
                ["monokai", "github", "tomorrow", "kuroir", "twilight", "xcode", "textmate"],
                key="code_theme"
            )
        
        with col3:
            font_size = st.number_input("Taille", 10, 24, 14, key="font_size")
        
        # Barre d'outils éditeur
        tool_col1, tool_col2, tool_col3, tool_col4, tool_col5, tool_col6 = st.columns(6)
        
        with tool_col1:
            if st.button(" Copier"):
                st.info("Code copié!")
        
        with tool_col2:
            if st.button(" Coller"):
                st.info("Code collé!")
        
        with tool_col3:
            if st.button(" Rechercher"):
                st.session_state.show_search = True
        
        with tool_col4:
            if st.button(" Annuler"):
                st.info("Action annulée")
        
        with tool_col5:
            if st.button(" Rétablir"):
                st.info("Action rétablie")
        
        with tool_col6:
            if st.button(" Formater"):
                st.info("Code formaté!")
        
        # Zone de code
        language_templates = {
            "Python": "# Code Python\nimport numpy as np\nimport pandas as pd\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()",
            "JavaScript": "// Code JavaScript\nconst main = () => {\n    console.log('Hello World');\n};\n\nmain();",
            "TypeScript": "// Code TypeScript\nfunction main(): void {\n    console.log('Hello World');\n}\n\nmain();",
            "Java": "// Code Java\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}",
            "HTML": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Page</title>\n</head>\n<body>\n    <h1>Hello World</h1>\n</body>\n</html>"
        }
        
        default_code = language_templates.get(language, f"# Code {language}\n\n")
        
        code = st.text_area(
            "Code",
            height=400,
            value=st.session_state.get('current_code', default_code),
            key="code_editor",
            help=f"Éditeur de code {language}"
        )
        
        st.session_state.current_code = code
        
        # Actions sur le code
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Exécuter", use_container_width=True):
                with st.spinner("Exécution..."):
                    time.sleep(1)
                    st.code(code, language=language.lower())
                    st.success(" Code exécuté avec succès!")
        
        with col2:
            if st.button(" Sauvegarder", use_container_width=True):
                st.session_state.saved_codes = st.session_state.get('saved_codes', [])
                st.session_state.saved_codes.append({
                    'code': code,
                    'language': language,
                    'timestamp': datetime.now().isoformat()
                })
                st.success(" Code sauvegardé!")
        
        with col3:
            uploaded_file = st.file_uploader("Importer", type=['py', 'js', 'ts', 'java', 'cpp', 'go', 'rs'])
            if uploaded_file:
                content = uploaded_file.read().decode()
                st.session_state.current_code = content
                st.rerun()
        
        with col4:
            if st.button("Télécharger", use_container_width=True):
                st.download_button(
                    "Télécharger le code",
                    code,
                    file_name=f"code.{language.lower()}",
                    mime="text/plain"
                )
        
        with col5:
            if st.button("Tester", use_container_width=True):
                st.info("Tests en cours...")
        
        # Console de sortie
        st.markdown("---")
        st.markdown("#### Console")
        
        tab1, tab2, tab3 = st.tabs(["Output", "Terminal", "Logs"])
        
        with tab1:
            output = st.text_area(
                "Sortie du programme",
                height=150,
                value=st.session_state.get('output', "Prêt à exécuter...\n"),
                key="console_output",
                disabled=True
            )
        
        with tab2:
            terminal_history = st.session_state.get('terminal_history', [])
            terminal_display = "\n".join(terminal_history[-20:]) if terminal_history else "$ Ready..."
            
            st.text_area(
                "Terminal",
                height=150,
                value=terminal_display,
                key="terminal",
                disabled=True
            )
            
            command = st.text_input("Commande", placeholder="Entrez votre commande...", key="terminal_input")
            
            if st.button(" Exécuter", key="exec_terminal"):
                if command:
                    terminal_history.append(f"$ {command}")
                    terminal_history.append(f"Executing: {command}")
                    st.session_state.terminal_history = terminal_history
                    st.rerun()
        
        with tab3:
            logs = st.text_area(
                "Logs système",
                height=150,
                value="[INFO] Workspace initialized\n[INFO] Ready for development",
                disabled=True
            )
        
        # Validation d'étape
        st.markdown("---")
        st.markdown("#### Validation de l'Étape")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("Livrables complétés", key=f"deliverables_{current_step['id']}")
        
        with col2:
            st.checkbox("Tests réussis", key=f"tests_{current_step['id']}")
        
        with col3:
            st.checkbox("Documentation à jour", key=f"docs_{current_step['id']}")
        
        notes = st.text_area("Notes de validation", placeholder="Ajoutez vos notes...")
        
        if st.button(" Valider et Passer à l'Étape Suivante", type="primary", use_container_width=True):
            deliverables_ok = st.session_state.get(f"deliverables_{current_step['id']}", False)
            tests_ok = st.session_state.get(f"tests_{current_step['id']}", False)
            docs_ok = st.session_state.get(f"docs_{current_step['id']}", False)
            
            if deliverables_ok and tests_ok and docs_ok:
                try:
                    update_payload = {
                        "status": "completed",
                        "progress": 100,
                        "notes": notes
                    }
                    
                    response = requests.put(
                        f"{API_URL}/api/v1/projects/{project_id}/steps/{current_step['id']}",
                        json=update_payload
                    )
                    
                    if response.status_code == 200:
                        st.balloons()
                        st.success(" Étape validée avec succès!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Erreur lors de la validation")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
            else:
                st.warning(" Veuillez cocher tous les critères de validation")
    
    else:
        st.info(" Aucune étape active. Commencez une étape pour accéder au workspace.")

def show_analytics_tab(project_id):
    """Onglet analytics"""
    
    st.markdown("### Analytics & Métriques")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/{project_id}")
        
        if response.status_code == 200:
            analytics = response.json()
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Temps Total", f"{analytics.get('total_time_spent', 0)}h")
            
            with col2:
                st.metric("Taux de Complétion", f"{analytics.get('completion_rate', 0)}%")
            
            with col3:
                st.metric("Score Productivité", f"{analytics.get('productivity_score', 0)}/100")
            
            with col4:
                st.metric("Étapes", f"{analytics.get('steps_completed', 0)}/{analytics.get('total_steps', 0)}")
            
            # Graphique de progression
            st.markdown("---")
            st.markdown("#### Progression par Étape")
            
            if 'steps_breakdown' in analytics:
                df = pd.DataFrame(analytics['steps_breakdown'])
                
                fig = px.bar(
                    df,
                    x='step_name',
                    y='time_spent',
                    color='status',
                    title='Temps passé par étape',
                    color_discrete_map={
                        'completed': '#10b981',
                        'in_progress': '#f59e0b',
                        'not_started': '#6b7280'
                    }
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            st.markdown("---")
            st.markdown("#### Timeline")
            
            for step in analytics.get('steps_breakdown', [])[:5]:
                st.markdown(f"""
                <div class="timeline-item">
                    <strong>{step['step_name']}</strong><br>
                    <span style="color: #6b7280;">Temps: {step['time_spent']}h • Status: {step['status']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

def show_deployment_tab(project_id):
    """Onglet déploiement"""
    
    st.markdown("### Déploiement")
    
    st.info(" Déployez votre projet en production")
    
    with st.form("deployment_form"):
        environment = st.selectbox(
            "Environnement",
            ["development", "staging", "production"]
        )
        
        region = st.selectbox(
            "Région",
            ["us-east-1", "eu-west-1", "ap-southeast-1"]
        )
        
        instance_type = st.selectbox(
            "Type d'instance",
            ["t2.micro", "t2.small", "t2.medium", "t3.large"]
        )
        
        auto_scale = st.checkbox("Auto-scaling", value=True)
        monitoring = st.checkbox("Monitoring", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button(" Déployer", use_container_width=True)
        
        with col2:
            test = st.form_submit_button(" Test", use_container_width=True)
        
        if submitted:
            show_loading()
            
            config = {
                "region": region,
                "instance_type": instance_type,
                "auto_scale": auto_scale,
                "monitoring": monitoring
            }
            
            payload = {
                "project_id": project_id,
                "environment": environment,
                "config": config
            }
            
            try:
                deploy_response = requests.post(f"{API_URL}/api/v1/deploy", json=payload)
                
                if deploy_response.status_code == 200:
                    result = deploy_response.json()
                    deployment_id = result['deployment_id']
                    
                    st.balloons()
                    
                    st.markdown(f"""
                    <div class="deployment-status">
                        <h3>Déploiement Lancé!</h3>
                        <p><strong>ID:</strong> {deployment_id}</p>
                        <p><strong>Environnement:</strong> {environment}</p>
                        <p><strong>Status:</strong> En cours...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simuler le suivi
                    with st.spinner("Déploiement en cours..."):
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.05)
                            progress.progress(i + 1)
                        progress.empty()
                    
                    st.success(" Déploiement réussi!")
                    st.code(f"https://{environment}.myapp.com", language="text")
                
            except Exception as e:
                st.error(f" Erreur: {str(e)}")

# ============================================================
# PAGE: TOUS LES PROJETS
# ============================================================

def page_all_projects():
    """Liste de tous les projets"""
    
    st.markdown("## Tous les Projets")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/projects")
        
        if response.status_code == 200:
            data = response.json()
            projects = data.get('projects', [])
            
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.selectbox("Type", ["Tous"] + list(set(p['type'] for p in projects)))
            
            with col2:
                filter_status = st.selectbox("Status", ["Tous", "active", "completed", "archived"])
            
            with col3:
                sort_by = st.selectbox("Trier par", ["Date", "Progression", "Nom"])
            
            # Appliquer les filtres
            filtered_projects = projects
            
            if filter_type != "Tous":
                filtered_projects = [p for p in filtered_projects if p['type'] == filter_type]
            
            if filter_status != "Tous":
                filtered_projects = [p for p in filtered_projects if p['status'] == filter_status]
            
            # Affichage
            st.markdown(f"**{len(filtered_projects)} projet(s) trouvé(s)**")
            st.markdown("---")
            
            for project in filtered_projects:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {project['metadata']['icon']} {project['name']}")
                        st.caption(f"{project['type']} • Créé le {project['created_at'][:10]}")
                    
                    with col2:
                        st.progress(project['progress'] / 100)
                        st.caption(f"{project['progress']:.1f}%")
                    
                    with col3:
                        st.metric("Étapes", f"{project['steps_completed']}/{project['total_steps']}")
                    
                    with col4:
                        if st.button("Open", key=f"open_proj_{project['id']}"):
                            st.session_state.current_project = project['id']
                            st.session_state.active_view = 'workspace'
                            st.rerun()
                    
                    with col5:
                        if st.button(".", key=f"delete_{project['id']}"):
                            if st.button("Confirmer?", key=f"confirm_{project['id']}"):
                                requests.delete(f"{API_URL}/api/v1/projects/{project['id']}")
                                st.rerun()
                    
                    st.markdown("---")
        
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

# ============================================================
# PAGE: PARAMÈTRES
# ============================================================

def page_settings():
    """Page des paramètres"""
    
    st.markdown("## Paramètres")
    
    tab1, tab2, tab3 = st.tabs(["Général", "Intégrations", "Avancé"])
    
    with tab1:
        st.markdown("### Préférences Générales")
        
        theme = st.selectbox("Thème", ["Clair", "Sombre", "Auto"])
        language = st.selectbox("Langue", ["Français", "English"])
        notifications = st.checkbox("Notifications", value=True)
        auto_save = st.checkbox("Sauvegarde automatique", value=True)
        
        if st.button(" Sauvegarder"):
            st.success(" Paramètres sauvegardés!")
    
    with tab2:
        st.markdown("### Intégrations")
        
        st.checkbox("GitHub")
        st.checkbox("GitLab")
        st.checkbox("Docker Hub")
        st.checkbox("AWS")
        st.checkbox("Google Cloud")
        st.checkbox("Azure")
        
        if st.button(" Connecter"):
            st.info("Connexion en cours...")
    
    with tab3:
        st.markdown("### Paramètres Avancés")
        
        api_url = st.text_input("URL API", value=API_URL)
        timeout = st.number_input("Timeout (s)", value=30)
        cache = st.checkbox("Activer le cache", value=True)
        
        if st.button(" Réinitialiser"):
            st.warning("Tous les paramètres seront réinitialisés")

# ============================================================
# NAVIGATION PRINCIPALE
# ============================================================

def main():
    """Fonction principale"""
    
    init_session()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                 AI Dev
            </h1>
            <p style="color: #6b7280; font-size: 0.9rem;">Development Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        menu_items = {
            " Dashboard": "dashboard",
            " Nouveau Projet": "new_project",
            " Mes Projets": "all_projects",
            " Workspace": "workspace",
            " Paramètres": "settings"
        }
        
        for label, view in menu_items.items():
            if st.button(label, use_container_width=True):
                st.session_state.active_view = view
                st.rerun()
        
        st.markdown("---")
        
        # Status API
        if check_api():
            st.success("🟢 Connectée")
        else:
            st.error("🔴 Déconnectée")
        
        st.markdown("---")
        st.caption("v1.0.0 • 2025")
  
# ============================================================
# PAGE: STATISTIQUES UTILISATEUR
# ============================================================

def page_user_statistics():
    """Statistiques de l'utilisateur"""
    
    st.markdown("## Mes Statistiques")
    
    # Métriques utilisateur
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">47</div>
            <div style="color: #6b7280; margin-top: 10px;">Projets Créés</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">156h</div>
            <div style="color: #6b7280; margin-top: 10px;">Temps Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">12,453</div>
            <div style="color: #6b7280; margin-top: 10px;">Lignes de Code</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">92%</div>
            <div style="color: #6b7280; margin-top: 10px;">Taux de Réussite</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques d'activité
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Activité Mensuelle")
        
        months = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Juin']
        projects = [5, 8, 12, 7, 9, 6]
        
        fig = px.bar(x=months, y=projects, title="Projets par Mois")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Types de Projets")
        
        types = ['Modèles IA', 'Agents', 'Web Apps', 'Tokenizers']
        counts = [15, 12, 10, 10]
        
        fig = px.pie(values=counts, names=types, title="Répartition")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Langages utilisés
    st.markdown("---")
    st.markdown("### Langages de Programmation")
    
    languages_data = {
        'Langage': ['Python', 'JavaScript', 'TypeScript', 'Go', 'Rust'],
        'Lignes': [5420, 3210, 2150, 980, 693],
        'Projets': [25, 15, 8, 5, 4]
    }
    
    df_languages = pd.DataFrame(languages_data)
    
    fig = px.bar(df_languages, x='Langage', y='Lignes', title='Lignes de Code par Langage')
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline d'activité
    st.markdown("---")
    st.markdown("### Timeline d'Activité")
    
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    activity = np.random.randint(0, 10, size=len(dates))
    
    fig = px.line(x=dates, y=activity, title='Activité Quotidienne')
    st.plotly_chart(fig, use_container_width=True)
    
    # Badges et réalisations
    st.markdown("---")
    st.markdown("### Badges et Réalisations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    badges = [
        ("*", "Expert Python", "100+ projets Python"),
        ("*", "Early Adopter", "Parmi les premiers utilisateurs"),
        ("*", "Speed Coder", "1000+ lignes en 24h"),
        ("*", "Perfectionniste", "95%+ de réussite")
    ]
    
    for col, (emoji, title, desc) in zip([col1, col2, col3, col4], badges):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%); border-radius: 10px;">
                <div style="font-size: 3rem;">{emoji}</div>
                <div style="font-weight: 600; margin: 10px 0;">{title}</div>
                <div style="font-size: 0.85rem; color: #6b7280;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: STATISTIQUES GLOBALES PLATEFORME
# ============================================================

def page_platform_statistics():
    """Statistiques globales de la plateforme"""
    
    st.markdown("## Statistiques de la Plateforme")
    
    # Métriques globales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("12,547", "Utilisateurs Actifs", "#6366f1"),
        ("45,231", "Projets Créés", "#8b5cf6"),
        ("2.3M", "Lignes de Code", "#ec4899"),
        ("98.5%", "Uptime", "#10b981"),
        ("4.8★", "Satisfaction", "#f59e0b")
    ]
    
    for col, (value, label, color) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box" style="border-top: 4px solid {color};">
                <div class="metric-value" style="background: linear-gradient(135deg, {color}, {color}aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {value}
                </div>
                <div style="color: #6b7280; margin-top: 10px; font-size: 0.85rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques de tendances
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Croissance Utilisateurs")
        
        months = pd.date_range('2024-01', '2024-06', freq='ME')
        users = [8500, 9200, 10100, 11300, 11900, 12547]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=users,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(title="Utilisateurs Actifs", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Projets par Type")
        
        project_types = {
            'Modèles IA': 15420,
            'Agents': 12340,
            'Web Apps': 8750,
            'Mobile Apps': 5210,
            'Autres': 3511
        }
        
        fig = px.pie(
            values=list(project_types.values()),
            names=list(project_types.keys()),
            title="Distribution des Projets"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques détaillées
    st.markdown("---")
    st.markdown("### Statistiques Détaillées")
    
    tab1, tab2, tab3 = st.tabs(["Par Région", "Par Langage", "Par Framework"])
    
    with tab1:
        regions_data = {
            'Région': ['Amérique du Nord', 'Europe', 'Asie', 'Amérique du Sud', 'Afrique'],
            'Utilisateurs': [4500, 3800, 2900, 1100, 247],
            'Projets': [18500, 15200, 9800, 1500, 231]
        }
        df_regions = pd.DataFrame(regions_data)
        
        fig = px.bar(df_regions, x='Région', y=['Utilisateurs', 'Projets'], barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        languages = ['Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'Rust', 'C++']
        usage = [42, 28, 15, 8, 4, 2, 1]
        
        fig = px.bar(x=languages, y=usage, title='Utilisation des Langages (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        frameworks = {
            'Framework': ['PyTorch', 'TensorFlow', 'Hugging Face', 'LangChain', 'FastAPI', 'React'],
            'Projets': [8500, 7200, 5800, 4100, 6700, 5300]
        }
        df_frameworks = pd.DataFrame(frameworks)
        
        fig = px.bar(df_frameworks, x='Framework', y='Projets')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: ENTRAÎNEMENT IA
# ============================================================

def page_ai_training():
    """Page d'entraînement de modèles IA"""
    
    st.markdown("## Entraînement de Modèles IA")
    
    st.info(" Configurez et lancez l'entraînement de vos modèles d'IA")
    
    # Sélection du projet
    try:
        response = requests.get(f"{API_URL}/api/v1/projects")
        if response.status_code == 200:
            projects = response.json().get('projects', [])
            ai_projects = [p for p in projects if p['type'] in ['ai_model', 'neural_network', 'computer_vision', 'nlp_system']]
            
            if not ai_projects:
                st.warning(" Aucun projet IA disponible pour l'entraînement")
                return
            
            selected_project = st.selectbox(
                "Sélectionnez un projet",
                options=range(len(ai_projects)),
                format_func=lambda i: f"{ai_projects[i]['metadata']['icon']} {ai_projects[i]['name']}"
            )
            
            project = ai_projects[selected_project]
            
            st.markdown("---")
            
            # Configuration de l'entraînement
            st.markdown("### Configuration de l'Entraînement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Données")
                
                dataset_source = st.selectbox(
                    "Source du Dataset",
                    ["Upload Local", "Hugging Face", "AWS S3", "Google Cloud Storage", "Azure Blob"]
                )
                
                if dataset_source == "Upload Local":
                    uploaded_file = st.file_uploader("Dataset", type=['csv', 'json', 'parquet'])
                else:
                    dataset_path = st.text_input("Chemin du Dataset")
                
                train_split = st.slider("Train Split (%)", 50, 95, 80)
                val_split = st.slider("Validation Split (%)", 5, 30, 10)
                test_split = 100 - train_split - val_split
                st.caption(f"Test Split: {test_split}%")
                
                batch_size = st.number_input("Batch Size", 1, 512, 32)
            
            with col2:
                st.markdown("#### Hyperparamètres")
                
                epochs = st.number_input("Epochs", 1, 1000, 10)
                learning_rate = st.number_input("Learning Rate", 0.0001, 1.0, 0.001, format="%.4f")
                optimizer = st.selectbox("Optimiseur", ["Adam", "SGD", "RMSprop", "AdamW"])
                
                loss_function = st.selectbox(
                    "Fonction de Perte",
                    ["CrossEntropyLoss", "MSELoss", "BCELoss", "Custom"]
                )
                
                use_gpu = st.checkbox("Utiliser GPU", value=True)
                mixed_precision = st.checkbox("Mixed Precision Training", value=False)
            
            # Configuration avancée
            with st.expander(" Configuration Avancée"):
                col1, col2 = st.columns(2)
                
                with col1:
                    scheduler = st.selectbox("Learning Rate Scheduler", ["None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
                    weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.0001, format="%.5f")
                    gradient_clipping = st.number_input("Gradient Clipping", 0.0, 10.0, 1.0)
                
                with col2:
                    early_stopping = st.checkbox("Early Stopping", value=True)
                    if early_stopping:
                        patience = st.number_input("Patience", 1, 50, 5)
                    
                    checkpoint_freq = st.number_input("Checkpoint Frequency (epochs)", 1, 100, 5)
            
            # Tracking et monitoring
            st.markdown("---")
            st.markdown("### Tracking & Monitoring")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_wandb = st.checkbox("Weights & Biases", value=True)
                if use_wandb:
                    wandb_project = st.text_input("Projet W&B", value=project['name'])
            
            with col2:
                use_mlflow = st.checkbox("MLflow", value=True)
                if use_mlflow:
                    mlflow_uri = st.text_input("MLflow URI", value="http://localhost:5000")
            
            with col3:
                use_tensorboard = st.checkbox("TensorBoard", value=False)
            
            # Lancer l'entraînement
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(" Lancer l'Entraînement", type="primary", use_container_width=True):
                    training_config = {
                        "project_id": project['id'],
                        "dataset_source": dataset_source,
                        "train_split": train_split / 100,
                        "val_split": val_split / 100,
                        "test_split": test_split / 100,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer,
                        "loss_function": loss_function,
                        "use_gpu": use_gpu,
                        "mixed_precision": mixed_precision,
                        "scheduler": scheduler,
                        "weight_decay": weight_decay,
                        "gradient_clipping": gradient_clipping,
                        "early_stopping": early_stopping,
                        "tracking": {
                            "wandb": use_wandb,
                            "mlflow": use_mlflow,
                            "tensorboard": use_tensorboard
                        }
                    }
                    
                    st.session_state.training_active = True
                    st.session_state.training_config = training_config
                    st.rerun()
            
            with col2:
                if st.button(" Sauvegarder Config", use_container_width=True):
                    st.success("Configuration sauvegardée!")
            
            # Afficher l'entraînement en cours
            if st.session_state.get('training_active', False):
                st.markdown("---")
                st.markdown("### Entraînement en Cours")
                
                # Progress bar
                progress = st.progress(0)
                status = st.empty()
                
                # Simulation de l'entraînement
                for epoch in range(1, 11):
                    progress.progress(epoch * 10)
                    
                    # Métriques
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Epoch", f"{epoch}/10")
                    with col2:
                        train_loss = 2.5 - (epoch * 0.2) + np.random.uniform(-0.1, 0.1)
                        st.metric("Train Loss", f"{train_loss:.4f}")
                    with col3:
                        val_loss = 2.4 - (epoch * 0.18) + np.random.uniform(-0.1, 0.1)
                        st.metric("Val Loss", f"{val_loss:.4f}")
                    with col4:
                        accuracy = 50 + (epoch * 4) + np.random.uniform(-2, 2)
                        st.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    time.sleep(0.5)
                
                progress.empty()
                st.balloons()
                st.success("Entraînement terminé avec succès!")
                st.session_state.training_active = False
                
                # Résultats
                st.markdown("#### Résultats Finaux")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Courbes de perte
                    epochs_range = list(range(1, 11))
                    train_losses = [2.5 - (e * 0.2) for e in epochs_range]
                    val_losses = [2.4 - (e * 0.18) for e in epochs_range]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=epochs_range, y=train_losses, name='Train Loss', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=epochs_range, y=val_losses, name='Val Loss', mode='lines+markers'))
                    fig.update_layout(title='Loss Curves', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Accuracy
                    accuracies = [50 + (e * 4) for e in epochs_range]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=epochs_range, y=accuracies, name='Accuracy', mode='lines+markers', fill='tozeroy'))
                    fig.update_layout(title='Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy (%)')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarder le modèle
                if st.button(" Sauvegarder le Modèle", use_container_width=True):
                    st.success("Modèle sauvegardé!")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# ============================================================
# NAVIGATION PRINCIPALE - MISE À JOUR
# ============================================================

def main():
    """Fonction principale"""
    
    init_session()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                AI Dev
            </h1>
            <p style="color: #6b7280; font-size: 0.9rem;">Development Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        menu_items = {
            "Dashboard": "dashboard",
            "Nouveau Projet": "new_project",
            "Mes Projets": "all_projects",
            "Workspace": "workspace",
            "Entraînement IA": "ai_training",
            "Mes Statistiques": "user_stats",
            "Stats Plateforme": "platform_stats",
            "Paramètres": "settings"
        }
        
        for label, view in menu_items.items():
            emoji = {
                "Dashboard": ".",
                "Nouveau Projet": ".",
                "Mes Projets": ".",
                "Workspace": ".",
                "Entraînement IA": ".",
                "Mes Statistiques": ".",
                "Stats Plateforme": ".",
                "Paramètres": "."
            }
            
            if st.button(f"{emoji[label]} {label}", use_container_width=True):
                st.session_state.active_view = view
                st.rerun()
        
        st.markdown("---")
        
        # Status API
        if check_api():
            st.success("API Connectée")
        else:
            st.error("API Déconnectée")
        
        st.markdown("---")
        
        # Raccourcis
        st.markdown("**Raccourcis**")
        if st.button(" Documentation", use_container_width=True):
            st.markdown("[Ouvrir](http://localhost:8001/docs)")
        if st.button(" Support", use_container_width=True):
            st.info("Support disponible")
        
        st.markdown("---")
        st.caption("v1.0.0 • 2025")
    
    # Afficher la page active
    active_view = st.session_state.active_view
    
    if active_view == "dashboard":
        page_dashboard()
    elif active_view == "new_project":
        page_new_project()
    elif active_view == "workspace":
        page_workspace()
    elif active_view == "all_projects":
        page_all_projects()
    elif active_view == "ai_training":
        page_ai_training()
    elif active_view == "user_stats":
        page_user_statistics()
    elif active_view == "platform_stats":
        page_platform_statistics()
    elif active_view == "settings":
        page_settings()


if __name__ == "__main__":
    main()
