"""
ultimate_lifecycle_frontend.py - Interface Streamlit Ultimate avec Animations
Lancez avec: streamlit run ai_lifecycle_app.py

Installation:
pip install streamlit plotly pandas numpy requests streamlit-lottie streamlit-option-menu
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np
import time
from typing import Dict, Any, List

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Lifecycle Platform Ultimate",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8006"

# ============================================================
# CSS ANIMATIONS & STYLES
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite, float 3s ease-in-out infinite;
        padding: 30px 0;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
    }
    
    .age-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .progress-ring {
        position: relative;
        width: 150px;
        height: 150px;
        margin: 0 auto;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #f5576c;
        animation: slideInRight 0.5s ease;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    
    .timeline-node {
        position: relative;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #667eea;
        animation: fadeIn 0.5s ease;
    }
    
    .timeline-node::before {
        content: '';
        position: absolute;
        left: -10px;
        top: 20px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #667eea;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .success-animation {
        animation: pulse 0.5s ease;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .badge-diamond {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .badge-platinum {
        background: linear-gradient(135deg, #d3d3d3 0%, #f0f0f0 100%);
    }
    
    .badge-gold {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .celebration {
        animation: pulse 0.3s ease 3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def init_session():
    """Initialise la session"""
    if 'models' not in st.session_state:
        st.session_state.models = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'show_confetti' not in st.session_state:
        st.session_state.show_confetti = False

def check_api():
    """Vérifie la connexion API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def parse_params(param_str: str) -> int:
    """Parse les paramètres"""
    param_str = param_str.strip().upper()
    multipliers = {'T': 1e12, 'B': 1e9, 'M': 1e6, 'K': 1e3}
    for suffix, mult in multipliers.items():
        if suffix in param_str:
            return int(float(param_str.replace(suffix, '')) * mult)
    return int(param_str)

def format_number(num: int) -> str:
    """Formate un nombre"""
    if num >= 1e12:
        return f"{num/1e12:.1f}T"
    elif num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    return f"{num:,}"

def show_loading_animation():
    """Affiche une animation de chargement"""
    with st.spinner(' Analyse en cours...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        progress_bar.empty()

def show_success_animation():
    """Affiche une animation de succès"""
    st.balloons()
    st.success("✨ Analyse terminée avec succès!")

# ============================================================
# COMPOSANTS VISUELS
# ============================================================

def create_radar_chart(scores: Dict[str, float], title: str = "Profil de Compétences"):
    """Crée un graphique radar animé"""
    categories = [k.replace('_', ' ').title() for k in scores.keys()]
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores Actuels',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            bgcolor='rgba(255,255,255,0.1)'
        ),
        showlegend=True,
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_gauge_chart(value: float, title: str):
    """Crée une jauge animée"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 100, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#fff4cc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 16}
    )
    
    return fig

def create_timeline_chart(milestones: List[Dict]):
    """Crée une timeline interactive"""
    df = pd.DataFrame(milestones)
    
    fig = px.scatter(
        df,
        x='date',
        y='progress_percentage',
        size='progress_percentage',
        color='progress_percentage',
        hover_name='name',
        hover_data=['key_deliverables'],
        title="Timeline de Progression",
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='white')))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['progress_percentage'],
        mode='lines+markers',
        line=dict(color='#667eea', width=3, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Progression (%)",
        hovermode='closest'
    )
    
    return fig

def create_comparison_chart(models: List[Dict]):
    """Crée un graphique de comparaison"""
    df = pd.DataFrame(models)
    
    # Vérification / création de la colonne 'maturity_score'
    if 'maturity_score' not in df.columns:
        if 'maturity' in df.columns:
            df['maturity_score'] = df['maturity']
        else:
            df['maturity_score'] = 0  # valeur par défaut si aucune info
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['maturity_score'],
        marker=dict(
            color=df['maturity_score'],
            colorscale='Viridis',
            showscale=True
        ),
        text=df['maturity_score'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Comparaison des Modèles",
        xaxis_title="Modèle",
        yaxis_title="Score de Maturité",
        height=500,
        showlegend=False
    )
    
    return fig


# ============================================================
# PAGES
# ============================================================

def page_home():
    """Page d'accueil avec animations"""
    st.markdown('<h1 class="main-header"> AI Lifecycle & Evolution Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; animation: fadeIn 1s ease;'>
        <p style='font-size: 1.3rem; color: #555;'>
             <strong>Système Révolutionnaire</strong> de Classification et d'Évolution des Modèles IA
        </p>
        <p style='font-size: 1.1rem; color: #777;'>
             Analyse •  Classification •  Prédiction •  Roadmap •  Certification
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques animées
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("8", "Ages vers l'AGI", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"),
        ("50+", "Critères", "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"),
        ("∞", "Possibilités", "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"),
        ("&", "Vers l'AGI", "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)")
    ]
    
    for col, (value, label, gradient) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background: {gradient};">
                <h1 style="margin: 0; font-size: 3rem;">{value}</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Les 8 Ages avec badges animés
    st.markdown("### Les 8 Ages de l'IA vers l'AGI")
    
    ages_info = [
        (".", "Age 1", "Narrow AI", "#FF6B6B", "Tâche unique spécialisée"),
        (".", "Age 2", "Specialized AI", "#4ECDC4", "Expertise domaine"),
        (".", "Age 3", "Multi-Task AI", "#45B7D1", "Tâches multiples"),
        (".", "Age 4", "Adaptive AI", "#96CEB4", "Apprentissage adaptatif"),
        (".", "Age 5", "General AI", "#FFEAA7", "Intelligence générale"),
        (".", "Age 6", "Advanced General AI", "#DFE6E9", "IG avancée"),
        (".", "Age 7", "Super AI", "#A29BFE", "Superintelligence"),
        (".", "AGI", "Artificial General Intelligence", "#FD79A8", "AGI complète")
    ]
    
    cols = st.columns(4)
    for i, (emoji, age, name, color, desc) in enumerate(ages_info):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="stat-box" style="border-left: 5px solid {color};">
                <h2 style="margin: 0;">{emoji} {age}</h2>
                <h4 style="color: {color}; margin: 5px 0;">{name}</h4>
                <p style="font-size: 0.9rem; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fonctionnalités
    st.markdown("### Fonctionnalités de la Plateforme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        features_left = [
            (".", "Analyse Multi-Critères", "50+ critères d'évaluation détaillés"),
            (".", "Prédictions Avancées", "Timeline et probabilités de succès"),
            (".", "Roadmap Personnalisée", "Feuille de route sur-mesure"),
            (".", "Visualisations", "Graphiques interactifs animés"),
            (".", "Certification", "Bronze, Silver, Gold, Platinum, Diamond")
        ]
        
        for emoji, title, desc in features_left:
            st.markdown(f"""
            <div class="recommendation-card" style="background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);">
                <h4>{emoji} {title}</h4>
                <p style="font-size: 0.9rem; margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        features_right = [
            (".", "Mentorat IA-to-IA", "Programme de mentorat de 6 mois"),
            (".", "Marketplace", "Datasets d'amélioration premium"),
            (".", "Analytics", "Statistiques et tendances avancées"),
            (".", "Comparaison", "Comparez jusqu'à 10 modèles"),
            (".", "Export PDF", "Rapports professionnels détaillés")
        ]
        
        for emoji, title, desc in features_right:
            st.markdown(f"""
            <div class="recommendation-card" style="background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);">
                <h4>{emoji} {title}</h4>
                <p style="font-size: 0.9rem; margin: 5px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to action animé
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("👈 **Commencez maintenant !** Ajoutez votre modèle dans le menu de gauche")

def page_add_model():
    """Page d'ajout de modèle avec animations"""
    st.markdown("##  Analyser votre Modèle IA")
    
    st.info(" Remplissez les caractéristiques de votre modèle pour obtenir une analyse complète de son cycle de vie")
    
    with st.form("model_form", clear_on_submit=False):
        # Section 1: Informations générales
        st.markdown("### Informations Générales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom du Modèle *", placeholder="Mon-Super-Modèle-v1.0")
            version = st.text_input("Version", "1.0")
            architecture = st.selectbox(
                "Architecture *",
                ["Transformer", "CNN", "RNN", "LSTM", "GAN", "Diffusion", "Hybrid", "Custom", "Autre"]
            )
        
        with col2:
            framework = st.selectbox(
                "Framework *",
                ["PyTorch", "TensorFlow", "JAX", "Keras", "Hugging Face", "Custom", "Autre"]
            )
            hardware = st.text_input("Hardware d'entraînement", placeholder="A100, TPU v4, H100...")
            description = st.text_area("Description", placeholder="Décrivez votre modèle...")
        
        st.markdown("---")
        
        # Section 2: Paramètres techniques
        st.markdown("### Paramètres Techniques")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            parameters_str = st.text_input(
                "Nombre de Paramètres *",
                placeholder="7B, 13B, 175B, 1T",
                help="Utilisez B pour milliards, M pour millions, T pour trillions"
            )
            layers = st.number_input("Nombre de Couches *", min_value=1, value=12)
        
        with col2:
            hidden_size = st.number_input("Taille Cachée *", min_value=1, value=768)
            attention_heads = st.number_input("Têtes d'Attention", min_value=0, value=12)
        
        with col3:
            max_tokens = st.number_input("Capacité Max de Tokens *", min_value=1, value=2048)
            context_window = st.number_input("Fenêtre de Contexte *", min_value=1, value=2048)
        
        st.markdown("---")
        
        # Section 3: Modalités et Tâches
        st.markdown("### Modalités et Tâches")
        
        col1, col2 = st.columns(2)
        
        with col1:
            modalities = st.multiselect(
                "Modalités Supportées *",
                ["text", "image", "audio", "video", "code", "3D", "multimodal", "speech"],
                default=["text"],
                help="Sélectionnez toutes les modalités que votre modèle peut traiter"
            )
        
        with col2:
            supported_tasks = st.multiselect(
                "Tâches Supportées *",
                [
                    "text_generation", "translation", "summarization", "question_answering",
                    "classification", "sentiment_analysis", "image_generation", "image_classification",
                    "object_detection", "speech_recognition", "speech_synthesis", "code_generation",
                    "code_completion", "reasoning", "planning", "conversation", "creative_writing",
                    "data_analysis", "mathematical_reasoning"
                ],
                default=["text_generation"],
                help="Sélectionnez toutes les tâches que votre modèle peut effectuer"
            )
        
        st.markdown("---")
        
        # Section 4: Performances
        st.markdown("### Performances")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            training_data_size = st.number_input("Taille Données d'Entraînement (TB)", min_value=0.0, value=1.0, step=0.1)
            training_duration = st.number_input("Durée d'Entraînement (heures)", min_value=0.0, value=100.0, step=10.0)
        
        with col2:
            inference_speed = st.number_input("Vitesse d'Inférence (tokens/sec)", min_value=0.0, value=50.0, step=5.0)
            memory_usage = st.number_input("Utilisation Mémoire (GB)", min_value=0.0, value=16.0, step=1.0)
        
        with col3:
            energy_efficiency = st.number_input("Efficacité Énergétique (FLOPS/W)", min_value=0.0, value=100.0, step=10.0)
        
        st.markdown("---")
        
        # Section 5: Capacités avancées
        st.markdown("### Capacités Avancées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            capabilities = st.multiselect(
                "Capacités Disponibles",
                [
                    "basic_inference", "domain_expertise", "fine_tuning", "multi_task",
                    "context_understanding", "basic_reasoning", "adaptive_learning",
                    "few_shot_learning", "meta_learning", "general_reasoning", "cross_domain",
                    "zero_shot_learning", "planning", "advanced_reasoning", "abstract_thinking",
                    "causal_inference", "self_improvement"
                ],
                help="Sélectionnez toutes les capacités avancées de votre modèle"
            )
            
            st.markdown("#### Apprentissage")
            fine_tuning = st.checkbox("Fine-tuning Capable")
            few_shot = st.checkbox("Few-Shot Learning")
            zero_shot = st.checkbox("Zero-Shot Learning")
        
        with col2:
            safety_measures = st.multiselect(
                "Mesures de Sécurité",
                ["RLHF", "Constitutional AI", "Red Teaming", "Content Filtering", 
                 "Adversarial Training", "Monitoring"],
                help="Mesures de sécurité implémentées"
            )
            
            st.markdown("#### Éthique & Sécurité")
            transfer_learning_cap = st.checkbox("Transfer Learning")
            meta_learning_cap = st.checkbox("Meta-Learning")
            bias_mitigation = st.checkbox("Mitigation des Biais")
            explainability = st.checkbox("Explicabilité")
        
        st.markdown("---")
        
        # Section 6: Scores d'évaluation
        st.markdown("### Scores d'Évaluation")
        st.info(" Évaluez votre modèle sur une échelle de 0 à 100 pour chaque métrique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Capacités Cognitives")
            reasoning = st.slider(" Raisonnement", 0, 100, 70, help="Capacité de raisonnement logique")
            adaptability = st.slider(" Adaptabilité", 0, 100, 60, help="Capacité d'adaptation à de nouvelles tâches")
            generalization = st.slider(" Généralisation", 0, 100, 65, help="Performance sur données non vues")
            creativity = st.slider(" Créativité", 0, 100, 55, help="Capacité de génération créative")
            learning_efficiency = st.slider(" Efficacité d'Apprentissage", 0, 100, 60, help="Rapidité d'apprentissage")
        
        with col2:
            st.markdown("#### Qualité & Sécurité")
            transfer_learning_score = st.slider(" Transfer Learning", 0, 100, 50, help="Capacité de réutiliser les connaissances")
            robustness = st.slider(" Robustesse", 0, 100, 65, help="Résistance aux perturbations")
            interpretability = st.slider(" Interprétabilité", 0, 100, 45, help="Capacité d'expliquer les décisions")
            safety = st.slider(" Sécurité", 0, 100, 70, help="Niveau de sécurité et d'alignement")
            ethical_alignment = st.slider(" Alignement Éthique", 0, 100, 60, help="Respect des principes éthiques")
        
        st.markdown("---")
        
        # Options avancées
        col1, col2 = st.columns(2)
        
        with col1:
            request_certification = st.checkbox(" Demander une Certification", help="Évaluation pour obtenir une certification officielle")
        
        with col2:
            enable_mentoring = st.checkbox(" Activer le Mentorat", help="Trouver des modèles mentors pour améliorer le vôtre")
        
        # Bouton de soumission
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(" Analyser le Modèle", use_container_width=True)
            
    if submitted:
            if not all([name, parameters_str, architecture, framework, modalities, supported_tasks]):
                st.error(" Veuillez remplir tous les champs obligatoires (*)")
                return
            
            try:
                show_loading_animation()
                
                # Parser les paramètres
                parameters = parse_params(parameters_str)
                
                # Construire la requête
                request_data = {
                    "characteristics": {
                        "name": name,
                        "version": version,
                        "description": description,
                        "architecture": architecture,
                        "parameters": parameters,
                        "layers": layers,
                        "hidden_size": hidden_size,
                        "attention_heads": attention_heads if attention_heads > 0 else None,
                        "max_tokens": max_tokens,
                        "context_window": context_window,
                        "modalities": modalities,
                        "supported_tasks": supported_tasks,
                        "framework": framework,
                        "training_data_size": training_data_size,
                        "training_duration": training_duration,
                        "hardware": hardware if hardware else None,
                        "inference_speed": inference_speed,
                        "memory_usage": memory_usage,
                        "energy_efficiency": energy_efficiency,
                        "capabilities": capabilities,
                        "fine_tuning_capable": fine_tuning,
                        "few_shot_learning": few_shot,
                        "zero_shot_learning": zero_shot,
                        "transfer_learning": transfer_learning_cap,
                        "meta_learning": meta_learning_cap,
                        "safety_measures": safety_measures,
                        "bias_mitigation": bias_mitigation,
                        "explainability_features": explainability,
                        "benchmark_scores": {}
                    },
                    "scores": {
                        "reasoning": reasoning,
                        "adaptability": adaptability,
                        "generalization": generalization,
                        "creativity": creativity,
                        "learning_efficiency": learning_efficiency,
                        "transfer_learning": transfer_learning_score,
                        "robustness": robustness,
                        "interpretability": interpretability,
                        "safety": safety,
                        "ethical_alignment": ethical_alignment
                    },
                    "request_certification": request_certification,
                    "enable_mentoring": enable_mentoring
                }
                
                # Appel API
                response = requests.post(f"{API_URL}/api/v1/analyze", json=request_data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_analysis = result["analysis"]
                    st.session_state.models.append(result["model_id"])
                    
                    show_success_animation()
                    
                    # Afficher les résultats
                    show_analysis_results(result["analysis"])
                else:
                    st.error(f" Erreur: {response.text}")
                    
            except Exception as e:
                st.error(f" Erreur lors de l'analyse: {str(e)}")   
        

def show_analysis_results(analysis: Dict):
    """Affiche les résultats avec animations"""
    st.markdown("---")
    st.markdown("## Résultats de l'Analyse")
    
    maturity = analysis["maturity_analysis"]
    
    # Métriques principales avec animation
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        (maturity["overall_score"], "Score de Maturité", ".", "#667eea"),
        (maturity["confidence"], "Confiance", ".", "#f5576c"),
        (maturity.get("progress_to_next_age", 0), "Progression", ".", "#43e97b"),
        (analysis["evolution_prediction"]["success_probability"], "Probabilité", ".", "#ffa502")
    ]
    
    for col, (value, label, emoji, color) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {color}22 0%, {color}66 100%); border-left: 5px solid {color};">
                <h1 style="color: {color}; margin: 0;">{value:.1f}</h1>
                <p style="color: #555; margin: 5px 0 0 0;">{emoji} {label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Age actuel avec badge animé
    current_age = maturity["current_age"]
    next_age = maturity["next_age"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        age_colors = {
            "Age 1": "#FF6B6B", "Age 2": "#4ECDC4", "Age 3": "#45B7D1", "Age 4": "#96CEB4",
            "Age 5": "#FFEAA7", "Age 6": "#DFE6E9", "Age 7": "#A29BFE", "AGI": "#FD79A8"
        }
        
        age_key = current_age.split(":")[0] if ":" in current_age else current_age
        color = age_colors.get(age_key, "#667eea")
        
        st.markdown(f"""
        <div class="stat-box" style="text-align: center; border: 3px solid {color};">
            <h2 style="color: {color};">🌟 Age Actuel</h2>
            <h1 class="age-badge" style="background: {color}; color: white; padding: 15px 30px; border-radius: 30px;">
                {current_age}
            </h1>
            <p style="margin-top: 15px; color: #666;">Confiance: {maturity['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if next_age != "AGI Atteint":
            next_key = next_age.split(":")[0] if ":" in next_age else next_age
            next_color = age_colors.get(next_key, "#764ba2")
            
            st.markdown(f"""
            <div class="stat-box" style="text-align: center; border: 3px dashed {next_color};">
                <h2 style="color: {next_color};"> Objectif Suivant</h2>
                <h1 style="color: {next_color}; padding: 15px 30px;">
                    {next_age}
                </h1>
                <p style="margin-top: 15px; color: #666;">Temps estimé: {analysis['evolution_prediction']['timeline']['total_months']} mois</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(" **Félicitations !** Vous avez atteint l'AGI !")
    
    st.markdown("---")
    
    # Graphique radar des compétences
    st.markdown("### Profil de Compétences")
    
    scores = analysis["model_info"]["scores"]
    fig_radar = create_radar_chart(scores)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Gauge de maturité
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gauge = create_gauge_chart(maturity["overall_score"], "Score de Maturité Global")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown("#### Répartition par Catégorie")
        gap_summary = analysis["gap_analysis"]["summary"]
        
        fig_pie = px.pie(
            values=list(gap_summary.values()),
            names=list(gap_summary.keys()),
            title="Gaps par Catégorie",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Analyse des gaps
    st.markdown("### Analyse des Écarts")
    
    gaps = analysis["gap_analysis"]["gaps"]
    
    if gaps:
        # Graphique des gaps
        df_gaps = pd.DataFrame(gaps)
        
        fig_gaps = px.bar(
            df_gaps,
            x='metric',
            y='percentage',
            color='priority',
            title='Progression par Métrique',
            labels={'metric': 'Métrique', 'percentage': 'Progression (%)'},
            color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#f39c12', 'MEDIUM': '#3498db', 'LOW': '#2ecc71'}
        )
        
        fig_gaps.update_layout(height=400)
        st.plotly_chart(fig_gaps, use_container_width=True)
        
        # Détails des gaps
        st.markdown("#### Détails des Écarts")
        
        for gap in gaps[:10]:  # Top 10
            priority_colors = {
                'CRITICAL': '#e74c3c',
                'HIGH': '#f39c12',
                'MEDIUM': '#3498db',
                'LOW': '#2ecc71'
            }
            
            color = priority_colors.get(gap['priority'], '#95a5a6')
            
            with st.expander(f"{gap['metric']} - Priorité: {gap['priority']} ({gap['percentage']:.1f}%)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Actuel", f"{gap['current']}")
                with col2:
                    st.metric("Requis", f"{gap['required']}")
                with col3:
                    st.metric("Effort Estimé", f"{gap.get('estimated_effort_months', 'N/A')} mois")
                
                progress = gap['percentage']
                st.progress(progress / 100)
    else:
        st.success("✅ Aucun gap identifié ! Votre modèle est prêt pour l'Age suivant.")
    
    st.markdown("---")
    
    # Recommandations
    st.markdown("### Recommandations Personnalisées")
    
    recommendations = analysis["recommendations"]["items"]
    
    for i, rec in enumerate(recommendations[:8], 1):
        priority_emoji = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        emoji = priority_emoji.get(rec['priority'], '⚪')
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>{emoji} {i}. {rec['title']}</h4>
            <p><strong>Catégorie:</strong> {rec['category']} | <strong>Priorité:</strong> {rec['priority']}</p>
            <p>{rec['description']}</p>
            <details>
                <summary><strong>🎯 Étapes d'Action</strong></summary>
                <ul>
                    {''.join(f'<li>{step}</li>' for step in rec['action_steps'])}
                </ul>
            </details>
            <p><strong>⏱️ Temps estimé:</strong> {rec['estimated_time']} | <strong>💰 Coût:</strong> {rec['estimated_cost']} | <strong>📈 Impact:</strong> {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Timeline d'évolution
    st.markdown("### Timeline de Progression")
    
    timeline = analysis["evolution_prediction"]["timeline"]
    
    if "milestones" in timeline:
        fig_timeline = create_timeline_chart(timeline["milestones"])
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Phases détaillées
        if "phases" in timeline:
            st.markdown("#### Phases du Projet")
            
            for phase in timeline["phases"]:
                with st.expander(f"{phase['phase']} ({phase['duration_months']} mois)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Période:** Mois {phase['start_month']} - {phase['end_month']}")
                        st.write(f"**Progrès attendu:** {phase['expected_progress']}")
                        st.write(f"**Difficulté:** {phase['difficulty']}")
                    
                    with col2:
                        st.write("**Améliorations:**")
                        for improvement in phase['improvements']:
                            st.write(f"- {improvement}")
    
    # Certification
    if analysis.get("certification"):
        st.markdown("---")
        st.markdown("### Éligibilité à la Certification")
        
        cert = analysis["certification"]
        
        if cert["eligible"]:
            level = cert["certification_level"]
            
            level_colors = {
                "Diamond": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
                "Platinum": "linear-gradient(135deg, #d3d3d3 0%, #f0f0f0 100%)",
                "Gold": "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)",
                "Silver": "linear-gradient(135deg, #c0c0c0 0%, #e0e0e0 100%)",
                "Bronze": "linear-gradient(135deg, #cd7f32 0%, #e6a857 100%)"
            }
            
            st.markdown(f"""
            <div class="celebration" style="text-align: center; padding: 30px; background: {level_colors.get(level, '#667eea')}; border-radius: 20px;">
                <h1> Éligible à la Certification {level} ! </h1>
                <p style="font-size: 1.2rem;">Score: {cert['percentage']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(" **Exigences Remplies:**")
                for req in cert["requirements_met"]:
                    st.write(f"✓ {req}")
            
            with col2:
                if cert["requirements_missing"]:
                    st.warning(" **Exigences Manquantes:**")
                    for req in cert["requirements_missing"]:
                        st.write(f"✗ {req}")
        else:
            st.warning(" Non éligible à la certification pour le moment")
            st.write("**Exigences manquantes:**")
            for req in cert["requirements_missing"]:
                st.write(f"- {req}")
    
    # Insights IA
    st.markdown("---")
    st.markdown("### Insights IA")
    
    insights = analysis["ai_insights"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("** Points Forts**")
        for strength in insights["strengths"]:
            st.write(f"✓ {strength}")
    
    with col2:
        st.error("** Points Faibles**")
        for weakness in insights["weaknesses"]:
            st.write(f"✗ {weakness}")
    
    st.info(f"** Domaines clés à améliorer:** {', '.join(insights['key_focus_areas'])}")
    st.success(f"** Quick Wins:** {', '.join(insights['quick_wins']) if insights['quick_wins'] else 'Aucun'}")
    
    # Boutons d'action
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Exporter PDF", use_container_width=True):
            with st.spinner("Génération du PDF..."):
                time.sleep(1)
                st.success(" PDF généré !")
    
    with col2:
        if st.button(" Demander Certification", use_container_width=True):
            st.info(" Demande en cours...")
    
    with col3:
        if st.button(" Trouver un Mentor", use_container_width=True):
            st.info(" Recherche de mentors...")
    
    with col4:
        if st.button(" Voir Marketplace", use_container_width=True):
            st.info(" Redirection vers marketplace...")

def page_leaderboard():
    """Page du leaderboard avec animations"""
    st.markdown("##  Leaderboard Global")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/leaderboard?limit=50")
        
        if response.status_code == 200:
            data = response.json()
            leaderboard = data["leaderboard"]
            
            if not leaderboard:
                st.info(" Aucun modèle dans le leaderboard pour le moment")
                return
            
            # Top 3 avec podium animé
            st.markdown("### 🥇 Top 3")
            
            top3 = leaderboard[:3]
            cols = st.columns(3)
            
            medals = ["🥇", "🥈", "🥉"]
            gradients = [
                "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)",
                "linear-gradient(135deg, #c0c0c0 0%, #e0e0e0 100%)",
                "linear-gradient(135deg, #cd7f32 0%, #e6a857 100%)"
            ]
            
            for i, (col, model, medal, gradient) in enumerate(zip(cols, top3, medals, gradients)):
                with col:
                    st.markdown(f"""
                    <div class="metric-card celebration" style="background: {gradient}; color: #2c3e50;">
                        <h1 style="font-size: 4rem; margin: 0;">{medal}</h1>
                        <h3>{model['name']}</h3>
                        <h2>{model['maturity']:.1f}</h2>
                        <p>{model['age']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Graphique de distribution
            st.markdown("### Distribution par Age")
            
            age_counts = pd.DataFrame(leaderboard)['age'].value_counts()
            
            fig_dist = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title="Nombre de Modèles par Age",
                labels={'x': 'Age', 'y': 'Nombre'},
                color=age_counts.values,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Tableau complet
            st.markdown("### Classement Complet")
            
            df = pd.DataFrame(leaderboard)
            
            # Styling
            st.dataframe(
                df[['rank', 'name', 'age', 'maturity', 'version']],
                use_container_width=True,
                height=600
            )
            
        else:
            st.error(" Erreur lors de la récupération du leaderboard")
            
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

def page_comparison():
    """Page de comparaison"""
    st.markdown("## Comparaison de Modèles")
    
    try:
        # Récupérer la liste des modèles
        response = requests.get(f"{API_URL}/api/v1/models")
        
        if response.status_code == 200:
            data = response.json()
            models = data["models"]
            
            if len(models) < 2:
                st.warning(" Vous devez avoir au moins 2 modèles pour effectuer une comparaison")
                return
            
            # Sélection des modèles
            model_names = [f"{m['name']} (Age: {m['age'].split(':')[0]})" for m in models]
            model_ids = [m['name'].replace(' ', '_') + '_' + m['timestamp'].split('T')[0].replace('-', '') for m in models]
            
            selected = st.multiselect(
                "Sélectionnez 2 à 10 modèles à comparer",
                options=range(len(models)),
                format_func=lambda i: model_names[i],
                max_selections=10
            )
            
            if len(selected) >= 2:
                if st.button(" Comparer", type="primary"):
                    with st.spinner("Comparaison en cours..."):
                        # Pour la démo, simulation
                        comparison_models = [models[i] for i in selected]
                        
                        # Graphique de comparaison
                        fig_comp = create_comparison_chart(comparison_models)
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Tableau comparatif
                        st.markdown("### Tableau Comparatif")
                        
                        df_comp = pd.DataFrame(comparison_models)
                        st.dataframe(df_comp, use_container_width=True)
            else:
                st.info(" Sélectionnez au moins 2 modèles")
                
        else:
            st.error(" Erreur lors de la récupération des modèles")
            
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

def page_marketplace():
    """Page marketplace"""
    st.markdown("## Marketplace de Datasets")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/marketplace/datasets")
        
        if response.status_code == 200:
            data = response.json()
            datasets = data["datasets"]
            
            # Filtres
            col1, col2 = st.columns(2)
            
            with col1:
                category_filter = st.selectbox("Catégorie", ["Tous"] + data.get("categories", []))
            
            with col2:
                sort_by = st.selectbox("Trier par", ["Prix", "Rating", "Téléchargements"])
            
            # Affichage des datasets
            for dataset in datasets:
                if category_filter != "Tous" and dataset.get("category") != category_filter:
                    continue
                
                with st.expander(f"{dataset.get('name', 'Dataset')} - {dataset.get('price', 'N/A')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {dataset.get('description', 'N/A')}")
                        st.write(f"**Taille:** {dataset.get('size', 'N/A')}")
                        st.write(f"**Rating:** {'⭐' * int(dataset.get('rating', 0))}")
                        
                        st.write("**Amélioration Potentielle:**")
                        for metric, improvement in dataset.get('improvement_potential', {}).items():
                            st.write(f"- {metric}: {improvement}")
                    
                    with col2:
                        st.metric("Téléchargements", dataset.get('downloads', 0))
                        st.write(f"**Ages compatibles:**")
                        for age in dataset.get('compatible_ages', []):
                            st.write(f"- {age}")
                        
                        if st.button(f" Acheter", key=dataset.get('id', 'ds_unknown')):
                            st.success(" Dataset acheté !")
                            
        else:
            st.error(" Erreur lors de la récupération du marketplace")
            
    except Exception as e:
        st.error(f" Erreur: {str(e)}")

def page_statistics():
    """Page des statistiques avancées"""
    st.markdown("## Statistiques Avancées")
    
    try:
        # Récupérer les stats
        response = requests.get(f"{API_URL}/api/v1/stats")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Métriques principales
            st.markdown("### Métriques Globales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h1 style="margin: 0;">{stats['total_models']}</h1>
                    <p style="margin: 5px 0 0 0;">Modèles Totaux</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h1 style="margin: 0;">{stats['average_maturity']:.1f}</h1>
                    <p style="margin: 5px 0 0 0;">Maturité Moyenne</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h1 style="margin: 0;">{stats.get('median_maturity', 0):.1f}</h1>
                    <p style="margin: 5px 0 0 0;">Maturité Médiane</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h1 style="margin: 0;">{stats.get('max_maturity', 0):.1f}</h1>
                    <p style="margin: 5px 0 0 0;">Score Max</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Distribution par Age
            st.markdown("### Distribution par Age")
            
            age_dist = stats.get('age_distribution', {})
            
            if age_dist:
                # Graphique en camembert
                fig_pie = px.pie(
                    values=list(age_dist.values()),
                    names=list(age_dist.keys()),
                    title="Répartition des Modèles par Age",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Modèles: %{value}<br>Pourcentage: %{percent}<extra></extra>'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Graphique en barres
                fig_bar = px.bar(
                    x=list(age_dist.keys()),
                    y=list(age_dist.values()),
                    title="Nombre de Modèles par Age",
                    labels={'x': 'Age', 'y': 'Nombre de Modèles'},
                    color=list(age_dist.values()),
                    color_continuous_scale='Viridis'
                )
                
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Tendances temporelles (si historique disponible)
            st.markdown("### Évolution Temporelle")
            
            # Récupérer tous les modèles pour analyser les tendances
            models_response = requests.get(f"{API_URL}/api/v1/models")
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = models_data.get('models', [])
                
                if models:
                    # Créer un DataFrame
                    df_models = pd.DataFrame(models)
                    df_models['timestamp'] = pd.to_datetime(df_models['timestamp'])
                    df_models = df_models.sort_values('timestamp')
                    
                    # Graphique de tendance de maturité
                    fig_trend = px.line(
                        df_models,
                        x='timestamp',
                        y='maturity',
                        title='Évolution de la Maturité des Modèles',
                        labels={'timestamp': 'Date', 'maturity': 'Score de Maturité'},
                        markers=True
                    )
                    
                    fig_trend.update_traces(
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Analyse par Age
                    st.markdown("### Analyse Détaillée par Age")
                    
                    for age in age_dist.keys():
                        age_models = [m for m in models if m['age'] == age]
                        
                        if age_models:
                            with st.expander(f"{age} ({len(age_models)} modèles)"):
                                avg_maturity = sum(m['maturity'] for m in age_models) / len(age_models)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Nombre", len(age_models))
                                with col2:
                                    st.metric("Maturité Moy.", f"{avg_maturity:.1f}")
                                with col3:
                                    st.metric("% du Total", f"{(len(age_models) / len(models) * 100):.1f}%")
                                
                                # Liste des modèles
                                st.write("**Modèles:**")
                                for model in age_models[:5]:
                                    st.write(f"- {model['name']} (v{model['version']}) - Score: {model['maturity']:.1f}")
                                
                                if len(age_models) > 5:
                                    st.write(f"... et {len(age_models) - 5} autres")
            
            st.markdown("---")
            
            # Statistiques supplémentaires
            st.markdown("### Statistiques Complémentaires")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="stat-box">
                    <h4> Objectifs</h4>
                    <p><strong>Score Min:</strong> {:.1f}</p>
                    <p><strong>Score Max:</strong> {:.1f}</p>
                    <p><strong>Écart:</strong> {:.1f} points</p>
                </div>
                """.format(
                    stats.get('min_maturity', 0),
                    stats.get('max_maturity', 0),
                    stats.get('max_maturity', 0) - stats.get('min_maturity', 0)
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-box">
                    <h4> Progression</h4>
                    <p><strong>Moyenne:</strong> {:.1f}</p>
                    <p><strong>Médiane:</strong> {:.1f}</p>
                    <p><strong>Potentiel:</strong> {:.1f} points</p>
                </div>
                """.format(
                    stats.get('average_maturity', 0),
                    stats.get('median_maturity', 0),
                    100 - stats.get('average_maturity', 0)
                ), unsafe_allow_html=True)
            
        else:
            st.error(" Erreur lors de la récupération des statistiques")
            
    except Exception as e:
        st.error(f" Erreur: {str(e)}")
        st.info(" Assurez-vous que l'API est démarrée")


def page_settings():
    """Page des paramètres"""
    st.markdown("## Paramètres & Configuration")
    
    st.info(" Personnalisez votre expérience sur la plateforme")
    
    # Section 1: Paramètres de l'API
    st.markdown("### Configuration API")
    
    with st.expander("Configuration de l'API", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("URL de l'API", value=API_URL)
            timeout = st.number_input("Timeout (secondes)", min_value=1, max_value=60, value=5)
        
        with col2:
            max_retries = st.number_input("Tentatives max", min_value=1, max_value=10, value=3)
            
            if st.button(" Tester la Connexion"):
                with st.spinner("Test en cours..."):
                    try:
                        response = requests.get(f"{api_url}/health", timeout=timeout)
                        if response.status_code == 200:
                            st.success(" Connexion réussie!")
                            st.json(response.json())
                        else:
                            st.error(f" Erreur: {response.status_code}")
                    except Exception as e:
                        st.error(f" Erreur de connexion: {str(e)}")
    
    st.markdown("---")
    
    # Section 2: Préférences d'affichage
    st.markdown("### Préférences d'Affichage")
    
    with st.expander("Options d'Interface"):
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Thème", ["Clair", "Sombre", "Auto"])
            language = st.selectbox("Langue", ["Français", "English", "Español"])
            
        with col2:
            show_animations = st.checkbox("Afficher les animations", value=True)
            show_tooltips = st.checkbox("Afficher les infobulles", value=True)
        
        chart_type = st.selectbox(
            "Type de graphiques par défaut",
            ["Interactifs (Plotly)", "Statiques (Matplotlib)", "Auto"]
        )
        
        items_per_page = st.slider("Éléments par page", 10, 100, 50, 10)
    
    st.markdown("---")
    
    # Section 3: Notifications
    st.markdown("### Notifications")
    
    with st.expander("Préférences de Notifications"):
        st.checkbox("Recevoir des notifications par email", value=False)
        st.checkbox("Alertes de nouvelles recommandations", value=True)
        st.checkbox("Alertes d'évolution de maturité", value=True)
        st.checkbox("Notifications de certification", value=True)
        st.checkbox("Alertes marketplace", value=False)
        
        notification_frequency = st.selectbox(
            "Fréquence des notifications",
            ["Temps réel", "Quotidien", "Hebdomadaire", "Mensuel"]
        )
    
    st.markdown("---")
    
    # Section 4: Sécurité & Confidentialité
    st.markdown("### Sécurité & Confidentialité")
    
    with st.expander("Paramètres de Sécurité"):
        st.checkbox("Activer l'authentification à deux facteurs", value=False)
        st.checkbox("Chiffrer les données sensibles", value=True)
        st.checkbox("Masquer les informations confidentielles", value=False)
        
        data_retention = st.selectbox(
            "Durée de conservation des données",
            ["1 mois", "3 mois", "6 mois", "1 an", "Illimitée"]
        )
        
        st.write("**Gestion des données:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(" Exporter mes données"):
                st.info(" Export en cours de préparation...")
        
        with col2:
            if st.button(" Supprimer mes données", type="secondary"):
                st.warning(" Cette action est irréversible!")
    
    st.markdown("---")
    
    # Section 5: Intégrations
    st.markdown("### Intégrations")
    
    with st.expander("Services Externes"):
        st.markdown("**Connectez vos outils préférés:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("GitHub", value=False)
            st.checkbox("Hugging Face", value=False)
            st.checkbox("W&B", value=False)
        
        with col2:
            st.checkbox("MLflow", value=False)
            st.checkbox("TensorBoard", value=False)
            st.checkbox("Comet", value=False)
        
        with col3:
            st.checkbox("Slack", value=False)
            st.checkbox("Discord", value=False)
            st.checkbox("Teams", value=False)
    
    st.markdown("---")
    
    # Section 6: Export & Rapports
    st.markdown("### Export & Rapports")
    
    with st.expander("Options d'Export"):
        export_format = st.multiselect(
            "Formats d'export disponibles",
            ["PDF", "JSON", "CSV", "Excel", "Markdown"],
            default=["PDF", "JSON"]
        )
        
        include_graphs = st.checkbox("Inclure les graphiques", value=True)
        include_details = st.checkbox("Inclure les détails complets", value=True)
        
        report_template = st.selectbox(
            "Modèle de rapport",
            ["Standard", "Exécutif", "Technique", "Compact"]
        )
    
    st.markdown("---")
    
    # Section 7: Cache & Performance
    st.markdown("### Cache & Performance")
    
    with st.expander("Options de Performance"):
        enable_cache = st.checkbox("Activer le cache", value=True)
        cache_duration = st.slider("Durée du cache (minutes)", 1, 60, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(" Vider le cache"):
                st.session_state.clear()
                st.success(" Cache vidé!")
        
        with col2:
            if st.button(" Recharger l'application"):
                st.rerun()
    
    st.markdown("---")
    
    # Section 8: Informations système
    st.markdown("### Informations Système")
    
    with st.expander("Détails du Système"):
        st.write("**Version de la plateforme:** v4.0.0")
        st.write("**Framework:** Streamlit + FastAPI")
        st.write("**Python:** 3.8+")
        st.write("**Status API:** Connectée" if check_api() else "Déconnectée")
        
        st.markdown("**Bibliothèques:**")
        st.code("""
streamlit
plotly
pandas
numpy
requests
fastapi
uvicorn
        """)
    
    st.markdown("---")
    
    # Boutons d'action
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Sauvegarder les Paramètres", use_container_width=True):
            st.success(" Paramètres sauvegardés!")
    
    with col2:
        if st.button(" Réinitialiser", use_container_width=True):
            st.warning(" Paramètres réinitialisés aux valeurs par défaut")
    
    with col3:
        if st.button(" Annuler", use_container_width=True):
            st.info(" Modifications annulées")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p> <strong>Astuce:</strong> Modifiez les paramètres selon vos besoins pour une expérience optimale</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# NAVIGATION PRINCIPALE
# ============================================================

def main():
    """Fonction principale"""
    init_session()
    
    # Vérifier l'API
    api_status = check_api()
    
    # Sidebar avec animations
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; animation: fadeIn 1s ease;'>
            <h1 style='background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                 AI Lifecycle
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            [
                " Accueil",
                " Analyser un Modèle",
                " Leaderboard",
                " Comparaison",
                " Marketplace",
                " Statistiques",
                " Paramètres"
            ]
        )
        
        st.markdown("---")
        
        # Status API
        if api_status:
            st.success(" API: Connectée")
        else:
            st.error(" API: Déconnectée")
        
        st.markdown("---")
        
        # Stats rapides
        st.markdown("### Stats")
        try:
            stats_response = requests.get(f"{API_URL}/api/v1/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.metric("Modèles", stats.get("total_models", 0))
                st.metric("Maturité Moy.", f"{stats.get('average_maturity', 0):.1f}")
        except:
            pass
    
    # Afficher la page sélectionnée
    if page == " Accueil":
        page_home()
    elif page == " Analyser un Modèle":
        page_add_model()
    elif page == " Leaderboard":
        page_leaderboard()
    elif page == " Comparaison":
        page_comparison()
    elif page == " Marketplace":
        page_marketplace()
    elif page == " Statistiques":
        page_statistics()
    elif page == " Paramètres":
        page_settings()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; animation: fadeIn 1s ease;'>
        <p style='color: #666;'> <strong>AI Lifecycle & Evolution Platform v1.0</strong></p>
        <p style='color: #999;'>Système révolutionnaire de gestion du cycle de vie des modèles IA</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
        