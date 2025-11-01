# universal_tokenizer_dashboard.py - Interface Streamlit pour Tokenizer Universel

import asyncio
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import uuid
from typing import Dict, List, Any
import base64
import io

from tokenizer_ai_api import get_platform_statistics

# Configuration de la page
st.set_page_config(
    page_title="Universal Tokenizer Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration API streamlit run tokenizer_ai_app.py
TOKENIZER_API_URL = "http://localhost:8046"
AI_TRAINING_API_URL = "http://localhost:8006"

# Style CSS avancé
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #6366f1;
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
    font-weight: 700;
}

.tokenizer-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.language-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border-left: 6px solid #6366f1;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.language-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.metric-card {
    background: linear-gradient(45deg, #6366f1, #8b5cf6);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.token-display {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'Courier New', monospace;
}

.corpus-stats {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.comparison-table {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.training-progress {
    background: #fef3c7;
    border: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.analysis-section {
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# État de session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'tokenization_history' not in st.session_state:
    st.session_state.tokenization_history = []

# Fonctions utilitaires
def call_tokenizer_api(endpoint, method="GET", data=None, files=None):
    """Appel API Tokenizer"""
    url = f"{TOKENIZER_API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def get_available_tokenizers():
    """Récupère la liste des tokenizers disponibles"""
    try:
        result, error = call_tokenizer_api("/tokenizers/available")
        if result and isinstance(result, dict):
            return result.get("tokenizers", {})
        return {}
    except Exception as e:
        st.error(f"Erreur récupération tokenizers: {e}")
        return {}

def get_available_corpus():
    """Récupère la liste des corpus disponibles"""
    try:
        result, error = call_tokenizer_api("/corpus/available")
        if result and isinstance(result, dict):
            return result.get("corpus", {})
        return {}
    except Exception as e:
        st.error(f"Erreur récupération corpus: {e}")
        return {}

def validate_tokenization_result(result: Dict) -> bool:
    """Valide la cohérence des données de tokenisation"""
    try:
        if not result:
            return False
        
        tokens = result.get('tokens', [])
        input_ids = result.get('input_ids', [])
        
        # Vérifications de base
        if not tokens and not input_ids:
            return False
        
        # Si les deux existent, vérifier la cohérence
        if tokens and input_ids:
            return len(tokens) > 0 and len(input_ids) > 0
        
        return True
        
    except Exception:
        return False

def safe_create_dataframe(data_dict: Dict) -> pd.DataFrame:
    """Crée un DataFrame en s'assurant que toutes les colonnes ont la même longueur"""
    try:
        if not data_dict:
            return pd.DataFrame()
        
        # Trouver la longueur minimale
        lengths = [len(v) if isinstance(v, list) else 1 for v in data_dict.values()]
        min_length = min(lengths) if lengths else 0
        
        if min_length == 0:
            return pd.DataFrame()
        
        # Tronquer toutes les listes à la longueur minimale
        safe_data = {}
        for key, value in data_dict.items():
            if isinstance(value, list):
                safe_data[key] = value[:min_length]
            else:
                safe_data[key] = [value] * min_length
        
        return pd.DataFrame(safe_data)
        
    except Exception:
        return pd.DataFrame()
    """Récupère les statistiques de la plateforme"""
    result, error = call_tokenizer_api("/statistics/usage")
    if result:
        return result
    return {}

def main():
    st.markdown('<h1 class="main-header"> Universal Tokenizer Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Sections:",
            [
                "Tokenizer Playground", 
                "Analyse de Langue", 
                "Corpus Manager", 
                "Comparaison de Tokenizers",
                "Entraîner Tokenizer",
                "Analytics & Stats",
                "Intégration IA Training"
            ]
        )
        
        st.divider()
        
        # Statistiques en temps réel
        st.subheader("Statistiques Platform")
        # stats = get_platform_statistics()
        stats = asyncio.run(get_platform_statistics())
        if stats:
            platform_stats = stats.get("platform_stats", {})
            
            st.metric("Tokenizations", platform_stats.get("total_tokenizations", 0))
            st.metric("Tokenizers", platform_stats.get("available_tokenizers", 0))
            st.metric("Corpus", platform_stats.get("available_corpus", 0))
            st.metric("Custom Trained", platform_stats.get("custom_tokenizers_trained", 0))
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Session Info")
        st.write(f"User ID: `{st.session_state.user_id[:8]}...`")
        st.write(f"History: {len(st.session_state.tokenization_history)} tokenizations")
        
        # Statut API
        health, error = call_tokenizer_api("/health")
        if health:
            st.success("🟢 Tokenizer API en ligne")
            st.write(f"Version: {health.get('version', 'N/A')}")
        else:
            st.error("🔴 Tokenizer API hors ligne")
    
    # Pages principales
    if page == "Tokenizer Playground":
        show_tokenizer_playground()
    elif page == "Analyse de Langue":
        show_language_analysis()
    elif page == "Corpus Manager":
        show_corpus_manager()
    elif page == "Comparaison de Tokenizers":
        show_tokenizer_comparison()
    elif page == "Entraîner Tokenizer":
        show_tokenizer_training()
    elif page == "Analytics & Stats":
        show_analytics_dashboard()
    elif page == "Intégration IA Training":
        show_ai_integration()

def show_tokenizer_playground():
    """Interface principale de tokenisation"""
    st.title("🎮 Tokenizer Playground")
    
    st.markdown("""
    <div class="tokenizer-card">
        <h3>Testez la Puissance du Tokenizer Universel</h3>
        <p>Supportez toutes les langues et analysez en détail le processus de tokenisation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration de tokenisation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Texte à Tokeniser")
        
        # Zone de texte avec exemples prédéfinis
        sample_texts = {
            "Anglais": "The quick brown fox jumps over the lazy dog. How are you today?",
            "Français": "Le renard brun et rapide saute par-dessus le chien paresseux. Comment allez-vous aujourd'hui?",
            "Espagnol": "El zorro marrón rápido salta sobre el perro perezoso. ¿Cómo estás hoy?",
            "Chinois": "快速的棕色狐狸跳过懒惰的狗。你今天好吗?",
            "Arabe": "الثعلب البني السريع يقفز فوق الكلب الكسول. كيف حالك اليوم؟",
            "Japonais": "素早い茶色のキツネが怠惰な犬を飛び越えます。今日はいかがですか？",
            "Technique": "import tensorflow as tf\nmodel = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu')])",
            "Multilingue": "Hello world! Bonjour le monde! ¡Hola mundo! 你好世界! مرحبا بالعالم!"
        }
        
        selected_sample = st.selectbox("Exemples prédéfinis:", ["Texte personnalisé"] + list(sample_texts.keys()))
        
        if selected_sample != "Texte personnalisé":
            input_text = st.text_area("Texte:", value=sample_texts[selected_sample], height=150)
        else:
            input_text = st.text_area("Texte:", height=150, placeholder="Entrez votre texte dans n'importe quelle langue...")
    
    with col2:
        st.subheader(" Configuration")
        
        # Sélection du tokenizer
        tokenizers = get_available_tokenizers()
        tokenizer_names = list(tokenizers.keys()) if tokenizers else []
        
        selected_tokenizer = st.selectbox(
            "Tokenizer:",
            ["auto"] + tokenizer_names
        )
        
        # Options d'affichage
        st.write("**Options d'affichage:**")
        show_tokens = st.checkbox("Afficher tokens", value=True)
        show_ids = st.checkbox("Afficher IDs", value=True)
        show_attention = st.checkbox("Attention mask", value=False)
        show_analysis = st.checkbox("Analyse complète", value=True)
        
        # Bouton de tokenisation
        tokenize_button = st.button(" Tokeniser", type="primary", use_container_width=True)
    
    # Traitement de la tokenisation
    if tokenize_button and input_text.strip():
        with st.spinner("Tokenisation en cours..."):
            tokenize_request = {
                "text": input_text,
                "tokenizer_name": selected_tokenizer if selected_tokenizer != "auto" else None,
                "return_tokens": show_tokens,
                "return_ids": show_ids,
                "return_attention_mask": show_attention,
                "return_analysis": show_analysis
            }
            
            result, error = call_tokenizer_api("/tokenize", method="POST", data=tokenize_request)
            
            if result:
                display_tokenization_results(result, input_text)
                
                # Ajouter à l'historique
                st.session_state.tokenization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "text_sample": input_text[:50] + "..." if len(input_text) > 50 else input_text,
                    "tokenizer": result.get("tokenizer_used", "unknown"),
                    "token_count": result.get("token_count", 0),
                    "language": result.get("detected_language", "unknown")
                })
            else:
                st.error(f"Erreur de tokenisation: {error}")
    
    # Historique des tokenisations
    if st.session_state.tokenization_history:
        with st.expander(" Historique des Tokenisations", expanded=False):
            history_df = pd.DataFrame(st.session_state.tokenization_history)
            st.dataframe(history_df, use_container_width=True)

def display_tokenization_results(result: Dict, original_text: str):
    """Affiche les résultats de tokenisation de manière détaillée"""
    
    # Validation des données avant affichage
    if not validate_tokenization_result(result):
        st.error("Données de tokenisation invalides ou incohérentes")
        return
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result.get('token_count', 0)}</h3>
            <p>Tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result.get('detected_language', 'N/A')}</h3>
            <p>Langue détectée</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{result.get('tokenizer_used', 'N/A')}</h3>
            <p>Tokenizer utilisé</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        processing_time = result.get('processing_time_seconds', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{processing_time:.3f}s</h3>
            <p>Temps de traitement</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Résultats détaillés
    tabs = st.tabs(["Tokens", "IDs", "Analyse Linguistique", "Métriques"])
    
    with tabs[0]:  # Tokens
        if 'tokens' in result:
            st.subheader("Tokens générés")
            
            tokens = result['tokens']
            
            # Affichage coloré des tokens
            token_display = ""
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8', '#f7dc6f']
            
            for i, token in enumerate(tokens):
                color = colors[i % len(colors)]
                token_display += f'<span style="background-color: {color}; padding: 2px 6px; margin: 2px; border-radius: 4px; color: white; font-weight: bold;">{token}</span>'
            
            st.markdown(f'<div class="token-display">{token_display}</div>', unsafe_allow_html=True)
            
            # Statistiques des tokens
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Nombre total de tokens:** {len(tokens)}")
                st.write(f"**Tokens uniques:** {len(set(tokens))}")
                st.write(f"**Ratio de compression:** {len(tokens) / len(original_text):.3f}")
            
            with col2:
                if tokens:
                    avg_token_length = np.mean([len(token) for token in tokens])
                    st.write(f"**Longueur moyenne des tokens:** {avg_token_length:.2f}")
                    st.write(f"**Token le plus long:** {max(tokens, key=len) if tokens else 'N/A'}")
                    st.write(f"**Token le plus court:** {min(tokens, key=len) if tokens else 'N/A'}")
    
    with tabs[1]:  # IDs
        if 'input_ids' in result:
            st.subheader("IDs des tokens")
            
            input_ids = result['input_ids']
            
            # Affichage des IDs
            tokens = result.get('tokens', [])
            
            # S'assurer que toutes les listes ont la même longueur
            min_length = min(len(tokens), len(input_ids))
            
            ids_df = pd.DataFrame({
                'Token': tokens[:min_length],
                'ID': input_ids[:min_length],
                'Index': range(min_length)
            })
            
            st.dataframe(ids_df, use_container_width=True)
            
            # Graphique de distribution des IDs
            if len(input_ids) > 1:
                fig = px.histogram(
                    x=input_ids, 
                    title="Distribution des IDs de tokens",
                    labels={'x': 'Token ID', 'y': 'Fréquence'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Analyse Linguistique
        if 'analysis' in result:
            analysis = result['analysis']
            
            st.subheader("Analyse Linguistique Complète")
            
            # Statistiques de base
            if 'basic_stats' in analysis:
                basic_stats = analysis['basic_stats']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Statistiques de base:**")
                    for key, value in basic_stats.items():
                        if isinstance(value, float):
                            st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")
                
                with col2:
                    # Graphique des métriques
                    metrics = {k: v for k, v in basic_stats.items() if isinstance(v, (int, float))}
                    if metrics:
                        fig = px.bar(
                            x=list(metrics.keys()), 
                            y=list(metrics.values()),
                            title="Métriques de base"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Caractéristiques linguistiques
            if 'linguistic_features' in analysis:
                ling_features = analysis['linguistic_features']
                
                st.subheader("Caractéristiques Linguistiques")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Langue:** {ling_features.get('language', 'N/A')}")
                    st.write(f"**Type d'écriture:** {ling_features.get('script_type', 'N/A')}")
                    st.write(f"**Contient ponctuation:** {'Oui' if ling_features.get('has_punctuation') else 'Non'}")
                    st.write(f"**Contient chiffres:** {'Oui' if ling_features.get('has_numbers') else 'Non'}")
                
                with col2:
                    # Catégories Unicode
                    if 'unicode_categories' in ling_features:
                        unicode_cats = ling_features['unicode_categories']
                        fig = px.pie(
                            values=list(unicode_cats.values()),
                            names=list(unicode_cats.keys()),
                            title="Catégories de caractères Unicode"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des tokens
            if 'token_analysis' in analysis:
                token_analysis = analysis['token_analysis']
                
                st.subheader("Analyse des Tokens")
                
                # Types de tokens
                if 'token_types' in token_analysis:
                    token_types = token_analysis['token_types']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for token_type, count in token_types.items():
                            st.write(f"**{token_type.replace('_', ' ').title()}:** {count}")
                    
                    with col2:
                        fig = px.pie(
                            values=list(token_types.values()),
                            names=[name.replace('_', ' ').title() for name in token_types.keys()],
                            title="Répartition des types de tokens"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Tokens les plus fréquents
                if 'most_frequent_tokens' in token_analysis:
                    freq_tokens = token_analysis['most_frequent_tokens']
                    
                    st.write("**Tokens les plus fréquents:**")
                    freq_df = pd.DataFrame(freq_tokens, columns=['Token', 'Fréquence'])
                    st.bar_chart(freq_df.set_index('Token'))
    
    with tabs[3]:  # Métriques
        if 'analysis' in result and 'complexity_metrics' in result['analysis']:
            complexity = result['analysis']['complexity_metrics']
            
            st.subheader("Métriques de Complexité")
            
            # Métriques de lisibilité
            readability_metrics = {
                'flesch_reading_ease': 'Facilité de lecture Flesch',
                'flesch_kincaid_grade': 'Niveau Flesch-Kincaid', 
                'automated_readability_index': 'Index de lisibilité automatisé',
                'gunning_fog': 'Index Gunning Fog',
                'coleman_liau_index': 'Index Coleman-Liau'
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Métriques de lisibilité:**")
                for metric, display_name in readability_metrics.items():
                    if metric in complexity:
                        value = complexity[metric]
                        if isinstance(value, (int, float)):
                            st.write(f"- {display_name}: {value:.2f}")
            
            with col2:
                # Graphique radar des métriques
                available_metrics = {k: v for k, v in complexity.items() if isinstance(v, (int, float)) and k in readability_metrics}
                
                if available_metrics:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=list(available_metrics.values()),
                        theta=[readability_metrics[k] for k in available_metrics.keys()],
                        fill='toself',
                        name='Métriques de complexité'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, max(available_metrics.values()) * 1.1])
                        ),
                        showlegend=False,
                        title="Profil de complexité textuelle"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Métriques linguistiques additionnelles
            linguistic_metrics = {
                'lexical_diversity': 'Diversité lexicale',
                'avg_sentence_length': 'Longueur moyenne des phrases',
                'sentence_length_variance': 'Variance longueur phrases'
            }
            
            st.write("**Métriques linguistiques:**")
            for metric, display_name in linguistic_metrics.items():
                if metric in complexity:
                    value = complexity[metric]
                    if isinstance(value, (int, float)):
                        st.write(f"- {display_name}: {value:.3f}")

def show_language_analysis():
    """Page d'analyse de langue avancée"""
    st.title(" Analyse de Langue Avancée")
    
    st.markdown("""
    <div class="analysis-section">
        <h3>Détection et Analyse Linguistique Multilingue</h3>
        <p>Analysez les caractéristiques linguistiques de n'importe quel texte</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zone de texte pour analyse
    text_to_analyze = st.text_area(
        "Texte à analyser:",
        height=200,
        placeholder="Entrez un texte dans n'importe quelle langue pour une analyse linguistique complète..."
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        analyze_button = st.button(" Analyser Langue", type="primary", use_container_width=True)
    
    if analyze_button and text_to_analyze.strip():
        with st.spinner("Analyse linguistique en cours..."):
            
            # Appel API d'analyse de langue
            result, error = call_tokenizer_api(f"/analyze/language?text={text_to_analyze}")
            
            if result:
                st.success("Analyse terminée!")
                
                # Résultats principaux
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="language-card">
                        <h4>Langue Détectée</h4>
                        <h2>{result.get('detected_language', 'N/A').upper()}</h2>
                        <p>Confiance: {result.get('confidence', 0):.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="language-card">
                        <h4>Tokenizer Suggéré</h4>
                        <h3>{result.get('suggested_tokenizer', 'N/A')}</h3>
                        <p>Optimisé pour cette langue</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    available_tokenizers = result.get('available_tokenizers', [])
                    st.markdown(f"""
                    <div class="language-card">
                        <h4>Tokenizers Disponibles</h4>
                        <h2>{len(available_tokenizers)}</h2>
                        <p>Options au total</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Caractéristiques linguistiques détaillées
                if 'linguistic_features' in result:
                    features = result['linguistic_features']
                    
                    st.subheader("Caractéristiques Linguistiques Détaillées")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Propriétés générales:**")
                        st.write(f"- Type d'écriture: {features.get('script_type', 'N/A')}")
                        st.write(f"- Contient ponctuation: {'✅' if features.get('has_punctuation') else '❌'}")
                        st.write(f"- Contient chiffres: {'✅' if features.get('has_numbers') else '❌'}")
                        st.write(f"- Majuscules: {'✅' if features.get('has_uppercase') else '❌'}")
                        st.write(f"- Minuscules: {'✅' if features.get('has_lowercase') else '❌'}")
                    
                    with col2:
                        # Distribution des catégories Unicode
                        if 'unicode_categories' in features:
                            unicode_cats = features['unicode_categories']
                            
                            # Créer un graphique des catégories Unicode
                            fig = px.bar(
                                x=list(unicode_cats.keys()),
                                y=list(unicode_cats.values()),
                                title="Distribution des catégories Unicode",
                                labels={'x': 'Catégorie Unicode', 'y': 'Nombre de caractères'}
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Recommandations de tokenizers
                st.subheader("Recommandations")
                
                recommended = result.get('suggested_tokenizer', '')
                all_tokenizers = result.get('available_tokenizers', [])
                
                st.write(f"**Tokenizer recommandé:** `{recommended}`")
                
                if recommended in all_tokenizers:
                    other_options = [t for t in all_tokenizers if t != recommended][:3]
                    if other_options:
                        st.write("**Alternatives possibles:**")
                        for tokenizer in other_options:
                            st.write(f"- `{tokenizer}`")
                
            else:
                st.error(f"Erreur d'analyse: {error}")

def show_corpus_manager():
    """Gestionnaire de corpus"""
    st.title(" Corpus Manager")
    
    tab1, tab2 = st.tabs(["Corpus Disponibles", "Upload Corpus"])
    
    with tab1:
        st.subheader("Corpus Multilingues Disponibles")
        
        corpus_collection = get_available_corpus()
        
        if corpus_collection:
            for corpus_name, corpus_info in corpus_collection.items():
                with st.expander(f" {corpus_name}", expanded=False):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {corpus_info.get('description', 'N/A')}")
                        st.write(f"**Langues:** {', '.join(corpus_info.get('languages', []))}")
                        st.write(f"**Nombre de textes:** {corpus_info.get('total_texts', 0):,}")
                        st.write(f"**Nombre de mots:** {corpus_info.get('total_words', 0):,}")
                    
                    with col2:
                        if st.button(f"Analyser {corpus_name}", key=f"analyze_{corpus_name}"):
                            analyze_corpus(corpus_name)
                        
                        if st.button(f"Utiliser pour entraînement", key=f"train_{corpus_name}"):
                            st.info("Redirection vers l'entraînement de tokenizer...")
        else:
            st.info("Aucun corpus disponible. Uploadez votre premier corpus!")
    
    with tab2:
        st.subheader("Upload d'un Corpus Personnalisé")
        
        with st.form("upload_corpus_form"):
            corpus_name = st.text_input("Nom du corpus:", placeholder="mon_corpus_multilingue")
            
            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox("Langue:", ["auto", "en", "fr", "es", "de", "zh", "ja", "ar", "hi", "ru", "other"])
            with col2:
                uploaded_file = st.file_uploader("Fichier corpus:", type=['txt', 'json'])
            
            submitted = st.form_submit_button("Upload Corpus", type="primary")
            
            if submitted and uploaded_file and corpus_name:
                with st.spinner("Upload et analyse du corpus..."):
                    
                    files = {"file": uploaded_file.getvalue()}
                    data = {
                        "corpus_name": corpus_name,
                        "language": language,
                        "user_id": st.session_state.user_id
                    }
                    
                    result, error = call_tokenizer_api("/corpus/upload", method="POST", data=data, files=files)
                    
                    if result:
                        st.success("Corpus uploadé avec succès!")
                        st.json(result)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Erreur upload: {error}")

def analyze_corpus(corpus_name: str):
    """Analyse un corpus spécifique"""
    with st.spinner(f"Analyse du corpus {corpus_name}..."):
        
        analysis_request = {
            "corpus_name": corpus_name,
            "analysis_type": "comprehensive"
        }
        
        result, error = call_tokenizer_api("/corpus/analyze", method="POST", data=analysis_request)
        
        if result:
            st.success("Analyse terminée!")
            
            # Statistiques globales
            global_stats = result.get('global_statistics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Textes", global_stats.get('total_texts', 0))
            with col2:
                st.metric("Total Mots", f"{global_stats.get('total_words', 0):,}")
            with col3:
                st.metric("Caractères", f"{global_stats.get('total_characters', 0):,}")
            with col4:
                st.metric("Langues", global_stats.get('languages_count', 0))
            
            # Analyse par langue
            if 'language_analysis' in result:
                lang_analysis = result['language_analysis']
                
                st.subheader("Analyse par Langue")
                
                for lang, analysis in lang_analysis.items():
                    with st.expander(f"Langue: {lang}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Nombre de textes:** {analysis.get('text_count', 0)}")
                            st.write(f"**Mots uniques:** {analysis.get('unique_words', 0):,}")
                            st.write(f"**Richesse vocabulaire:** {analysis.get('vocabulary_richness', 0):.3f}")
                        
                        with col2:
                            # Mots les plus fréquents
                            if 'most_frequent_words' in analysis:
                                freq_words = analysis['most_frequent_words'][:10]
                                
                                words_df = pd.DataFrame(freq_words, columns=['Mot', 'Fréquence'])
                                fig = px.bar(words_df, x='Mot', y='Fréquence', title=f"Mots fréquents - {lang}")
                                fig.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse comparative
            if 'cross_language_analysis' in result:
                cross_analysis = result['cross_language_analysis']
                
                st.subheader("Analyse Comparative entre Langues")
                
                # Tailles de vocabulaire
                if 'vocabulary_sizes' in cross_analysis:
                    vocab_sizes = cross_analysis['vocabulary_sizes']
                    
                    fig = px.bar(
                        x=list(vocab_sizes.keys()),
                        y=list(vocab_sizes.values()),
                        title="Taille du vocabulaire par langue"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Longueurs moyennes des mots
                if 'avg_word_lengths' in cross_analysis:
                    word_lengths = cross_analysis['avg_word_lengths']
                    
                    fig = px.bar(
                        x=list(word_lengths.keys()),
                        y=list(word_lengths.values()),
                        title="Longueur moyenne des mots par langue"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Erreur d'analyse: {error}")

def show_tokenizer_comparison():
    """Page de comparaison de tokenizers"""
    st.title("⚖️ Comparaison de Tokenizers")
    
    st.markdown("""
    <div class="comparison-table">
        <h3>Comparez les performances de différents tokenizers</h3>
        <p>Testez plusieurs tokenizers sur les mêmes textes pour identifier le plus efficace</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration de la comparaison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Textes de Test")
        
        # Textes prédéfinis pour comparaison
        comparison_sets = {
            "Multilingue Standard": [
                "Hello, how are you today?",
                "Bonjour, comment allez-vous aujourd'hui?", 
                "Hola, ¿cómo estás hoy?",
                "Hallo, wie geht es dir heute?"
            ],
            "Technique/Code": [
                "import numpy as np\narray = np.zeros((10, 10))",
                "SELECT * FROM users WHERE age > 18;",
                "function calculateSum(a, b) { return a + b; }",
                "class MyClass:\n    def __init__(self):\n        pass"
            ],
            "Langues Complexes": [
                "这是一个中文句子，包含了复杂的字符。",
                "هذه جملة باللغة العربية مع أحرف معقدة.",
                "これは複雑な文字を含む日本語の文です。",
                "यह एक हिंदी वाक्य है जिसमें जटिल वर्ण हैं।"
            ]
        }
        
        selected_set = st.selectbox("Set de test prédéfini:", list(comparison_sets.keys()))
        
        # Zone pour textes personnalisés
        custom_texts = st.text_area(
            "Ou entrez vos propres textes (un par ligne):",
            height=150,
            value="\n".join(comparison_sets[selected_set])
        )
        
        test_texts = [line.strip() for line in custom_texts.split('\n') if line.strip()]
    
    with col2:
        st.subheader("Tokenizers à Comparer")
        
        available_tokenizers = list(get_available_tokenizers().keys())
        
        if available_tokenizers:
            selected_tokenizers = st.multiselect(
                "Sélectionner tokenizers:",
                available_tokenizers,
                default=available_tokenizers[:3]  # 3 premiers par défaut
            )
            
            compare_button = st.button(" Lancer Comparaison", type="primary", use_container_width=True)
        else:
            st.error("Aucun tokenizer disponible")
            compare_button = False
    
    # Lancement de la comparaison
    if compare_button and test_texts and selected_tokenizers:
        with st.spinner("Comparaison en cours..."):
            
            comparison_data = {
                "texts": test_texts,
                "tokenizer_names": selected_tokenizers
            }
            
            result, error = call_tokenizer_api("/compare/tokenizers", method="POST", data=comparison_data)
            
            if result:
                display_comparison_results(result, test_texts, selected_tokenizers)
            else:
                st.error(f"Erreur de comparaison: {error}")

def display_comparison_results(result: Dict, test_texts: List[str], tokenizers: List[str]):
    """Affiche les résultats de comparaison"""
    
    st.success("Comparaison terminée!")
    
    # Statistiques résumées
    if 'summary_statistics' in result:
        summary = result['summary_statistics']
        
        st.subheader("Résumé des Performances")
        
        # Créer un DataFrame pour les statistiques
        summary_data = []
        for tokenizer, stats in summary.items():
            summary_data.append({
                'Tokenizer': tokenizer,
                'Tokens Moyens': f"{stats.get('avg_token_count', 0):.1f}",
                'Ratio Compression': f"{stats.get('avg_compression_ratio', 0):.3f}",
                'Temps Moyen (ms)': f"{stats.get('avg_processing_time', 0)*1000:.1f}",
                'Consistance': f"{stats.get('consistency', 0):.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Graphiques de comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique des tokens moyens
            fig = px.bar(
                summary_df,
                x='Tokenizer',
                y='Tokens Moyens',
                title="Nombre moyen de tokens par tokenizer"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique des temps de traitement
            processing_times = [float(t.split()[0]) for t in summary_df['Temps Moyen (ms)']]
            fig = px.bar(
                x=summary_df['Tokenizer'],
                y=processing_times,
                title="Temps de traitement moyen (ms)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommandation
    if 'recommendation' in result:
        recommendation = result['recommendation']
        
        st.subheader("Recommandation")
        
        recommended_tokenizer = recommendation.get('recommended_tokenizer', 'N/A')
        confidence = recommendation.get('confidence_score', 0)
        reasoning = recommendation.get('reasoning', 'N/A')
        
        st.success(f"**Tokenizer recommandé:** `{recommended_tokenizer}`")
        st.write(f"**Score de confiance:** {confidence:.2f}")
        st.write(f"**Raisonnement:** {reasoning}")
        
        # Scores détaillés
        if 'all_scores' in recommendation:
            scores = recommendation['all_scores']
            
            fig = px.bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                title="Scores de performance par tokenizer",
                labels={'y': 'Score de performance', 'x': 'Tokenizer'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Résultats détaillés par texte
    if 'detailed_results' in result:
        detailed = result['detailed_results']
        
        st.subheader("Résultats Détaillés par Texte")
        
        for i, text_result in enumerate(detailed):
            with st.expander(f"Texte {i+1}: {text_result.get('text', 'N/A')[:50]}..."):
                
                # Créer un DataFrame pour ce texte
                text_data = []
                for tokenizer in tokenizers:
                    if tokenizer in text_result:
                        data = text_result[tokenizer]
                        text_data.append({
                            'Tokenizer': tokenizer,
                            'Tokens': data.get('token_count', 0),
                            'Ratio': f"{data.get('compression_ratio', 0):.3f}",
                            'Temps (ms)': f"{data.get('processing_time', 0)*1000:.1f}"
                        })
                
                if text_data:
                    text_df = pd.DataFrame(text_data)
                    st.dataframe(text_df, use_container_width=True)

def show_tokenizer_training():
    """Page d'entraînement de tokenizer personnalisé"""
    st.title(" Entraîner un Tokenizer Personnalisé")
    
    st.markdown("""
    <div class="training-progress">
        <h3>Créez votre Tokenizer Personnalisé</h3>
        <p>Entraînez un tokenizer adapté à vos données spécifiques</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("tokenizer_training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration du Tokenizer")
            
            tokenizer_name = st.text_input("Nom du tokenizer:", placeholder="mon_tokenizer_custom")
            
            tokenizer_type = st.selectbox("Type d'algorithme:", ["bpe", "wordpiece", "unigram", "wordlevel"])
            
            vocab_size = st.slider("Taille du vocabulaire:", 1000, 100000, 30000)
            
            min_frequency = st.slider("Fréquence minimale:", 1, 10, 2)
        
        with col2:
            st.subheader("Corpus d'Entraînement")
            
            available_corpus = get_available_corpus()
            corpus_names = list(available_corpus.keys()) if available_corpus else []
            
            selected_corpus = st.multiselect(
                "Sélectionner corpus:",
                corpus_names,
                default=corpus_names[:2] if len(corpus_names) >= 2 else corpus_names
            )
            
            st.subheader("Options Avancées")
            
            languages = st.multiselect(
                "Langues cibles:",
                ["multi", "en", "fr", "es", "de", "zh", "ja", "ar", "hi", "ru"],
                default=["multi"]
            )
            
            normalization = st.checkbox("Normalisation", value=True)
            lowercase = st.checkbox("Minuscules", value=True)
            strip_accents = st.checkbox("Supprimer accents", value=False)
        
        # Configuration des tokens spéciaux
        st.subheader("Tokens Spéciaux")
        special_tokens_input = st.text_input(
            "Tokens spéciaux (séparés par virgules):",
            value="<unk>,<pad>,<s>,</s>,<mask>"
        )
        special_tokens = [token.strip() for token in special_tokens_input.split(",") if token.strip()]
        
        submitted = st.form_submit_button(" Lancer l'Entraînement", type="primary")
        
        if submitted:
            # Validation des champs obligatoires
            if not tokenizer_name:
                st.error("Veuillez donner un nom au tokenizer")
                return
            
            if not selected_corpus:
                st.error("Veuillez sélectionner au moins un corpus")
                return
            
            # Construction de la configuration
            config = {
                "name": tokenizer_name,
                "tokenizer_type": tokenizer_type,
                "vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "languages": languages,
                "special_tokens": special_tokens,
                "normalization": normalization,
                "lowercase": lowercase,
                "strip_accents": strip_accents
            }
            
            # Construction de la requête d'entraînement
            training_request = {
                "name": tokenizer_name,
                "corpus_sources": selected_corpus,
                "config": config,
                "user_id": st.session_state.user_id
            }
            
            with st.spinner("Démarrage de l'entraînement..."):
                result, error = call_tokenizer_api("/tokenizer/train", method="POST", data=training_request)
                
                if result:
                    st.success("Entraînement démarré!")
                    st.json(result)
                    
                    training_id = result.get("training_id")
                    if training_id:
                        st.session_state.current_training = training_id
                        
                else:
                    st.error(f"Erreur d'entraînement: {error}")
    # monitor_training_progress(training_id)
    # Monitoring des entraînements en cours
    
    if hasattr(st.session_state, 'current_training'):
        st.divider()
        st.subheader("Suivi de l'Entraînement")
        monitor_training_progress(st.session_state.current_training)

def monitor_training_progress(training_id: str):
    """Monitore le progrès d'entraînement"""
    result, error = call_tokenizer_api(f"/tokenizer/training/{training_id}/status")
    
    if result:
        st.write(f"**Training ID:** {training_id}")
        st.write(f"**Nom:** {result.get('name', 'N/A')}")
        st.write(f"**Type:** {result.get('tokenizer_type', 'N/A')}")
        st.write(f"**Statut:** {result.get('status', 'N/A')}")
        
        status = result.get('status', 'unknown')
        
        if status == "active":
            st.success("✅ Entraînement terminé avec succès!")
            
            # Métriques de performance
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ratio Compression", f"{metrics.get('avg_compression_ratio', 0):.3f}")
                with col2:
                    st.metric("Caractères Traités", f"{metrics.get('total_chars_processed', 0):,}")
                with col3:
                    st.metric("Score Efficacité", f"{metrics.get('efficiency_score', 0):.2f}")
            
            # Option de test du nouveau tokenizer (en dehors du form)
            st.subheader("Tester le Nouveau Tokenizer")
            
            # Créer un nouveau form pour le test
            with st.form(f"test_tokenizer_{training_id}"):
                test_text = st.text_area("Texte de test:", placeholder="Entrez un texte pour tester votre nouveau tokenizer...")
                test_submitted = st.form_submit_button("Tester")
                
                if test_submitted and test_text:
                    st.info("Fonctionnalité de test en développement...")
                    # Note: Il faudrait d'abord charger le tokenizer personnalisé dans l'API
        
        elif status == "failed":
            st.error("❌ Entraînement échoué")
        
        elif status == "pending":
            st.info("⏳ Entraînement en cours...")
            
            # Auto-refresh
            if st.button("Actualiser"):
                st.rerun()
    else:
        st.error(f"Erreur récupération statut: {error}")

def show_analytics_dashboard():
    """Dashboard d'analytiques de la plateforme"""
    st.title(" Analytics & Statistiques")
    
    # Récupérer les statistiques avec gestion d'erreur
    try:
        # stats = get_platform_statistics()
        stats = asyncio.run(get_platform_statistics())
        
        if not stats or not isinstance(stats, dict):
            st.warning("Statistiques temporairement indisponibles. Vérifiez que l'API Tokenizer est démarrée.")
            # Afficher une interface simplifiée
            st.info("Démarrez l'API Tokenizer pour voir les statistiques complètes")
            return
        
        platform_stats = stats.get('platform_stats', {})
        usage_patterns = stats.get('usage_patterns', {})
        system_info = stats.get('system_info', {})
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des statistiques: {e}")
        st.info("Vérifiez que l'API Tokenizer est accessible sur http://localhost:8008")
        return
    
    # Métriques principales
    st.subheader("Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{platform_stats.get('total_tokenizations', 0):,}</h3>
            <p>Tokenisations Total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{platform_stats.get('available_tokenizers', 0)}</h3>
            <p>Tokenizers Disponibles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{platform_stats.get('available_corpus', 0)}</h3>
            <p>Corpus Disponibles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{platform_stats.get('custom_tokenizers_trained', 0)}</h3>
            <p>Tokenizers Entraînés</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Patterns d'usage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tokenizers les Plus Utilisés")
        
        most_used = usage_patterns.get('most_used_tokenizers', [])
        if most_used:
            usage_df = pd.DataFrame(most_used)
            
            fig = px.bar(
                usage_df,
                x='usage',
                y='tokenizer',
                orientation='h',
                title="Popularité des tokenizers"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée d'usage disponible")
    
    with col2:
        st.subheader("Langues Supportées")
        
        supported_langs = system_info.get('supported_languages', [])
        if supported_langs:
            # Créer un graphique de répartition des langues
            lang_counts = {lang: 1 for lang in supported_langs}  # Données simulées
            
            fig = px.pie(
                values=list(lang_counts.values()),
                names=list(lang_counts.keys()),
                title="Répartition des langues supportées"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Informations système
    st.subheader("Informations Système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Types de Tokenizers:**")
        tokenizer_types = system_info.get('tokenizer_types', [])
        for ttype in tokenizer_types:
            st.write(f"• {ttype}")
        
        st.write(f"**Taille max vocabulaire:** {system_info.get('max_vocab_size', 0):,}")
    
    with col2:
        st.write("**Écritures Supportées:**")
        supported_scripts = system_info.get('supported_scripts', [])
        for script in supported_scripts:
            st.write(f"• {script.title()}")
    
    # Graphique temporel (simulé)
    st.subheader("Évolution d'Usage (Simulation)")
    
    # Générer des données temporelles simulées
    dates = pd.date_range(start="2024-01-01", end=datetime.now(), freq='D')
    usage_data = np.random.poisson(10, len(dates)) + np.random.randint(0, 50, len(dates))
    
    usage_df = pd.DataFrame({
        'Date': dates,
        'Tokenisations': usage_data
    })
    
    fig = px.line(usage_df, x='Date', y='Tokenisations', title="Usage quotidien de la plateforme")
    st.plotly_chart(fig, use_container_width=True)

def show_ai_integration():
    """Page d'intégration avec la plateforme d'entraînement IA"""
    st.title(" Intégration IA Training Platform")
    
    st.markdown("""
    <div class="tokenizer-card">
        <h3>Connectez vos Tokenizers à l'Entraînement IA</h3>
        <p>Utilisez vos tokenizers personnalisés pour entraîner des modèles d'IA avancés</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier la connexion à l'API d'entraînement
    try:
        ai_response = requests.get(f"{AI_TRAINING_API_URL}/health", timeout=5)
        ai_platform_available = ai_response.status_code == 200
    except:
        ai_platform_available = False
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if ai_platform_available:
            st.success(" AI Training Platform connectée")
            
            st.subheader("Workflow Intégré")
            
            st.write("**Étapes du workflow intégré:**")
            st.write("1.  Préparer votre corpus de données")
            st.write("2.  Entraîner un tokenizer personnalisé")
            st.write("3.  Utiliser le tokenizer pour entraîner un modèle IA")
            st.write("4.  Analyser les performances du modèle")
            st.write("5.  Déployer le modèle en production")
            
            # Sélection du tokenizer à utiliser
            st.subheader("Exporter un Tokenizer pour l'IA")
            
            tokenizers = get_available_tokenizers()
            if tokenizers:
                selected_tokenizer = st.selectbox(
                    "Choisir un tokenizer à exporter:",
                    list(tokenizers.keys())
                )
                
                if selected_tokenizer:
                    tokenizer_info = tokenizers[selected_tokenizer]
                    
                    st.write("**Informations du tokenizer:**")
                    st.write(f"- Type: {tokenizer_info.get('type', 'N/A')}")
                    st.write(f"- Taille vocabulaire: {tokenizer_info.get('vocab_size', 'N/A')}")
                    st.write(f"- Longueur max: {tokenizer_info.get('model_max_length', 'N/A')}")
                    
                    if st.button(" Exporter vers AI Training", type="primary"):
                        export_tokenizer_to_ai_platform(selected_tokenizer, tokenizer_info)
        else:
            st.error(" AI Training Platform non disponible")
            st.write("Démarrez l'API d'entraînement IA pour activer l'intégration")
    
    with col2:
        # Bouton de redirection
        st.subheader("Accès Direct")
        
        if st.button(" Ouvrir AI Training Platform", use_container_width=True):
            # Note: Dans une vraie app, ceci ouvrirait un nouvel onglet
            st.markdown(f"""
            <script>
                window.open('{AI_TRAINING_API_URL.replace('8006', '8007')}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.info(f"Ouvrir: {AI_TRAINING_API_URL.replace('8006', '8007')}")
        
        # Statistiques d'intégration
        st.subheader("Stats Intégration")
        
        # Données simulées
        integration_stats = {
            "Modèles entraînés avec tokenizers personnalisés": 5,
            "Tokenizers exportés": 3,
            "Projets intégrés": 12
        }
        
        for stat_name, value in integration_stats.items():
            st.metric(stat_name, value)
    
    # Guide d'intégration
    st.divider()
    st.subheader(" Guide d'Intégration")
    
    with st.expander("Comment utiliser vos tokenizers avec l'IA Training"):
        st.markdown("""
        ### Étapes détaillées:
        
        1. **Entraînement du Tokenizer**
           - Uploadez votre corpus dans la section "Corpus Manager"
           - Configurez et entraînez votre tokenizer personnalisé
           - Testez les performances de tokenisation
        
        2. **Export vers AI Training**
           - Sélectionnez votre tokenizer dans cette section
           - Cliquez sur "Exporter vers AI Training"
           - Le tokenizer sera disponible dans la plateforme d'entraînement
        
        3. **Entraînement du Modèle IA**
           - Allez sur la AI Training Platform
           - Créez un nouveau modèle
           - Sélectionnez votre tokenizer personnalisé
           - Configurez et lancez l'entraînement
        
        4. **Optimisation**
           - Analysez les performances du modèle
           - Ajustez le tokenizer si nécessaire
           - Re-entraînez le modèle avec le tokenizer optimisé
        
        ### Avantages:
        - **Performance améliorée** avec des tokenizers adaptés à vos données
        - **Cohérence** entre preprocessing et entraînement
        - **Flexibilité** pour différents domaines et langues
        """)

def export_tokenizer_to_ai_platform(tokenizer_name: str, tokenizer_info: Dict):
    """Exporte un tokenizer vers la plateforme d'entraînement IA"""
    with st.spinner("Export du tokenizer..."):
        try:
            # Simuler l'export (dans la réalité, ceci ferait un appel API)
            time.sleep(2)
            
            st.success(f"✅ Tokenizer '{tokenizer_name}' exporté avec succès!")
            
            st.info(f"""
            Le tokenizer est maintenant disponible dans la AI Training Platform:
            - Nom: {tokenizer_name}
            - Type: {tokenizer_info.get('type', 'N/A')}
            - Vocabulaire: {tokenizer_info.get('vocab_size', 'N/A')} tokens
            
            Vous pouvez maintenant l'utiliser pour entraîner vos modèles d'IA.
            """)
            
        except Exception as e:
            st.error(f"❌ Erreur d'export: {e}")


def safe_api_call(endpoint, method="GET", data=None, files=None, default_return=None, timeout=30):
    """
    Effectue un appel API en toute sécurité et gère les erreurs.
    
    Args:
        endpoint (str): URL de l'API.
        method (str): Méthode HTTP ("GET", "POST", "PUT", "DELETE").
        data (dict, optional): Données à envoyer (pour POST/PUT).
        files (dict, optional): Fichiers à envoyer.
        default_return: Valeur à retourner en cas d'erreur.
        timeout (int, optional): Timeout en secondes pour la requête.

    Returns:
        tuple: (response_json, error_message)
            - response_json: dict de la réponse si succès, sinon default_return
            - error_message: None si succès, sinon message d'erreur
    """
    try:
        method = method.upper()
        if method == "GET":
            response = requests.get(endpoint, timeout=timeout)
        elif method in ["POST", "PUT", "DELETE"]:
            response = requests.request(method, endpoint, json=data, files=files, timeout=timeout)
        else:
            return default_return, f"Méthode HTTP non supportée: {method}"

        # Vérifie le code HTTP
        if response.status_code >= 200 and response.status_code < 300:
            try:
                return response.json(), None
            except ValueError:
                return response.text, None  # Si la réponse n'est pas JSON
        else:
            return default_return, f"Erreur API ({response.status_code}): {response.text}"

    except requests.Timeout:
        return default_return, "Erreur: Timeout lors de l'appel API."
    except requests.ConnectionError:
        return default_return, "Erreur: Problème de connexion à l'API."
    except Exception as e:
        return default_return, f"Erreur inattendue: {str(e)}"

# Fonction principale
def main():
    st.markdown('<h1 class="main-header"> Universal Tokenizer Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        page = st.selectbox(
            "Sections:",
            [
                "Tokenizer Playground", 
                "Analyse de Langue", 
                "Corpus Manager", 
                "Comparaison de Tokenizers",
                "Entraîner Tokenizer",
                "Analytics & Stats",
                "Intégration IA Training"
            ]
        )
        
        st.divider()
        
        # Statistiques en temps réel
        st.subheader("Statistiques Platform")
        
        # Utiliser la fonction API sécurisée
        stats, error = safe_api_call("/statistics/usage", default_return={})
        
        if stats and isinstance(stats, dict):
            platform_stats = stats.get("platform_stats", {})
            
            if platform_stats:
                st.metric("Tokenisations", platform_stats.get("total_tokenizations", 0))
                st.metric("Tokenizers", platform_stats.get("available_tokenizers", 0))
                st.metric("Corpus", platform_stats.get("available_corpus", 0))
                st.metric("Custom Trained", platform_stats.get("custom_tokenizers_trained", 0))
            else:
                st.info("Stats en chargement...")
        else:
            st.info("Stats indisponibles")
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Session Info")
        st.write(f"User ID: `{st.session_state.user_id[:8]}...`")
        st.write(f"History: {len(st.session_state.tokenization_history)} tokenizations")
        
        # Statut API
        try:
            health, error = call_tokenizer_api("/health")
            if health and isinstance(health, dict):
                st.success("🟢 Tokenizer API en ligne")
                version = health.get('version', 'N/A')
                if version != 'N/A':
                    st.write(f"Version: {version}")
            elif error:
                st.error(f"🔴 API hors ligne: {error}")
            else:
                st.warning("🟡 Statut API inconnu")
        except Exception as e:
            st.error(f"🔴 Erreur connexion API: {e}")
    
    # Pages principales
    if page == "Tokenizer Playground":
        show_tokenizer_playground()
    elif page == "Analyse de Langue":
        show_language_analysis()
    elif page == "Corpus Manager":
        show_corpus_manager()
    elif page == "Comparaison de Tokenizers":
        show_tokenizer_comparison()
    elif page == "Entraîner Tokenizer":
        show_tokenizer_training()
    elif page == "Analytics & Stats":
        show_analytics_dashboard()
    elif page == "Intégration IA Training":
        show_ai_integration()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Universal Tokenizer Platform**")
        st.markdown("Tokenisation multilingue avancée")
    
    with col2:
        st.markdown("**Fonctionnalités:**")
        st.markdown("- Tokenisation multilingue")
        st.markdown("- Corpus personnalisés")
        st.markdown("- Entraînement de tokenizers")
        st.markdown("- Intégration IA")
    
    with col3:
        st.markdown("**Support:**")
        st.markdown("Version 1.0.0")
        st.markdown(f"Dernière MAJ: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()

