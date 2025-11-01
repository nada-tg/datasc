# media_intelligence_dashboard.py - Interface Streamlit pour Media Intelligence Platform

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import uuid
from PIL import Image
import io

# Configuration de la page streamlit run media_intelligence_platform_app.py
st.set_page_config(
    page_title="Media Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration API
MEDIA_API_URL = "http://localhost:8032"
PERSONAL_DATA_URL = "http://localhost:8504"
AUTOSCI_URL = "http://localhost:8501"

# Style CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF6B35;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.upload-card {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 15px;
    border: 2px dashed #FF6B35;
    text-align: center;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.upload-card:hover {
    border-color: #F7931E;
    background-color: #fff;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.media-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
    border-left: 4px solid #FF6B35;
}
.feature-badge {
    background-color: #FF6B35;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# État de session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'

# Fonctions utilitaires
def call_media_api(endpoint, method="GET", data=None, files=None):
    """Appel API Media Intelligence Platform"""
    url = f"{MEDIA_API_URL}{endpoint}"
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

def get_system_usage_data():
    """Récupère les données d'utilisation des applications/sites"""
    # Simulation des données d'utilisation système
    # Dans la réalité, ceci se connecterait aux APIs système
    sample_data = [
        {"platform": "Google Chrome", "data_used_mb": 450, "category": "Navigation", "active_time": 120},
        {"platform": "Microsoft Teams", "data_used_mb": 320, "category": "Communication", "active_time": 90},
        {"platform": "Spotify", "data_used_mb": 200, "category": "Streaming", "active_time": 180},
        {"platform": "Visual Studio Code", "data_used_mb": 150, "category": "Développement", "active_time": 240},
        {"platform": "WhatsApp Desktop", "data_used_mb": 100, "category": "Communication", "active_time": 60},
        {"platform": "Adobe Photoshop", "data_used_mb": 80, "category": "Création", "active_time": 45},
        {"platform": "YouTube", "data_used_mb": 600, "category": "Streaming", "active_time": 150},
        {"platform": "Microsoft Word", "data_used_mb": 50, "category": "Bureautique", "active_time": 75},
    ]
    return pd.DataFrame(sample_data)

# Interface principale
def main():
    st.markdown('<h1 class="main-header">Media Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar avec navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Boutons de navigation vers autres plateformes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("AutoSci ML", use_container_width=True, type="secondary"):
                st.markdown(f'<meta http-equiv="refresh" content="0; url={AUTOSCI_URL}">', unsafe_allow_html=True)
        with col2:
            if st.button("Personal Data", use_container_width=True, type="secondary"):
                st.markdown(f'<meta http-equiv="refresh" content="0; url={PERSONAL_DATA_URL}">', unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation interne
        page = st.selectbox(
            "Sections:",
            ["Upload Media", "Mes Médias", "Analyses", "Études ML", "Marketplace", "Usage Système"]
        )
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID: `{st.session_state.user_id[:8]}...`")
        
        # Statut API
        health, error = call_media_api("/health")
        if health:
            st.success("API Media en ligne")
        else:
            st.error("API Media hors ligne")
    
    # Pages principales
    if page == "Upload Media":
        show_upload_page()
    elif page == "Mes Médias":
        show_media_management_page()
    elif page == "Analyses":
        show_analyses_page()
    elif page == "Études ML":
        show_studies_page()
    elif page == "Marketplace":
        show_marketplace_page()
    elif page == "Usage Système":
        show_system_usage_page()

def show_upload_page():
    """Page d'upload de médias"""
    st.title("Upload et Analyse Multimodale")
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>Uploadez vos médias pour extraction intelligente de données</h3>
        <p>Notre IA extrait automatiquement toutes les données possibles de vos images, vidéos, audios et textes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zone d'upload avec 4 boutons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="upload-card">
            <h4>📸 Images</h4>
            <p>JPG, PNG, GIF, TIFF...</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Choisir une image",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            key="image_upload"
        )
        
        if uploaded_image:
            process_uploaded_file(uploaded_image, "image")
    
    with col2:
        st.markdown("""
        <div class="upload-card">
            <h4>🎥 Vidéos</h4>
            <p>MP4, AVI, MOV, WMV...</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader(
            "Choisir une vidéo",
            type=['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'],
            key="video_upload"
        )
        
        if uploaded_video:
            process_uploaded_file(uploaded_video, "video")
    
    with col3:
        st.markdown("""
        <div class="upload-card">
            <h4>🎵 Audio</h4>
            <p>MP3, WAV, FLAC, AAC...</p>

        </div>
        """, unsafe_allow_html=True)
        
        uploaded_audio = st.file_uploader(
            "Choisir un fichier audio",
            type=['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a'],
            key="audio_upload"
        )
        
        if uploaded_audio:
            process_uploaded_file(uploaded_audio, "audio")
    
    with col4:
        st.markdown("""
        <div class="upload-card">
            <h4>📄 Textes</h4>
            <p>TXT, MD, DOC, RTF...</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_text = st.file_uploader(
            "Choisir un fichier texte",
            type=['txt', 'md', 'rtf', 'doc', 'docx'],
            key="text_upload"
        )
        
        if uploaded_text:
            process_uploaded_file(uploaded_text, "text")
    
    # Zone de saisie de texte direct
    st.divider()
    st.subheader("Ou saisissez du texte directement")
    
    text_input = st.text_area(
        "Entrez votre texte à analyser:",
        height=150,
        placeholder="Collez ou tapez votre texte ici pour une analyse NLP complète..."
    )
    
    if text_input and st.button("Analyser le texte", type="primary"):
        # Créer un fichier temporaire pour le texte
        temp_filename = f"temp_text_{uuid.uuid4().hex[:8]}.txt"
        temp_file = io.StringIO(text_input)
        temp_file.name = temp_filename
        
        process_text_input(text_input, temp_filename)

def process_uploaded_file(uploaded_file, media_type):
    """Traite un fichier uploadé"""
    with st.spinner(f"Upload et traitement du fichier {media_type}..."):
        try:
            # Préparer le fichier pour l'API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"user_id": st.session_state.user_id}
            
            # Appel API d'upload
            result, error = call_media_api("/media/upload", method="POST", data=data, files=files)
            
            if result:
                st.success(f"✅ Fichier {uploaded_file.name} uploadé avec succès!")
                
                # Afficher les détails
                with st.expander("Détails du traitement", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Media ID", result["media_id"][:8] + "...")
                    with col2:
                        st.metric("Type", result["media_type"])
                    with col3:
                        st.metric("Statut", result["status"])
                
                st.info("🔄 Extraction des données en cours... Consultez 'Mes Médias' dans quelques instants.")
                
                # Auto-refresh suggestions
                if st.button("Rafraîchir la page"):
                    st.rerun()
                    
            else:
                st.error(f"❌ Erreur lors de l'upload: {error}")
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

def process_text_input(text_content, filename):
    """Traite une saisie de texte directe"""
    with st.spinner("Analyse du texte en cours..."):
        try:
            # Créer un fichier en mémoire
            text_file = io.BytesIO(text_content.encode('utf-8'))
            text_file.name = filename
            
            files = {"file": (filename, text_file.getvalue(), "text/plain")}
            data = {"user_id": st.session_state.user_id}
            
            result, error = call_media_api("/media/upload", method="POST", data=data, files=files)
            
            if result:
                st.success("✅ Texte analysé avec succès!")
                
                with st.expander("Aperçu de l'analyse", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Longueur du texte:**", len(text_content), "caractères")
                        st.write("**Mots:**", len(text_content.split()))
                    with col2:
                        st.write("**Media ID:**", result["media_id"][:12] + "...")
                        st.write("**Statut:**", result["status"])
                
                st.info("🔄 Analyse NLP complète en cours...")
            else:
                st.error(f"❌ Erreur: {error}")
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

def show_media_management_page():
    """Page de gestion des médias"""
    st.title("Gestion de Mes Médias")
    
    # Récupérer les médias de l'utilisateur
    media_data, error = call_media_api(f"/media/user/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        return
    
    if not media_data or not media_data.get("media_files"):
        st.info("Aucun média uploadé pour le moment. Utilisez la section 'Upload Media' pour commencer.")
        return
    
    media_files = media_data["media_files"]
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Médias", len(media_files))
    with col2:
        completed = len([m for m in media_files if m["processing_status"] == "completed"])
        st.metric("Traités", completed)
    with col3:
        total_size = sum([m["file_size"] for m in media_files]) / (1024*1024)  # MB
        st.metric("Taille Totale", f"{total_size:.1f} MB")
    with col4:
        media_types = len(set([m["media_type"] for m in media_files]))
        st.metric("Types Différents", media_types)
    
    # Filtrages
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox("Filtrer par type:", 
                                 ["Tous"] + list(set([m["media_type"] for m in media_files])))
    with col2:
        status_filter = st.selectbox("Filtrer par statut:",
                                   ["Tous"] + list(set([m["processing_status"] for m in media_files])))
    
    # Appliquer les filtres
    filtered_files = media_files
    if type_filter != "Tous":
        filtered_files = [m for m in filtered_files if m["media_type"] == type_filter]
    if status_filter != "Tous":
        filtered_files = [m for m in filtered_files if m["processing_status"] == status_filter]
    
    # Affichage des médias
    for media in filtered_files:
        with st.container():
            st.markdown(f"""
            <div class="media-card">
                <h4>📁 {media['filename']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            
            with col1:
                st.write(f"**Type:** {media['media_type']}")
                st.write(f"**Taille:** {media['file_size'] / 1024:.1f} KB")
            
            with col2:
                status_color = "🟢" if media["processing_status"] == "completed" else "🟡" if media["processing_status"] == "pending" else "🔴"
                st.write(f"**Statut:** {status_color} {media['processing_status']}")
            
            with col3:
                extraction_status = "✅" if media["has_extracted_data"] else "⏳"
                st.write(f"**Données:** {extraction_status}")
            
            with col4:
                # Actions disponibles
                if media["processing_status"] == "completed":
                    col_analyze, col_study = st.columns(2)
                    with col_analyze:
                        if st.button(f"Analyser", key=f"analyze_{media['media_id']}"):
                            start_media_analysis(media['media_id'])
                    with col_study:
                        if st.button(f"Étude ML", key=f"study_{media['media_id']}"):
                            start_media_study(media['media_id'])
                else:
                    st.write("En traitement...")
            
            st.divider()

def start_media_analysis(media_id):
    """Lance une analyse de média"""
    with st.spinner("Lancement de l'analyse..."):
        data = {
            "media_id": media_id,
            "user_id": st.session_state.user_id,
            "analysis_type": "comprehensive"
        }
        
        result, error = call_media_api("/media/analyze", method="POST", data=data)
        
        if result:
            st.success(f"✅ Analyse lancée! ID: {result['analysis_id'][:12]}...")
            st.info("Consultez la section 'Analyses' pour suivre le progrès.")
        else:
            st.error(f"❌ Erreur: {error}")

def start_media_study(media_id):
    """Lance une étude ML de média"""
    with st.spinner("Lancement de l'étude ML..."):
        data = {
            "media_id": media_id,
            "user_id": st.session_state.user_id,
            "target_task": "auto"
        }
        
        result, error = call_media_api("/media/study", method="POST", data=data)
        
        if result:
            st.success(f"✅ Étude ML lancée! ID: {result['study_id'][:12]}...")
            st.info("Consultez la section 'Études ML' pour suivre le progrès.")
        else:
            st.error(f"❌ Erreur: {error}")

def show_analyses_page():
    """Page des analyses multimodales"""
    st.title("Analyses Multimodales")
    
    # Récupérer les analyses
    analyses_data, error = call_media_api(f"/media/analyses/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur: {error}")
        return
    
    if not analyses_data or not analyses_data.get("analyses"):
        st.info("Aucune analyse disponible. Analysez vos médias depuis 'Mes Médias'.")
        return
    
    analyses = analyses_data["analyses"]
    
    # Métriques des analyses
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyses", len(analyses))
    with col2:
        completed = len([a for a in analyses if a["status"] == "completed"])
        st.metric("Terminées", completed)
    with col3:
        in_progress = len([a for a in analyses if a["status"] == "in_progress"])
        st.metric("En cours", in_progress)
    
    # Affichage des analyses
    for analysis in analyses:
        with st.expander(f"📊 Analyse: {analysis['filename']} - {analysis['analysis_type']}", 
                        expanded=(analysis["status"] == "completed")):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Fichier:** {analysis['filename']}")
                st.write(f"**Type de média:** {analysis['media_type']}")
                st.write(f"**Type d'analyse:** {analysis['analysis_type']}")
                st.write(f"**Créée le:** {analysis['created_at']}")
            
            with col2:
                status_color = {"completed": "🟢", "in_progress": "🟡", "failed": "🔴"}.get(analysis["status"], "⚪")
                st.write(f"**Statut:** {status_color} {analysis['status']}")
                st.write(f"**ID:** {analysis['analysis_id'][:12]}...")
            
            if analysis["status"] == "completed" and analysis["results"]:
                st.subheader("Résultats de l'Analyse")
                
                results = analysis["results"]
                
                # Onglets pour différents types de résultats
                tabs = st.tabs(["Statistiques", "Patterns", "Anomalies", "Recommandations", "Visualisations"])
                
                with tabs[0]:  # Statistiques
                    if "summary_statistics" in results:
                        display_summary_statistics(results["summary_statistics"])
                
                with tabs[1]:  # Patterns
                    if "pattern_detection" in results:
                        st.json(results["pattern_detection"])
                
                with tabs[2]:  # Anomalies
                    if "anomaly_detection" in results:
                        st.json(results["anomaly_detection"])
                
                with tabs[3]:  # Recommandations
                    if "recommendations" in results:
                        st.json(results["recommendations"])
                
                with tabs[4]:  # Visualisations
                    if analysis.get("visualizations"):
                        display_visualizations(analysis["visualizations"])
            
            elif analysis["status"] == "failed":
                st.error("❌ Analyse échouée")
                if "error" in analysis.get("results", {}):
                    st.error(f"Détail: {analysis['results']['error']}")

def display_summary_statistics(stats):
    """Affiche les statistiques résumées"""
    try:
        # Créer des métriques pour les statistiques numériques
        numeric_stats = {}
        
        for feature, stat in stats.items():
            if isinstance(stat, dict):
                if "mean" in stat:
                    numeric_stats[feature] = {
                        "mean": round(stat["mean"], 3),
                        "std": round(stat.get("std", 0), 3),
                        "min": round(stat.get("min", 0), 3),
                        "max": round(stat.get("max", 0), 3)
                    }
        
        if numeric_stats:
            # Affichage en colonnes
            cols = st.columns(min(3, len(numeric_stats)))
            for i, (feature, values) in enumerate(numeric_stats.items()):
                with cols[i % len(cols)]:
                    st.metric(f"{feature} (moyenne)", values["mean"])
                    st.write(f"Écart-type: {values['std']}")
                    st.write(f"Min-Max: {values['min']} - {values['max']}")
        else:
            st.json(stats)
            
    except Exception as e:
        st.error(f"Erreur affichage statistiques: {e}")
        st.json(stats)

def display_visualizations(visualizations):
    """Affiche les visualisations Plotly"""
    try:
        for viz_name, viz_data in visualizations.items():
            if viz_data:
                st.subheader(viz_name.replace("_", " ").title())
                
                # Charger et afficher le graphique Plotly
                import plotly.graph_objects as go
                fig_dict = json.loads(viz_data)
                fig = go.Figure(fig_dict)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Erreur affichage visualisations: {e}")
        st.json(visualizations)

def show_studies_page():
    """Page des études ML"""
    st.title("Études Machine Learning")
    
    # Récupérer les études
    studies_data, error = call_media_api(f"/media/studies/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur: {error}")
        return
    
    if not studies_data or not studies_data.get("studies"):
        st.info("Aucune étude ML disponible. Lancez des études depuis 'Mes Médias'.")
        return
    
    studies = studies_data["studies"]
    
    # Métriques des études
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Études", len(studies))
    with col2:
        completed = len([s for s in studies if s["status"] == "completed"])
        st.metric("Terminées", completed)
    with col3:
        tasks = len(set([s["target_task"] for s in studies]))
        st.metric("Types de Tâches", tasks)
    
    # Affichage des études
    for study in studies:
        with st.expander(f"🤖 Étude ML: {study['filename']} - {study['target_task']}", 
                        expanded=(study["status"] == "completed")):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Fichier:** {study['filename']}")
                st.write(f"**Type de média:** {study['media_type']}")
                st.write(f"**Tâche cible:** {study['target_task']}")
                st.write(f"**Entraînée le:** {study['trained_at']}")
            
            with col2:
                status_color = {"completed": "🟢", "in_progress": "🟡", "failed": "🔴"}.get(study["status"], "⚪")
                st.write(f"**Statut:** {status_color} {study['status']}")
                st.write(f"**ID:** {study['study_id'][:12]}...")
                if study.get("model_path"):
                    st.write("**Modèle:** ✅ Sauvegardé")
            
            if study["status"] == "completed" and study["performance_metrics"]:
                st.subheader("Métriques de Performance")
                
                metrics = study["performance_metrics"]
                
                # Affichage selon le type de tâche
                if study["target_task"] == "clustering" or metrics.get("task") == "clustering":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nombre de Clusters", metrics.get("n_clusters", "N/A"))
                    with col2:
                        silhouette = metrics.get("silhouette_score", 0)
                        st.metric("Score Silhouette", f"{silhouette:.3f}")
                    with col3:
                        inertia = metrics.get("inertia", 0)
                        st.metric("Inertie", f"{inertia:.2f}")
                
                # Détails complets
                with st.expander("Détails complets des métriques"):
                    st.json(metrics)
            
            elif study["status"] == "failed":
                st.error("❌ Étude échouée")
                if "error" in study.get("performance_metrics", {}):
                    st.error(f"Détail: {study['performance_metrics']['error']}")

def show_marketplace_page():
    """Page du marketplace de médias"""
    st.title("Marketplace de Médias")
    
    tab1, tab2 = st.tabs(["Parcourir", "Mes Offres"])
    
    with tab1:
        st.subheader("Médias Disponibles à l'Achat")
        
        # Récupérer les offres du marketplace
        listings_data, error = call_media_api("/media/marketplace/listings")
        
        if error:
            st.error(f"Erreur: {error}")
            return
        
        if not listings_data or not listings_data.get("listings"):
            st.info("Aucune offre disponible sur le marketplace.")
            return
        
        listings = listings_data["listings"]
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.selectbox("Type de média:", 
                                     ["Tous"] + list(set([l["media_type"] for l in listings])))
        with col2:
            max_price = st.slider("Prix maximum:", 0, 100, 50)
        
        # Filtrer les offres
        filtered_listings = listings
        if type_filter != "Tous":
            filtered_listings = [l for l in filtered_listings if l["media_type"] == type_filter]
        filtered_listings = [l for l in filtered_listings if l["price"] <= max_price]
        
        # Afficher les offres
        for listing in filtered_listings:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{listing['filename']}**")
                    st.write(f"Type: {listing['media_type']}")
                    st.write(f"Description: {listing['description']}")
                    st.write(f"Vendeur: {listing['seller_id'][:8]}...")
                
                with col2:
                    st.metric("Prix", f"{listing['price']}€")
                    st.write(f"Publié: {listing['created_at'][:10]}")
                
                with col3:
                    if st.button(f"Acheter", key=f"buy_{listing['listing_id']}"):
                        st.success("🛒 Fonctionnalité d'achat à implémenter!")
                        st.info("Contact vendeur en cours...")
                
                st.divider()
    
    with tab2:
        st.subheader("Créer une Offre de Vente")
        
        # Récupérer les médias de l'utilisateur pour la vente
        media_data, error = call_media_api(f"/media/user/{st.session_state.user_id}")
        
        if error or not media_data or not media_data.get("media_files"):
            st.info("Vous devez d'abord uploader des médias pour les vendre.")
            return
        
        # Sélection du média à vendre
        completed_media = [m for m in media_data["media_files"] if m["processing_status"] == "completed"]
        
        if not completed_media:
            st.info("Aucun média traité disponible pour la vente.")
            return
        
        media_options = {f"{m['filename']} ({m['media_type']})": m['media_id'] for m in completed_media}
        
        with st.form("create_listing"):
            selected_media = st.selectbox("Choisir un média à vendre:", list(media_options.keys()))
            price = st.number_input("Prix de vente (€):", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
            description = st.text_area("Description de l'offre:", 
                                     placeholder="Décrivez votre média, ses caractéristiques, son utilité...")
            
            if st.form_submit_button("Créer l'offre", type="primary"):
                create_marketplace_listing(media_options[selected_media], price, description)

def create_marketplace_listing(media_id, price, description):
    """Crée une offre sur le marketplace"""
    with st.spinner("Création de l'offre..."):
        try:
            # Paramètres pour l'API
            params = {
                "media_id": media_id,
                "seller_id": st.session_state.user_id,
                "price": price,
                "description": description
            }
            
            # Construire l'URL avec les paramètres
            url_params = "&".join([f"{k}={v}" for k, v in params.items()])
            
            result, error = call_media_api(f"/media/marketplace/list?{url_params}", method="POST")
            
            if result:
                st.success("✅ Offre créée avec succès!")
                st.balloons()
                st.info("Votre média est maintenant disponible sur le marketplace.")
            else:
                st.error(f"❌ Erreur lors de la création: {error}")
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

def show_system_usage_page():
    """Page d'analyse d'usage système"""
    st.title("Analyse d'Usage Système")
    
    st.markdown("""
    Cette section analyse votre utilisation des applications et sites web pour identifier
    les patterns de consommation de données et optimiser votre productivité.
    """)
    
    # Récupérer les données d'usage
    usage_data = get_system_usage_data()
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_data = usage_data['data_used_mb'].sum()
        st.metric("Données Totales", f"{total_data:.0f} MB")
    with col2:
        total_time = usage_data['active_time'].sum()
        st.metric("Temps Total", f"{total_time:.0f} min")
    with col3:
        avg_efficiency = total_data / total_time if total_time > 0 else 0
        st.metric("Efficacité", f"{avg_efficiency:.1f} MB/min")
    with col4:
        top_category = usage_data.groupby('category')['data_used_mb'].sum().idxmax()
        st.metric("Top Catégorie", top_category)
    
    # Graphiques d'analyse
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Consommation de Données par Application")
        fig_data = px.bar(usage_data.sort_values('data_used_mb', ascending=True), 
                         x='data_used_mb', y='platform', 
                         orientation='h',
                         title="Données utilisées (MB)")
        st.plotly_chart(fig_data, use_container_width=True)
    
    with col2:
        st.subheader("Répartition par Catégorie")
        category_data = usage_data.groupby('category')['data_used_mb'].sum().reset_index()
        fig_pie = px.pie(category_data, values='data_used_mb', names='category',
                        title="Répartition des données par catégorie")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Temps d'activité vs données
    st.subheader("Relation Temps d'Activité / Consommation de Données")
    fig_scatter = px.scatter(usage_data, x='active_time', y='data_used_mb', 
                           color='category', size='data_used_mb',
                           hover_data=['platform'],
                           title="Temps actif vs Données consommées")
    fig_scatter.update_layout(xaxis_title="Temps actif (minutes)", 
                            yaxis_title="Données (MB)")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tableau détaillé
    st.subheader("Détails par Application")
    
    # Calcul de métriques supplémentaires
    usage_data['efficiency'] = usage_data['data_used_mb'] / usage_data['active_time']
    usage_data['efficiency'] = usage_data['efficiency'].round(2)
    
    # Formatage pour l'affichage
    display_data = usage_data[['platform', 'category', 'data_used_mb', 'active_time', 'efficiency']].copy()
    display_data.columns = ['Application', 'Catégorie', 'Données (MB)', 'Temps (min)', 'Efficacité (MB/min)']
    
    st.dataframe(display_data, use_container_width=True)
    
    # Recommandations basées sur l'usage
    st.subheader("Recommandations d'Optimisation")
    
    # Identifier les applications les plus consommatrices
    top_consumer = usage_data.loc[usage_data['data_used_mb'].idxmax()]
    low_efficiency = usage_data.loc[usage_data['efficiency'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning(f"**Application la plus consommatrice:**")
        st.write(f"{top_consumer['platform']}: {top_consumer['data_used_mb']:.0f} MB")
        st.write("💡 Considérez limiter l'usage ou optimiser les paramètres.")
    
    with col2:
        st.info(f"**Efficacité la plus faible:**")
        st.write(f"{low_efficiency['platform']}: {low_efficiency['efficiency']:.2f} MB/min")
        st.write("💡 Vérifiez les paramètres de qualité ou les extensions.")
    
    # Export des données
    if st.button("Exporter les données d'usage"):
        csv = usage_data.to_csv(index=False)
        st.download_button(
            label="Télécharger CSV",
            data=csv,
            file_name=f"usage_system_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Fonctions utilitaires supplémentaires
def display_media_preview(media_file):
    """Affiche un aperçu du média selon son type"""
    try:
        if media_file["media_type"] == "image":
            # Pour les images, on pourrait afficher une miniature
            st.write("📸 Image - Aperçu non disponible dans cette démo")
        elif media_file["media_type"] == "video":
            st.write("🎥 Vidéo - Aperçu non disponible dans cette démo")
        elif media_file["media_type"] == "audio":
            st.write("🎵 Audio - Aperçu non disponible dans cette démo")
        elif media_file["media_type"] == "text":
            st.write("📄 Texte - Aperçu non disponible dans cette démo")
    except Exception as e:
        st.write(f"Aperçu indisponible: {e}")

def get_extracted_data_preview(media_id):
    """Récupère un aperçu des données extraites"""
    try:
        # Cette fonction pourrait appeler un endpoint spécifique pour récupérer
        # les données extraites d'un média spécifique
        # Pour la démo, on retourne des données simulées
        return {
            "preview_available": True,
            "features_count": 45,
            "data_size": "2.3 MB",
            "extraction_time": "00:02:15"
        }
    except Exception:
        return {"preview_available": False}

def export_analysis_results(analysis_id, results):
    """Exporte les résultats d'analyse en différents formats"""
    try:
        # Préparer les données pour export
        export_data = {
            "analysis_id": analysis_id,
            "export_timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Format JSON
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Boutons de téléchargement
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Télécharger JSON",
                data=json_data,
                file_name=f"analysis_{analysis_id[:8]}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col2:
            # Convertir en CSV si possible
            try:
                if isinstance(results, dict) and "summary_statistics" in results:
                    df = pd.DataFrame.from_dict(results["summary_statistics"], orient='index')
                    csv_data = df.to_csv()
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv_data,
                        file_name=f"analysis_{analysis_id[:8]}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            except Exception:
                st.write("Export CSV non disponible")
                
    except Exception as e:
        st.error(f"Erreur lors de l'export: {e}")

def show_advanced_settings():
    """Affiche les paramètres avancés de l'application"""
    with st.sidebar.expander("Paramètres Avancés"):
        st.subheader("Configuration API")
        
        # Paramètres de timeout
        timeout_setting = st.slider("Timeout API (secondes)", 10, 120, 30)
        st.session_state.api_timeout = timeout_setting
        
        # Mode debug
        debug_mode = st.checkbox("Mode Debug", value=False)
        st.session_state.debug_mode = debug_mode
        
        # Paramètres d'affichage
        st.subheader("Affichage")
        max_items = st.slider("Max éléments par page", 5, 50, 20)
        st.session_state.max_items_per_page = max_items
        
        # Cache
        if st.button("Vider le cache"):
            st.cache_data.clear()
            st.success("Cache vidé!")

def show_help_documentation():
    """Affiche la documentation d'aide"""
    st.title("Documentation - Media Intelligence Platform")
    
    help_sections = {
        "Premiers Pas": """
        **Comment commencer:**
        1. Uploadez vos médias via la section 'Upload Media'
        2. Attendez que le traitement soit terminé (statut 'completed')
        3. Lancez des analyses depuis 'Mes Médias'
        4. Consultez les résultats dans 'Analyses'
        """,
        
        "Types de Médias Supportés": """
        **Images:** JPG, PNG, GIF, TIFF, BMP, WebP
        - Extraction: propriétés, features visuelles, objets détectés, texte (OCR)
        - Analyses: esthétique, couleurs dominantes, classification
        
        **Vidéos:** MP4, AVI, MOV, WMV, FLV, WebM, MKV  
        - Extraction: propriétés vidéo, analyse frames, features audio
        - Analyses: détection de scènes, analyse de mouvement, transcription
        
        **Audio:** MP3, WAV, FLAC, AAC, OGG, M4A
        - Extraction: features spectrales, temporelles, transcription
        - Analyses: émotion audio, features musicales
        
        **Texte:** TXT, MD, RTF, DOC, DOCX
        - Extraction: propriétés linguistiques, analyse sémantique
        - Analyses: sentiment, émotions, entités nommées
        """,
        
        "Analyses ML": """
        **Types d'études disponibles:**
        - Classification automatique
        - Clustering (regroupement)
        - Détection d'anomalies  
        - Analyse de similarité
        - Génération de contenu
        
        **Métriques de performance:**
        - Score Silhouette (clustering)
        - Précision/Rappel (classification)
        - Inertie (cohésion des clusters)
        """,
        
        "Marketplace": """
        **Vendre vos médias:**
        1. Vos médias doivent être traités (statut 'completed')
        2. Définissez un prix et une description
        3. Votre média devient disponible à l'achat
        
        **Acheter des médias:**
        - Parcourez les offres disponibles
        - Filtrez par type et prix
        - Contactez les vendeurs
        """
    }
    
    for section, content in help_sections.items():
        with st.expander(section, expanded=False):
            st.markdown(content)

def add_footer():
    """Ajoute un footer à l'application"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Media Intelligence Platform**")
        st.markdown("Analyse multimodale avancée")
    
    with col2:
        st.markdown("**Liens:**")
        st.markdown("- [API Docs](http://localhost:8005/docs)")
        st.markdown("- [AutoSci ML](http://localhost:8501)")
        st.markdown("- [Personal Data](http://localhost:8504)")
    
    with col3:
        st.markdown("**Support:**")
        st.markdown("Version 1.0.0")
        st.markdown(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d')}")

# Fonction principale mise à jour
def main():
    st.markdown('<h1 class="main-header">Media Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Afficher les paramètres avancés
    show_advanced_settings()
    
    # Sidebar avec navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Boutons de navigation vers autres plateformes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("AutoSci ML", use_container_width=True, type="secondary"):
                st.markdown(f'<meta http-equiv="refresh" content="0; url={AUTOSCI_URL}">', unsafe_allow_html=True)
        with col2:
            if st.button("Personal Data", use_container_width=True, type="secondary"):
                st.markdown(f'<meta http-equiv="refresh" content="0; url={PERSONAL_DATA_URL}">', unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation interne
        pages = ["Upload Media", "Mes Médias", "Analyses", "Études ML", "Marketplace", "Usage Système", "Documentation"]
        page = st.selectbox("Sections:", pages)
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID: `{st.session_state.user_id[:8]}...`")
        
        # Statut API
        health, error = call_media_api("/health")
        if health:
            st.success("API Media en ligne")
            if st.session_state.get("debug_mode", False):
                st.json(health)
        else:
            st.error("API Media hors ligne")
            if error and st.session_state.get("debug_mode", False):
                st.error(f"Détail: {error}")
        
        # Bouton d'aide rapide
        if st.button("❓ Aide", use_container_width=True):
            st.session_state.show_help = True
    
    # Afficher l'aide si demandée
    if st.session_state.get("show_help", False):
        show_help_documentation()
        if st.button("Fermer l'aide"):
            st.session_state.show_help = False
        return
    
    # Pages principales
    if page == "Upload Media":
        show_upload_page()
    elif page == "Mes Médias":
        show_media_management_page()
    elif page == "Analyses":
        show_analyses_page()
    elif page == "Études ML":
        show_studies_page()
    elif page == "Marketplace":
        show_marketplace_page()
    elif page == "Usage Système":
        show_system_usage_page()
    elif page == "Documentation":
        show_help_documentation()
    
    # Footer
    add_footer()

# Point d'entrée
if __name__ == "__main__":
    main()